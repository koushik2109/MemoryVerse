"""
Public Leakage-Free Image-Text Retrieval Benchmark (MS-COCO Karpathy Test Split)
================================================================================
Evaluates retrieval on standard public MS-COCO test split with ZERO annotation leakage:
- Corpus: 1,000 distinct images, indexed using human Caption 0
- Queries: 500 held-out test queries drawn from human Caption 3/4 (independent human paraphrases)
- Ground Truth: The exact corresponding image document ID

Compares:
1. BM25 (Sparse Inverted Index)
2. Dense Only (all-MiniLM-L6-v2)
3. Hybrid (Convex alpha=0.7)
4. Hybrid (RRF k=60)
5. Proposed MM-RAG (Dual-Signal Reranker)
6. Trained Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

Metrics:
- Hit@1, Hit@5, Hit@10
- MRR
- NDCG@10
- Latency (ms)
- 95% Bootstrap Confidence Intervals
"""

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

RAG_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RAG_DIR / "data" / "coco_karpathy_test.json"
OUTPUT_PATH = RAG_DIR / "data" / "coco_leakage_free_results.json"
RANDOM_SEED = 42
CORPUS_SIZE = 1000
QUERY_COUNT = 500
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    relevances = [1 if doc_id in ground_truth_ids else 0 for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = [1] * min(len(ground_truth_ids), k)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & ground_truth_ids))


def ranked_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:limit]


def compute_bootstrap_ci(scores: List[float], n_resamples: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def main():
    print("=" * 80)
    print("Public Leakage-Free Image-Text Retrieval Benchmark (MS-COCO)")
    print("=" * 80)

    with open(DATA_PATH, encoding="utf-8") as f:
        coco_records = json.load(f)
    print(f"Loaded {len(coco_records)} total records from MS-COCO Karpathy split.")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(coco_records), size=CORPUS_SIZE, replace=False)
    selected_records = [coco_records[i] for i in indices]

    # Index: Caption 0 as document representation
    corpus = []
    for doc_idx, item in enumerate(selected_records):
        doc_id = str(doc_idx)
        # Use first caption as index document
        index_caption = item["caption"][0]
        corpus.append({
            "doc_id": doc_id,
            "image": item["image"],
            "text": index_caption,
            "all_captions": item["caption"],
        })

    # Queries: Use Caption 3 as held-out paraphrased query
    query_indices = rng.choice(CORPUS_SIZE, size=QUERY_COUNT, replace=False)
    queries = []
    for q_idx in query_indices:
        item = corpus[q_idx]
        held_out_caption = item["all_captions"][3] if len(item["all_captions"]) > 3 else item["all_captions"][-1]
        queries.append({
            "query": held_out_caption,
            "ground_truth_doc_id": item["doc_id"],
            "index_caption": item["text"],
        })

    print(f"Constructed benchmark: {len(corpus)} index documents, {len(queries)} held-out queries.")
    print("Example pairing:")
    print(f"  [Index Doc]  : {queries[0]['index_caption']}")
    print(f"  [Held-Out Q] : {queries[0]['query']}")

    # Build BM25
    print("\nBuilding BM25 sparse index...")
    corpus_tokens = [tokenize(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    # Build Dense Embeddings
    print("Encoding dense corpus with all-MiniLM-L6-v2...")
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_embeddings = dense_encoder.encode(corpus_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Load Cross-Encoder
    print(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    methods = [
        "BM25 (Sparse Only)",
        "Dense Only (all-MiniLM)",
        "Hybrid (Convex alpha=0.7)",
        "Hybrid (RRF k=60)",
        "Proposed MM-RAG (Dual-Signal Reranker)",
        "Trained Cross-Encoder (ms-marco-MiniLM)",
    ]

    metrics = {m: {"Hit@1": [], "Hit@5": [], "Hit@10": [], "MRR": [], "NDCG@10": [], "latency_ms": []} for m in methods}

    print(f"\nEvaluating {len(queries)} queries across {len(methods)} retrieval architectures...")
    for q in queries:
        q_text = q["query"]
        gt_set = {q["ground_truth_doc_id"]}
        q_tokens = tokenize(q_text)

        # 1. BM25
        t0 = time.time()
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_lat = (time.time() - t0) * 1000.0
        bm25_top = ranked_indices(bm25_scores, 30)
        bm25_ids = [corpus[i]["doc_id"] for i in bm25_top]

        # 2. Dense
        t0 = time.time()
        q_emb = dense_encoder.encode([q_text], normalize_embeddings=True)[0]
        dense_scores = np.dot(corpus_embeddings, q_emb)
        dense_lat = (time.time() - t0) * 1000.0
        dense_top = ranked_indices(dense_scores, 30)
        dense_ids = [corpus[i]["doc_id"] for i in dense_top]

        # 3. Hybrid Convex
        t0 = time.time()
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        norm_b = (bm25_scores - b_min) / (b_max - b_min + 1e-9)
        d_min, d_max = dense_scores.min(), dense_scores.max()
        norm_d = (dense_scores - d_min) / (d_max - d_min + 1e-9)
        conv_scores = 0.7 * norm_d + 0.3 * norm_b
        conv_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        conv_top = ranked_indices(conv_scores, 30)
        conv_ids = [corpus[i]["doc_id"] for i in conv_top]

        # 4. RRF
        t0 = time.time()
        rrf_scores = np.zeros(len(corpus))
        for rank, idx in enumerate(ranked_indices(dense_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(ranked_indices(bm25_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        rrf_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        rrf_top = ranked_indices(rrf_scores, 30)
        rrf_ids = [corpus[i]["doc_id"] for i in rrf_top]

        # 5. Proposed MM-RAG Dual-Signal Reranker (from top 30 RRF)
        t0 = time.time()
        dual_pool = []
        q_token_set = set(q_tokens)
        for idx in rrf_top[:30]:
            d_tokens = set(tokenize(corpus[idx]["text"]))
            overlap = len(q_token_set & d_tokens) / max(1, len(q_token_set))
            cos_sim = float(dense_scores[idx])
            score = 0.65 * cos_sim + 0.35 * overlap
            dual_pool.append((score, corpus[idx]["doc_id"]))
        dual_pool.sort(key=lambda x: (-x[0], x[1]))
        dual_lat = rrf_lat + ((time.time() - t0) * 1000.0)
        dual_ids = [x[1] for x in dual_pool]

        # 6. Trained Cross-Encoder Reranker (from top 30 RRF)
        t0 = time.time()
        candidate_docs = [corpus[idx]["text"] for idx in rrf_top[:30]]
        ce_pairs = [[q_text, d_text] for d_text in candidate_docs]
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        ce_pool = [(float(score), corpus[idx]["doc_id"]) for score, idx in zip(ce_scores, rrf_top[:30])]
        ce_pool.sort(key=lambda x: (-x[0], x[1]))
        ce_lat = rrf_lat + ((time.time() - t0) * 1000.0)
        ce_ids = [x[1] for x in ce_pool]

        id_map = {
            "BM25 (Sparse Only)": (bm25_ids, bm25_lat),
            "Dense Only (all-MiniLM)": (dense_ids, dense_lat),
            "Hybrid (Convex alpha=0.7)": (conv_ids, conv_lat),
            "Hybrid (RRF k=60)": (rrf_ids, rrf_lat),
            "Proposed MM-RAG (Dual-Signal Reranker)": (dual_ids, dual_lat),
            "Trained Cross-Encoder (ms-marco-MiniLM)": (ce_ids, ce_lat),
        }

        for m, (r_ids, lat) in id_map.items():
            metrics[m]["Hit@1"].append(compute_hit_rate(r_ids, gt_set, 1))
            metrics[m]["Hit@5"].append(compute_hit_rate(r_ids, gt_set, 5))
            metrics[m]["Hit@10"].append(compute_hit_rate(r_ids, gt_set, 10))
            metrics[m]["MRR"].append(compute_mrr(r_ids, gt_set))
            metrics[m]["NDCG@10"].append(compute_ndcg(r_ids, gt_set, 10))
            metrics[m]["latency_ms"].append(lat)

    summary = {}
    print("\n" + "=" * 95)
    print(f"MS-COCO LEAKAGE-FREE BENCHMARK RESULTS ({len(queries)} Paraphrased Queries, N={len(corpus)} Images)")
    print("=" * 95)
    print(f"{'Method':<42} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'MRR':<7} | {'NDCG@10':<8} | {'Latency':<8}")
    print("-" * 95)

    for m in methods:
        h1 = float(np.mean(metrics[m]["Hit@1"]))
        h5 = float(np.mean(metrics[m]["Hit@5"]))
        h10 = float(np.mean(metrics[m]["Hit@10"]))
        mrr = float(np.mean(metrics[m]["MRR"]))
        ndcg = float(np.mean(metrics[m]["NDCG@10"]))
        lat = float(np.mean(metrics[m]["latency_ms"]))
        ci_h10 = compute_bootstrap_ci(metrics[m]["Hit@10"])
        ci_mrr = compute_bootstrap_ci(metrics[m]["MRR"])
        summary[m] = {
            "Hit@1": round(h1, 4),
            "Hit@5": round(h5, 4),
            "Hit@10": round(h10, 4),
            "Hit@10_95CI": [round(ci_h10[0], 4), round(ci_h10[1], 4)],
            "MRR": round(mrr, 4),
            "MRR_95CI": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
            "NDCG@10": round(ndcg, 4),
            "latency_ms": round(lat, 2),
        }
        print(f"{m:<42} | {h1:<7.4f} | {h5:<7.4f} | {h10:<7.4f} | {mrr:<7.4f} | {ndcg:<8.4f} | {lat:<8.2f}ms")

    output = {
        "metadata": {
            "benchmark": "MS-COCO Karpathy Test Split (Zero-Leakage)",
            "corpus_size": len(corpus),
            "query_count": len(queries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": RANDOM_SEED,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
        },
        "results": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll MS-COCO leakage-free results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
