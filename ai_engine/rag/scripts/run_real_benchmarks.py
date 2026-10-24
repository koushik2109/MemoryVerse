"""
Real Empirical Benchmark Script for Multimodal RAG Paper
==========================================================
Evaluates real retrieval performance on unified_rag_dataset.jsonl:
1. BM25 (Sparse Inverted Index)
2. Dense Semantic Vector Search (all-MiniLM-L6-v2)
3. Hybrid Dense-Sparse (Convex Combination alpha = 0.7)
4. Hybrid Dense-Sparse (Reciprocal Rank Fusion RRF k=60)
5. Multi-Stage Pipeline (Hybrid Candidate Retrieval + Reranking)
6. Parameter sweep across alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
7. Modality breakdown: Images vs Audio
"""

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RAG_DIR / "data" / "unified_rag_dataset.jsonl"
OUTPUT_PATH = RAG_DIR / "data" / "real_benchmark_results.json"
MAX_QUERIES = 500
MAX_IMAGE_QUERIES = 350
RANDOM_SEED = 42
TOP_K_VALUES = [1, 5, 10]


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


def compute_recall(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Compute standard Recall@K over the complete relevant-document set."""
    if not ground_truth_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & ground_truth_ids) / len(ground_truth_ids)


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Return 1 when any relevant document appears in the top K, else 0."""
    return float(bool(set(retrieved_ids[:k]) & ground_truth_ids))


def ranked_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """Rank scores with a stable corpus-order tie break."""
    return np.argsort(-scores, kind="stable")[:limit]


def select_eval_queries(
    image_query_map: Dict[str, Set[str]],
    audio_category_map: Dict[str, Set[str]],
) -> Tuple[List[Tuple[str, Set[str], str]], Dict[str, int]]:
    """Select a deterministic, explicitly capped image/audio query mix."""
    rng = np.random.default_rng(RANDOM_SEED)
    image_queries = sorted(image_query_map)
    audio_categories = sorted(audio_category_map)

    # Preserve the historical preference for up to 350 image queries, while
    # treating MAX_QUERIES as a real cap rather than a display-only comment.
    audio_limit = min(len(audio_categories), MAX_QUERIES)
    selected_audio_categories = audio_categories[:audio_limit]
    image_limit = min(MAX_IMAGE_QUERIES, MAX_QUERIES - len(selected_audio_categories))
    if len(image_queries) > image_limit:
        selected_image_queries = sorted(
            rng.choice(image_queries, size=image_limit, replace=False).tolist()
        )
    else:
        selected_image_queries = image_queries[:image_limit]

    selected = [
        (query, image_query_map[query], "image")
        for query in selected_image_queries
    ]
    selected.extend(
        (f"{category} sound", audio_category_map[category], "audio")
        for category in selected_audio_categories
    )
    composition = {
        "image": len(selected_image_queries),
        "audio": len(selected_audio_categories),
    }
    return selected, composition


def main():
    print("=" * 70)
    print("Running Genuine Multimodal RAG Empirical Evaluation")
    print("=" * 70)

    # 1. Load Dataset once.  The benchmark identity is the corpus row, not the
    # human-readable item_id, because older datasets can contain basename
    # collisions across image sources.
    print(f"Loading corpus from {DATA_PATH}...")
    corpus = []
    image_query_map = defaultdict(set)
    audio_category_map = defaultdict(set)

    with open(DATA_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            record["doc_idx"] = idx
            corpus.append(record)

            modality = record.get("modality", "")
            doc_id = str(idx)

            if modality == "image":
                queries_en = record.get("metadata", {}).get("queries_en", [])
                for q in queries_en:
                    q_clean = q.strip()
                    if len(q_clean) > 3:
                        image_query_map[q_clean].add(doc_id)
            elif modality == "audio":
                cat = record.get("metadata", {}).get("category", "")
                if cat:
                    audio_category_map[cat].add(doc_id)

    print(f"Total corpus documents: {len(corpus):,}")
    print(f"Discovered {len(image_query_map)} unique image queries and {len(audio_category_map)} audio categories.")

    eval_queries, query_composition = select_eval_queries(image_query_map, audio_category_map)
    if not eval_queries:
        raise ValueError("No evaluation queries were discovered in the dataset")
    print(
        f"Selected {len(eval_queries)} evaluation queries "
        f"({query_composition['image']} Image, {query_composition['audio']} Audio; "
        f"seed={RANDOM_SEED})."
    )

    # 2. Build BM25 Index
    print("\nBuilding BM25 sparse lexical index...")
    corpus_tokens = [tokenize(doc.get("text_representation", "")) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)
    print("BM25 index built successfully.")

    # 3. Build Dense Embeddings
    print("\nEncoding corpus using all-MiniLM-L6-v2 dense embeddings...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [doc.get("text_representation", "") for doc in corpus]

    t0 = time.time()
    corpus_embeddings = encoder.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Encoded {len(corpus)} documents in {time.time() - t0:.2f}s.")

    # 4. Evaluation Loop
    methods = [
        "BM25 (Sparse Only)",
        "Dense Only (all-MiniLM)",
        "Hybrid (Convex alpha=0.7)",
        "Hybrid (RRF k=60)",
        "Multi-Stage (Hybrid + Cross-Scorer)",
    ]

    results = {
        m: {
            "HitRate@1": [],
            "HitRate@5": [],
            "HitRate@10": [],
            "MRR": [],
            "NDCG@10": [],
            "latency_ms": [],
        }
        for m in methods
    }
    modality_results = {
        "image": {m: {"HitRate@10": [], "MRR": []} for m in methods},
        "audio": {m: {"HitRate@10": [], "MRR": []} for m in methods},
    }

    # Alpha sweep results
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alpha_sweep = {a: {"HitRate@10": [], "MRR": []} for a in alpha_values}

    print(f"\nEvaluating {len(eval_queries)} queries across all retrieval pipelines...")

    for q_idx, (q_text, gt_set, modality) in enumerate(eval_queries):

        # ─── A. BM25 Search ───
        t_bm25_start = time.time()
        q_tokens = tokenize(q_text)
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_latency = (time.time() - t_bm25_start) * 1000.0

        bm25_top_indices = ranked_indices(bm25_scores, 30)
        bm25_retrieved_ids = [str(corpus[i]["doc_idx"]) for i in bm25_top_indices]

        # ─── B. Dense Search ───
        t_dense_start = time.time()
        q_embed = encoder.encode(q_text, normalize_embeddings=True)
        dense_scores = np.dot(corpus_embeddings, q_embed)
        dense_latency = (time.time() - t_dense_start) * 1000.0

        dense_top_indices = ranked_indices(dense_scores, 30)
        dense_retrieved_ids = [str(corpus[i]["doc_idx"]) for i in dense_top_indices]

        # ─── C. Hybrid Search (Convex alpha=0.7) ───
        t_hyb_start = time.time()
        # Min-max normalization
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        norm_dense = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        hybrid_scores = 0.7 * norm_dense + 0.3 * norm_bm25
        hyb_latency = dense_latency + bm25_latency + ((time.time() - t_hyb_start) * 1000.0)

        hybrid_top_indices = ranked_indices(hybrid_scores, 30)
        hybrid_retrieved_ids = [str(corpus[i]["doc_idx"]) for i in hybrid_top_indices]

        # ─── D. Hybrid Search (RRF k=60) ───
        t_rrf_start = time.time()
        rrf_scores = np.zeros(len(corpus))
        k_rrf = 60
        # RRF over top 100 of each
        for rank, idx in enumerate(ranked_indices(dense_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)
        for rank, idx in enumerate(ranked_indices(bm25_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)
        rrf_latency = dense_latency + bm25_latency + ((time.time() - t_rrf_start) * 1000.0)

        rrf_top_indices = ranked_indices(rrf_scores, 30)
        rrf_retrieved_ids = [str(corpus[i]["doc_idx"]) for i in rrf_top_indices]

        # ─── E. Multi-Stage (Hybrid Candidate + Cross-Scorer Reranker) ───
        t_rerank_start = time.time()
        # Top 30 candidate pool from RRF
        candidate_indices = rrf_top_indices[:30]
        # Fine-grained token overlap + dense agreement reranking score
        reranked_pool = []
        for idx in candidate_indices:
            doc = corpus[idx]
            d_text = doc.get("text_representation", "").lower()
            query_tokens = set(tokenize(q_text))
            document_tokens = set(tokenize(d_text))
            # Exact normalized token coverage bonus + dense similarity.
            exact_overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            cos_sim = float(dense_scores[idx])
            final_score = 0.65 * cos_sim + 0.35 * exact_overlap
            reranked_pool.append((final_score, str(doc["doc_idx"])))

        reranked_pool.sort(key=lambda x: (-x[0], x[1]))
        multistage_retrieved_ids = [x[1] for x in reranked_pool]
        multistage_latency = rrf_latency + ((time.time() - t_rerank_start) * 1000.0)

        # ─── Log Metrics for all methods ───
        candidates_map = {
            "BM25 (Sparse Only)": (bm25_retrieved_ids, bm25_latency),
            "Dense Only (all-MiniLM)": (dense_retrieved_ids, dense_latency),
            "Hybrid (Convex alpha=0.7)": (hybrid_retrieved_ids, hyb_latency),
            "Hybrid (RRF k=60)": (rrf_retrieved_ids, rrf_latency),
            "Multi-Stage (Hybrid + Cross-Scorer)": (multistage_retrieved_ids, multistage_latency),
        }

        for m_name, (r_ids, lat) in candidates_map.items():
            results[m_name]["HitRate@1"].append(compute_hit_rate(r_ids, gt_set, 1))
            results[m_name]["HitRate@5"].append(compute_hit_rate(r_ids, gt_set, 5))
            results[m_name]["HitRate@10"].append(compute_hit_rate(r_ids, gt_set, 10))
            results[m_name]["MRR"].append(compute_mrr(r_ids, gt_set))
            results[m_name]["NDCG@10"].append(compute_ndcg(r_ids, gt_set, 10))
            results[m_name]["latency_ms"].append(lat)

            modality_results[modality][m_name]["HitRate@10"].append(compute_hit_rate(r_ids, gt_set, 10))
            modality_results[modality][m_name]["MRR"].append(compute_mrr(r_ids, gt_set))

        # Alpha sweep computation
        for a in alpha_values:
            comb = a * norm_dense + (1.0 - a) * norm_bm25
            top_a_indices = ranked_indices(comb, 10)
            top_a_ids = [str(corpus[i]["doc_idx"]) for i in top_a_indices]
            alpha_sweep[a]["HitRate@10"].append(compute_hit_rate(top_a_ids, gt_set, 10))
            alpha_sweep[a]["MRR"].append(compute_mrr(top_a_ids, gt_set))

    # 5. Aggregate Results
    summary = {}
    print("\n" + "=" * 80)
    print(f"EMPIRICAL BENCHMARK RESULTS (Averaged over {len(eval_queries)} Test Queries)")
    print("=" * 80)
    print(f"{'Method':<36} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'MRR':<7} | {'NDCG@10':<8} | {'Latency':<8}")
    print("-" * 85)

    for m in methods:
        hit1 = float(np.mean(results[m]["HitRate@1"]))
        hit5 = float(np.mean(results[m]["HitRate@5"]))
        hit10 = float(np.mean(results[m]["HitRate@10"]))
        mrr = float(np.mean(results[m]["MRR"]))
        ndcg = float(np.mean(results[m]["NDCG@10"]))
        lat = float(np.mean(results[m]["latency_ms"]))

        summary[m] = {
            "HitRate@1": round(hit1, 4),
            "HitRate@5": round(hit5, 4),
            "HitRate@10": round(hit10, 4),
            "MRR": round(mrr, 4),
            "NDCG@10": round(ndcg, 4),
            "Latency_ms": round(lat, 2),
        }
        print(f"{m:<36} | {hit1:<7.3f} | {hit5:<7.3f} | {hit10:<7.3f} | {mrr:<7.3f} | {ndcg:<8.3f} | {lat:<6.1f} ms")

    # Alpha Sweep Summary
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: DENSE WEIGHT ALPHA SWEEP")
    print("=" * 60)
    alpha_summary = {}
    for a in alpha_values:
        hit10 = float(np.mean(alpha_sweep[a]["HitRate@10"]))
        mrr = float(np.mean(alpha_sweep[a]["MRR"]))
        alpha_summary[str(a)] = {"HitRate@10": round(hit10, 4), "MRR": round(mrr, 4)}
        print(f"Alpha = {a:<4} | HitRate@10 = {hit10:.4f} | MRR = {mrr:.4f}")

    # Modality Breakdown Summary
    print("\n" + "=" * 60)
    print("PER-MODALITY BREAKDOWN (HitRate@10)")
    print("=" * 60)
    mod_summary = {}
    for mod in ["image", "audio"]:
        mod_summary[mod] = {}
        print(f"\n--- Modality: {mod.upper()} ---")
        for m in methods:
            hit10 = float(np.mean(modality_results[mod][m]["HitRate@10"]))
            mrr = float(np.mean(modality_results[mod][m]["MRR"]))
            mod_summary[mod][m] = {"HitRate@10": round(hit10, 4), "MRR": round(mrr, 4)}
            print(f"{m:<36} | HitRate@10 = {hit10:.4f} | MRR = {mrr:.4f}")

    # Save to JSON
    final_output = {
        "dataset_size": len(corpus),
        "dataset_composition": dict(sorted(Counter(doc.get("modality", "unknown") for doc in corpus).items())),
        "evaluation_scope": {
            "proxy_only": True,
            "image_text_source": "metadata.queries_en annotation strings",
            "audio_text_source": "metadata.category ESC-50 label",
            "raw_media_inference": False,
            "document_records": 0,
        },
        "available_query_composition": {
            "image": len(image_query_map),
            "audio": len(audio_category_map),
        },
        "test_queries_count": len(eval_queries),
        "test_query_composition": query_composition,
        "query_selection": {
            "random_seed": RANDOM_SEED,
            "max_queries": MAX_QUERIES,
            "max_image_queries": MAX_IMAGE_QUERIES,
        },
        "retrieval_metric": "HitRate@K (at least one relevant document in top K)",
        "summary": summary,
        "alpha_sweep": alpha_summary,
        "modality_breakdown": mod_summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved empirical results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
