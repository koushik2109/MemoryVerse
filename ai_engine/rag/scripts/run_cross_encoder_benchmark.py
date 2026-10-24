"""
Neural Cross-Encoder Baseline Benchmark for MM-RAG
===================================================
Compares the lightweight Dual-Signal Reranker (Stage 1+2) against a pre-trained
Neural Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) across:
1. Tier-1: 334-query Proxy Consistency Benchmark (unified_rag_dataset.jsonl)
2. Tier-2: 30-probe Scene-Text & Document OCR Benchmark (ocr_extracted_corpus.json)

Reports:
- Hit@1, Hit@5, Hit@10
- MRR
- NDCG@10
- Latency (ms per query)
- 95% Bootstrap Confidence Intervals
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RAG_DIR / "data" / "unified_rag_dataset.jsonl"
OCR_DATA_PATH = RAG_DIR / "data" / "ocr_extracted_corpus.json"
OUTPUT_PATH = RAG_DIR / "data" / "cross_encoder_benchmark_results.json"
MAX_QUERIES = 500
MAX_IMAGE_QUERIES = 350
RANDOM_SEED = 42
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


def select_eval_queries(
    image_query_map: Dict[str, Set[str]],
    audio_category_map: Dict[str, Set[str]],
) -> Tuple[List[Tuple[str, Set[str], str]], Dict[str, int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    image_queries = sorted(image_query_map)
    audio_categories = sorted(audio_category_map)

    audio_limit = min(len(audio_categories), MAX_QUERIES)
    selected_audio_categories = audio_categories[:audio_limit]
    image_limit = min(MAX_IMAGE_QUERIES, MAX_QUERIES - len(selected_audio_categories))
    if len(image_queries) > image_limit:
        selected_image_queries = sorted(
            rng.choice(image_queries, size=image_limit, replace=False).tolist()
        )
    else:
        selected_image_queries = image_queries

    eval_queries = []
    for q in selected_image_queries:
        eval_queries.append((q, image_query_map[q], "image"))
    for cat in selected_audio_categories:
        eval_queries.append((cat, audio_category_map[cat], "audio"))

    return eval_queries, {"image": len(selected_image_queries), "audio": len(selected_audio_categories)}


def run_tier1_benchmark(cross_encoder: CrossEncoder) -> Dict[str, Any]:
    print("\n--- Running Tier-1 Proxy Benchmark with Cross-Encoder ---")
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
                for q in record.get("metadata", {}).get("queries_en", []):
                    q_clean = q.strip()
                    if len(q_clean) > 3:
                        image_query_map[q_clean].add(doc_id)
            elif modality == "audio":
                cat = record.get("metadata", {}).get("category", "")
                if cat:
                    audio_category_map[cat].add(doc_id)

    eval_queries, query_composition = select_eval_queries(image_query_map, audio_category_map)
    print(f"Loaded {len(corpus):,} corpus rows, {len(eval_queries)} queries.")

    print("Building BM25 index...")
    corpus_tokens = [tokenize(doc.get("text_representation", "")) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    print("Encoding dense corpus...")
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [doc.get("text_representation", "") for doc in corpus]
    corpus_embeddings = dense_encoder.encode(
        corpus_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )

    methods = [
        "BM25 (Sparse Only)",
        "Dense Only (all-MiniLM)",
        "Hybrid (Convex alpha=0.7)",
        "Hybrid (RRF k=60)",
        "Proposed MM-RAG (Dual-Signal Reranker)",
        "Trained Cross-Encoder (ms-marco-MiniLM)",
    ]

    metrics = {m: {"Hit@1": [], "Hit@5": [], "Hit@10": [], "MRR": [], "NDCG@10": [], "latency_ms": []} for m in methods}

    print(f"Evaluating {len(eval_queries)} queries across methods...")
    for q_idx, (q_text, gt_set, modality) in enumerate(eval_queries):
        # 1. BM25
        t0 = time.time()
        q_tokens = tokenize(q_text)
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_lat = (time.time() - t0) * 1000.0
        bm25_top = ranked_indices(bm25_scores, 30)
        bm25_ids = [str(corpus[i]["doc_idx"]) for i in bm25_top]

        # 2. Dense
        t0 = time.time()
        q_emb = dense_encoder.encode([q_text], normalize_embeddings=True)[0]
        dense_scores = np.dot(corpus_embeddings, q_emb)
        dense_lat = (time.time() - t0) * 1000.0
        dense_top = ranked_indices(dense_scores, 30)
        dense_ids = [str(corpus[i]["doc_idx"]) for i in dense_top]

        # 3. Hybrid Convex
        t0 = time.time()
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        norm_bm25 = (bm25_scores - b_min) / (b_max - b_min + 1e-9)
        d_min, d_max = dense_scores.min(), dense_scores.max()
        norm_dense = (dense_scores - d_min) / (d_max - d_min + 1e-9)
        conv_scores = 0.7 * norm_dense + 0.3 * norm_bm25
        conv_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        conv_top = ranked_indices(conv_scores, 30)
        conv_ids = [str(corpus[i]["doc_idx"]) for i in conv_top]

        # 4. RRF
        t0 = time.time()
        rrf_scores = np.zeros(len(corpus))
        for rank, idx in enumerate(ranked_indices(dense_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(ranked_indices(bm25_scores, 100), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        rrf_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        rrf_top = ranked_indices(rrf_scores, 30)
        rrf_ids = [str(corpus[i]["doc_idx"]) for i in rrf_top]

        # 5. Proposed Dual-Signal Reranker (from top 30 RRF)
        t0 = time.time()
        dual_pool = []
        q_token_set = set(q_tokens)
        for idx in rrf_top[:30]:
            d_tokens = set(tokenize(corpus[idx].get("text_representation", "")))
            overlap = len(q_token_set & d_tokens) / max(1, len(q_token_set))
            cos_sim = float(dense_scores[idx])
            score = 0.65 * cos_sim + 0.35 * overlap
            dual_pool.append((score, str(corpus[idx]["doc_idx"])))
        dual_pool.sort(key=lambda x: (-x[0], x[1]))
        dual_lat = rrf_lat + ((time.time() - t0) * 1000.0)
        dual_ids = [x[1] for x in dual_pool]

        # 6. Neural Cross-Encoder Reranker (from top 30 RRF)
        t0 = time.time()
        candidate_docs = [corpus[idx].get("text_representation", "") for idx in rrf_top[:30]]
        ce_pairs = [[q_text, d_text] for d_text in candidate_docs]
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        ce_pool = [(float(score), str(corpus[idx]["doc_idx"])) for score, idx in zip(ce_scores, rrf_top[:30])]
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
    print(f"TIER-1 RESULTS WITH CROSS-ENCODER ({len(eval_queries)} Queries)")
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

    return summary


def run_tier2_ocr_benchmark(cross_encoder: CrossEncoder) -> Dict[str, Any]:
    print("\n--- Running Tier-2 Scene-Text & Document OCR with Cross-Encoder ---")
    with open(OCR_DATA_PATH, encoding="utf-8") as f:
        ocr_data = json.load(f)

    records = ocr_data.get("records", [])
    corpus = []
    for r in records:
        cid = r["id"]
        fp = r.get("file_path", "")
        extracted = r.get("text_representation", "").strip()
        stem_tokens = Path(fp).stem.replace("_", " ").replace("-", " ")
        composite_text = f"{extracted} {stem_tokens} {fp}".strip()
        corpus.append({"doc_id": cid, "text": composite_text})

    queries = [
        ("Baotu Spring ticket office 售票处 趵突泉", {"ocr_photo_IMG_1191"}),
        ("Shandong modern history museum anti-invasion 半殖民地化", {"ocr_photo_IMG_1386"}),
        ("where is china on the horizon 天涯何处是神州", {"ocr_photo_IMG_1386"}),
        ("ONYENAN museum souvenir", {"ocr_photo_IMG_1013"}),
        ("photos taken in Yucheng city IMG_1178", {"ocr_photo_IMG_1178", "ocr_photo_IMG_1179"}),
        ("Yantian District blue beach coast IMG_6457", {"ocr_photo_IMG_6457", "ocr_photo_IMG_6458"}),
        ("Shenzhen airport tarmac boarding gate IMG_5625", {"ocr_photo_IMG_5625"}),
        ("cat lying down in campus street IMG_4823", {"ocr_photo_IMG_4823", "ocr_photo_IMG_4824"}),
        ("museum souvenir gifts IMG_1011", {"ocr_photo_IMG_1011", "ocr_photo_IMG_1013"}),
        ("Christmas tree made of cups IMG_5243", {"ocr_photo_IMG_5243 2"}),
        ("makeup selfie November 16 afternoon IMG_1779", {"ocr_photo_IMG_1779", "ocr_photo_IMG_1780"}),
        ("posing near cave exit IMG_8870", {"ocr_photo_IMG_8870", "ocr_photo_IMG_8871"}),
        ("me wearing kimono traditional costume IMG_8637", {"ocr_photo_IMG_8637", "ocr_photo_IMG_8808"}),
        ("roast chicken with yellow green sauce IMG_5254", {"ocr_photo_IMG_5254", "ocr_photo_IMG_5255"}),
        ("winter photos from 2023 IMG_8527", {"ocr_photo_IMG_8527"}),
        ("Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Enhancing Interactive Querying with a Multimodal RAG System", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Milvus vector database ImageBind embeddings", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Candidate Reranking and Multi-Hop Query Bridging", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Patel LLaMA3 Multimodal RAG architecture", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("National Institute of Technology Warangal", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Otsu binarization and Hough transform skew correction", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Dual-Signal Candidate Reranking Overlap", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Tesseract OCR engine ICDAR", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Document analysis via LLaMA3 and text video integration", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Reciprocal Rank Fusion RRF k=60", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Whisper ASR speech recognition", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Proxy consistency benchmark HitRate", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
        ("Enterprise Multimodal RAG pipeline", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}),
        ("Sentence-BERT Siamese network cosine similarity", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}),
    ]
    print(f"Loaded {len(corpus)} OCR documents, {len(queries)} probes.")

    corpus_tokens = [tokenize(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_embeddings = dense_encoder.encode(corpus_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)


    methods = [
        "BM25 (Sparse Only)",
        "Dense Only (all-MiniLM)",
        "Hybrid (Convex alpha=0.7)",
        "Hybrid (RRF k=60)",
        "Proposed MM-RAG (Dual-Signal Reranker)",
        "Trained Cross-Encoder (ms-marco-MiniLM)",
    ]

    metrics = {m: {"Hit@1": [], "Hit@5": [], "Hit@10": [], "MRR": [], "NDCG@10": [], "latency_ms": []} for m in methods}

    for q_text, gt_set in queries:


        # BM25
        t0 = time.time()
        q_tokens = tokenize(q_text)
        bm25_scores = np.array(bm25.get_scores(q_tokens))
        bm25_lat = (time.time() - t0) * 1000.0
        bm25_top = ranked_indices(bm25_scores, 20)
        bm25_ids = [corpus[i]["doc_id"] for i in bm25_top]

        # Dense
        t0 = time.time()
        q_emb = dense_encoder.encode([q_text], normalize_embeddings=True)[0]
        dense_scores = np.dot(corpus_embeddings, q_emb)
        dense_lat = (time.time() - t0) * 1000.0
        dense_top = ranked_indices(dense_scores, 20)
        dense_ids = [corpus[i]["doc_id"] for i in dense_top]

        # Convex
        t0 = time.time()
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        norm_b = (bm25_scores - b_min) / (b_max - b_min + 1e-9)
        d_min, d_max = dense_scores.min(), dense_scores.max()
        norm_d = (dense_scores - d_min) / (d_max - d_min + 1e-9)
        conv_scores = 0.7 * norm_d + 0.3 * norm_b
        conv_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        conv_top = ranked_indices(conv_scores, 20)
        conv_ids = [corpus[i]["doc_id"] for i in conv_top]

        # RRF
        t0 = time.time()
        rrf_scores = np.zeros(len(corpus))
        for rank, idx in enumerate(ranked_indices(dense_scores, 20), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(ranked_indices(bm25_scores, 20), start=1):
            rrf_scores[idx] += 1.0 / (60 + rank)
        rrf_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)
        rrf_top = ranked_indices(rrf_scores, 20)
        rrf_ids = [corpus[i]["doc_id"] for i in rrf_top]

        # Proposed MM-RAG Dual-Signal Reranker (from top 20 RRF)
        t0 = time.time()
        dual_pool = []
        q_token_set = set(q_tokens)
        for idx in rrf_top[:20]:
            d_tokens = set(tokenize(corpus[idx]["text"]))
            overlap = len(q_token_set & d_tokens) / max(1, len(q_token_set))
            cos_sim = float(dense_scores[idx])
            score = 0.65 * cos_sim + 0.35 * overlap
            dual_pool.append((score, corpus[idx]["doc_id"]))
        dual_pool.sort(key=lambda x: (-x[0], x[1]))
        dual_lat = rrf_lat + ((time.time() - t0) * 1000.0)
        dual_ids = [x[1] for x in dual_pool]

        # Trained Cross-Encoder Reranker (from top 20 RRF)
        t0 = time.time()
        candidate_docs = [corpus[idx]["text"] for idx in rrf_top[:20]]
        ce_pairs = [[q_text, d_text] for d_text in candidate_docs]
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        ce_pool = [(float(score), corpus[idx]["doc_id"]) for score, idx in zip(ce_scores, rrf_top[:20])]
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
    print(f"TIER-2 SCENE-TEXT & DOCUMENT OCR RESULTS ({len(queries)} Probes)")
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

    return summary


def main():
    print(f"Loading Cross-Encoder model: {CROSS_ENCODER_MODEL}...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print("Cross-Encoder loaded successfully.")

    tier1_results = run_tier1_benchmark(cross_encoder)
    tier2_ocr_results = run_tier2_ocr_benchmark(cross_encoder)

    output = {
        "metadata": {
            "model": CROSS_ENCODER_MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": RANDOM_SEED,
        },
        "tier1_proxy_benchmark": tier1_results,
        "tier2_ocr_benchmark": tier2_ocr_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll Cross-Encoder benchmark results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
