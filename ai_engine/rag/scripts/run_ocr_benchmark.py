#!/usr/bin/env python3
"""
Real Hybrid OCR Search Benchmark for Multimodal RAG
===================================================
Evaluates retrieval over real OCR-extracted scene-text and PDF documents:
1. BM25 (Sparse Inverted Index with bilingual CJK + English tokenization)
2. Dense Semantic Search (all-MiniLM-L6-v2)
3. Hybrid Dense-Sparse (Convex Combination alpha = 0.7)
4. Hybrid Dense-Sparse (Reciprocal Rank Fusion RRF k=60)
5. Proposed MM-RAG (Stage 1+2: RRF Candidate Retrieval + Dual-Signal Reranking)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_DATA_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "ocr_extracted_corpus.json"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "ocr_benchmark_results.json"


def tokenize(text: str) -> List[str]:
    """Tokenize supporting English/numeric words and individual Chinese characters."""
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


class PurePythonBM25:
    """Fast, dependency-free BM25Okapi implementation."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0

        # Term frequencies and document frequencies
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_tfs: List[Dict[str, int]] = []
        for doc in corpus:
            tf = Counter(doc)
            self.doc_tfs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        # Precompute IDF (Robertson-Spärck Jones)
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        query_terms = [t for t in query if t in self.idf]
        if not query_terms:
            return scores

        for term in query_terms:
            idf = self.idf[term]
            for idx, tf_dict in enumerate(self.doc_tfs):
                f = tf_dict.get(term, 0)
                if f > 0:
                    denom = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avgdl))
                    scores[idx] += idf * (f * (self.k1 + 1.0)) / denom
        return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_token_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    return len(q_set & d_set) / len(q_set)


def compute_metrics(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> Dict[str, float]:
    hit_1 = 1.0 if retrieved_ids and retrieved_ids[0] in ground_truth_ids else 0.0
    hit_5 = 1.0 if any(r in ground_truth_ids for r in retrieved_ids[:5]) else 0.0
    hit_10 = 1.0 if any(r in ground_truth_ids for r in retrieved_ids[:10]) else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            mrr = 1.0 / rank
            break

    # Binary NDCG@10
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:10]):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(len(ground_truth_ids), 10)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count)) if ideal_count > 0 else 1.0
    ndcg_10 = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return {
        "hit@1": hit_1,
        "hit@5": hit_5,
        "hit@10": hit_10,
        "mrr": mrr,
        "ndcg@10": ndcg_10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid OCR Search Benchmark")
    parser.add_argument("--ocr-path", type=Path, default=OCR_DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.ocr_path.exists():
        raise FileNotFoundError(f"OCR corpus not found at: {args.ocr_path}")

    with open(args.ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    print(f"Loaded {len(records)} OCR records.")

    # 1. Prepare Corpus Representations
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    corpus_tokenized: List[List[str]] = []

    for r in records:
        cid = r["id"]
        fp = r.get("file_path", "")
        # Build composite representation: extracted OCR text + filename/stem tokens
        extracted = r.get("text_representation", "").strip()
        stem_tokens = Path(fp).stem.replace("_", " ").replace("-", " ")
        composite_text = f"{extracted} {stem_tokens} {fp}".strip()

        corpus_ids.append(cid)
        corpus_texts.append(composite_text)
        corpus_tokenized.append(tokenize(composite_text))

    # 2. Build Inverted Index (BM25)
    print("Building BM25 lexical index...")
    bm25 = PurePythonBM25(corpus_tokenized)

    # 3. Build Dense Index (SentenceTransformer)
    print("Computing Dense Embeddings with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = st_model.encode(corpus_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

    # 4. Define Cross-Modal Benchmark Probes (Photos + Documents)
    eval_queries = [
        # Scene-Text Photographic Queries
        ("Baotu Spring ticket office 售票处 趵突泉", {"ocr_photo_IMG_1191"}, "scene_photo"),
        ("Shandong modern history museum anti-invasion 半殖民地化", {"ocr_photo_IMG_1386"}, "scene_photo"),
        ("where is china on the horizon 天涯何处是神州", {"ocr_photo_IMG_1386"}, "scene_photo"),
        ("ONYENAN museum souvenir", {"ocr_photo_IMG_1013"}, "scene_photo"),
        ("photos taken in Yucheng city IMG_1178", {"ocr_photo_IMG_1178", "ocr_photo_IMG_1179"}, "scene_photo"),
        ("Yantian District blue beach coast IMG_6457", {"ocr_photo_IMG_6457", "ocr_photo_IMG_6458"}, "scene_photo"),
        ("Shenzhen airport tarmac boarding gate IMG_5625", {"ocr_photo_IMG_5625"}, "scene_photo"),
        ("cat lying down in campus street IMG_4823", {"ocr_photo_IMG_4823", "ocr_photo_IMG_4824"}, "scene_photo"),
        ("museum souvenir gifts IMG_1011", {"ocr_photo_IMG_1011", "ocr_photo_IMG_1013"}, "scene_photo"),
        ("Christmas tree made of cups IMG_5243", {"ocr_photo_IMG_5243 2"}, "scene_photo"),
        ("makeup selfie November 16 afternoon IMG_1779", {"ocr_photo_IMG_1779", "ocr_photo_IMG_1780"}, "scene_photo"),
        ("posing near cave exit IMG_8870", {"ocr_photo_IMG_8870", "ocr_photo_IMG_8871"}, "scene_photo"),
        ("me wearing kimono traditional costume IMG_8637", {"ocr_photo_IMG_8637", "ocr_photo_IMG_8808"}, "scene_photo"),
        ("roast chicken with yellow green sauce IMG_5254", {"ocr_photo_IMG_5254", "ocr_photo_IMG_5255"}, "scene_photo"),
        ("winter photos from 2023 IMG_8527", {"ocr_photo_IMG_8527"}, "scene_photo"),

        # Scanned Document / Administrative PDF Queries
        ("Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Enhancing Interactive Querying with a Multimodal RAG System", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}, "pdf_doc"),
        ("Milvus vector database ImageBind embeddings", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}, "pdf_doc"),
        ("Candidate Reranking and Multi-Hop Query Bridging", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Patel LLaMA3 Multimodal RAG architecture", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}, "pdf_doc"),
        ("National Institute of Technology Warangal", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Otsu binarization and Hough transform skew correction", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Dual-Signal Candidate Reranking Overlap", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Tesseract OCR engine ICDAR", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Document analysis via LLaMA3 and text video integration", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}, "pdf_doc"),
        ("Reciprocal Rank Fusion RRF k=60", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Whisper ASR speech recognition", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Proxy consistency benchmark HitRate", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
        ("Enterprise Multimodal RAG pipeline", {"ocr_doc_Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3_p1"}, "pdf_doc"),
        ("Sentence-BERT Siamese network cosine similarity", {"ocr_doc_IEEE_Paper_MemoryVerse_p1"}, "pdf_doc"),
    ]

    print(f"Evaluating {len(eval_queries)} benchmark queries across 5 retrieval architectures...")

    methods = [
        "BM25 Baseline",
        "Dense Only Baseline",
        "Hybrid (Convex a=0.7)",
        "Hybrid (RRF k=60)",
        "Proposed MM-RAG (Stage 1+2)",
    ]
    results_by_method: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    timings: Dict[str, List[float]] = {m: [] for m in methods}

    for query_text, gt_ids, mod_type in eval_queries:
        q_tokens = tokenize(query_text)
        t0 = time.perf_counter()
        q_dense = st_model.encode(query_text, normalize_embeddings=True)
        t_dense_encode = time.perf_counter() - t0

        # Method 1: BM25
        t0 = time.perf_counter()
        bm25_raw = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(-bm25_raw, kind="stable")
        bm25_retrieved = [corpus_ids[idx] for idx in bm25_ranks[:10]]
        timings["BM25 Baseline"].append((time.perf_counter() - t0) * 1000)
        results_by_method["BM25 Baseline"].append(compute_metrics(bm25_retrieved, gt_ids))

        # Method 2: Dense Only
        t0 = time.perf_counter()
        dense_raw = doc_embeddings @ q_dense
        dense_ranks = np.argsort(-dense_raw, kind="stable")
        dense_retrieved = [corpus_ids[idx] for idx in dense_ranks[:10]]
        timings["Dense Only Baseline"].append((t_dense_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Dense Only Baseline"].append(compute_metrics(dense_retrieved, gt_ids))

        # Method 3: Hybrid Convex (alpha=0.7)
        t0 = time.perf_counter()
        norm_dense = min_max_normalize(dense_raw)
        norm_bm25 = min_max_normalize(bm25_raw)
        convex_scores = 0.7 * norm_dense + 0.3 * norm_bm25
        convex_ranks = np.argsort(-convex_scores, kind="stable")
        convex_retrieved = [corpus_ids[idx] for idx in convex_ranks[:10]]
        timings["Hybrid (Convex a=0.7)"].append((t_dense_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Hybrid (Convex a=0.7)"].append(compute_metrics(convex_retrieved, gt_ids))

        # Method 4: Hybrid RRF (k=60)
        t0 = time.perf_counter()
        rrf_scores = np.zeros(len(corpus_ids), dtype=np.float32)
        k_rrf = 60
        # dense ranks
        for rank_pos, idx in enumerate(dense_ranks):
            rrf_scores[idx] += 1.0 / (k_rrf + rank_pos + 1)
        # bm25 ranks
        for rank_pos, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += 1.0 / (k_rrf + rank_pos + 1)
        rrf_ranks = np.argsort(-rrf_scores, kind="stable")
        rrf_retrieved = [corpus_ids[idx] for idx in rrf_ranks[:10]]
        timings["Hybrid (RRF k=60)"].append((t_dense_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Hybrid (RRF k=60)"].append(compute_metrics(rrf_retrieved, gt_ids))

        # Method 5: Proposed MM-RAG (Stage 1 RRF candidates (top 30) + Stage 2 Reranker)
        t0 = time.perf_counter()
        top_candidates = rrf_ranks[:30]
        beta = 0.65
        rerank_scores = []
        for cand_idx in top_candidates:
            s_dense = float(dense_raw[cand_idx])
            s_overlap = compute_token_overlap(q_tokens, corpus_tokenized[cand_idx])
            s_final = beta * s_dense + (1.0 - beta) * s_overlap
            rerank_scores.append((s_final, cand_idx))
        rerank_scores.sort(key=lambda x: -x[0])
        final_retrieved = [corpus_ids[item[1]] for item in rerank_scores[:10]]
        timings["Proposed MM-RAG (Stage 1+2)"].append((t_dense_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Proposed MM-RAG (Stage 1+2)"].append(compute_metrics(final_retrieved, gt_ids))

    # Compile Table
    summary_table = []
    print("\n" + "=" * 90)
    print(f"{'Method':<32} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'MRR':<7} | {'NDCG@10':<7} | {'Lat (ms)'}")
    print("-" * 90)

    compiled_results = {}
    for m in methods:
        rows = results_by_method[m]
        h1 = float(np.mean([r["hit@1"] for r in rows]))
        h5 = float(np.mean([r["hit@5"] for r in rows]))
        h10 = float(np.mean([r["hit@10"] for r in rows]))
        mrr = float(np.mean([r["mrr"] for r in rows]))
        ndcg = float(np.mean([r["ndcg@10"] for r in rows]))
        lat = float(np.mean(timings[m]))

        compiled_results[m] = {
            "hit@1": round(h1, 4),
            "hit@5": round(h5, 4),
            "hit@10": round(h10, 4),
            "mrr": round(mrr, 4),
            "ndcg@10": round(ndcg, 4),
            "latency_ms": round(lat, 2),
        }
        print(f"{m:<32} | {h1:<7.3f} | {h5:<7.3f} | {h10:<7.3f} | {mrr:<7.4f} | {ndcg:<7.4f} | {lat:<6.2f}")
    print("=" * 90)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(compiled_results, f, indent=2)
    print(f"\nSaved benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
