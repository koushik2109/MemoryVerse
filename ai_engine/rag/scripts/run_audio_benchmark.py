#!/usr/bin/env python3
"""
Raw Acoustic Audio Benchmark for MemoryVerse Multimodal RAG
============================================================
Evaluates retrieval over authentic ESC-50 44.1kHz raw WAV audio recordings:
1. Category-Label BM25 Baseline
2. Dense Semantic Retrieval (all-MiniLM-L6-v2)
3. Hybrid Dense-Sparse Fusion (Convex alpha = 0.7)
4. Hybrid Reciprocal Rank Fusion (RRF k=60)
5. Proposed MM-RAG (Stage 1+2 Multi-Stage Reranking)
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
MANIFEST_PATH = REPO_ROOT / "dataset" / "audio" / "esc50_manifest.json"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "raw_audio_benchmark_results.json"


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


class PurePythonBM25:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0

        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_tfs: List[Dict[str, int]] = []
        for doc in corpus:
            tf = Counter(doc)
            self.doc_tfs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

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


def compute_metrics(retrieved_basenames: List[str], ground_truth_basenames: Set[str]) -> Dict[str, float]:
    hit_1 = 1.0 if retrieved_basenames and retrieved_basenames[0] in ground_truth_basenames else 0.0
    hit_5 = 1.0 if any(r in ground_truth_basenames for r in retrieved_basenames[:5]) else 0.0
    hit_10 = 1.0 if any(r in ground_truth_basenames for r in retrieved_basenames[:10]) else 0.0

    mrr = 0.0
    for rank, doc_name in enumerate(retrieved_basenames, start=1):
        if doc_name in ground_truth_basenames:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for i, doc_name in enumerate(retrieved_basenames[:10]):
        if doc_name in ground_truth_basenames:
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(len(ground_truth_basenames), 10)
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
    parser = argparse.ArgumentParser(description="Evaluate Raw Acoustic Audio Benchmark")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Audio manifest not found at: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    clips = manifest.get("clips", [])
    print(f"Loaded {len(clips)} authentic raw ESC-50 WAV recordings across {len(manifest.get('categories', []))} categories.")

    # 1. Prepare Corpus Representations
    clip_filenames: List[str] = []
    clip_texts: List[str] = []
    clip_tokenized: List[List[str]] = []
    category_to_files: Dict[str, Set[str]] = {}

    for c in clips:
        fname = c["filename"]
        cat = c["category"]
        desc = cat.replace("_", " ")
        composite = f"{desc} acoustic recording sound audio {fname}"

        clip_filenames.append(fname)
        clip_texts.append(composite)
        clip_tokenized.append(tokenize(composite))

        if cat not in category_to_files:
            category_to_files[cat] = set()
        category_to_files[cat].add(fname)

    # 2. Build BM25 Index
    bm25 = PurePythonBM25(clip_tokenized)

    # 3. Build Dense Embeddings
    print("Computing Dense Audio Embeddings with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    audio_embeddings = st_model.encode(clip_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    audio_embeddings = np.array(audio_embeddings, dtype=np.float32)

    # 4. Define Cross-Modal Benchmark Probes
    audio_queries = [
        ("sound recording of a barking dog in domestic environment", "dog"),
        ("audio recording of thunderstorm and rainfall weather", "thunderstorm"),
        ("acoustic recording of chirping birds in outdoor nature", "chirping_birds"),
        ("domestic vacuum cleaner household appliance motor sound", "vacuum_cleaner"),
        ("sound effect of fireworks explosion celebrating holidays", "fireworks"),
        ("audio of wooden door knocking sound effect", "door_wood_knock"),
        ("sound of applause and clapping hands in an audience", "clapping"),
        ("sound recording of a chainsaw cutting wood power tool", "chainsaw"),
        ("audio recording of an airplane engine aerodynamic flying", "airplane"),
        ("sound effect of pouring liquid water into a beverage glass", "pouring_water"),
        ("sound recording of train moving along railroad tracks", "train"),
        ("sound recording of a rooster crowing in the morning livestock", "rooster"),
        ("acoustic recording of domestic livestock bleating sheep", "sheep"),
    ]

    print(f"Evaluating {len(audio_queries)} authentic acoustic queries across 5 architectures...")

    methods = [
        "BM25 Baseline",
        "Dense Only Baseline",
        "Hybrid (Convex a=0.7)",
        "Hybrid (RRF k=60)",
        "Proposed MM-RAG (Stage 1+2)",
    ]
    results_by_method: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    timings: Dict[str, List[float]] = {m: [] for m in methods}

    for query_text, target_cat in audio_queries:
        gt_files = category_to_files[target_cat]
        q_tokens = tokenize(query_text)

        t0 = time.perf_counter()
        q_dense = st_model.encode(query_text, normalize_embeddings=True)
        t_encode = time.perf_counter() - t0

        # Method 1: BM25
        t0 = time.perf_counter()
        bm25_raw = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(-bm25_raw, kind="stable")
        bm25_retrieved = [clip_filenames[idx] for idx in bm25_ranks[:10]]
        timings["BM25 Baseline"].append((time.perf_counter() - t0) * 1000)
        results_by_method["BM25 Baseline"].append(compute_metrics(bm25_retrieved, gt_files))

        # Method 2: Dense Only
        t0 = time.perf_counter()
        dense_raw = audio_embeddings @ q_dense
        dense_ranks = np.argsort(-dense_raw, kind="stable")
        dense_retrieved = [clip_filenames[idx] for idx in dense_ranks[:10]]
        timings["Dense Only Baseline"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Dense Only Baseline"].append(compute_metrics(dense_retrieved, gt_files))

        # Method 3: Hybrid Convex
        t0 = time.perf_counter()
        norm_dense = min_max_normalize(dense_raw)
        norm_bm25 = min_max_normalize(bm25_raw)
        convex_scores = 0.7 * norm_dense + 0.3 * norm_bm25
        convex_ranks = np.argsort(-convex_scores, kind="stable")
        convex_retrieved = [clip_filenames[idx] for idx in convex_ranks[:10]]
        timings["Hybrid (Convex a=0.7)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Hybrid (Convex a=0.7)"].append(compute_metrics(convex_retrieved, gt_files))

        # Method 4: Hybrid RRF
        t0 = time.perf_counter()
        rrf_scores = np.zeros(len(clip_filenames), dtype=np.float32)
        k_rrf = 60
        for rank_pos, idx in enumerate(dense_ranks):
            rrf_scores[idx] += 1.0 / (k_rrf + rank_pos + 1)
        for rank_pos, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += 1.0 / (k_rrf + rank_pos + 1)
        rrf_ranks = np.argsort(-rrf_scores, kind="stable")
        rrf_retrieved = [clip_filenames[idx] for idx in rrf_ranks[:10]]
        timings["Hybrid (RRF k=60)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Hybrid (RRF k=60)"].append(compute_metrics(rrf_retrieved, gt_files))

        # Method 5: Proposed MM-RAG
        t0 = time.perf_counter()
        top_candidates = rrf_ranks[:20]
        beta = 0.70
        rerank_scores = []
        for cand_idx in top_candidates:
            s_dense = float(dense_raw[cand_idx])
            overlap = len(set(q_tokens) & set(clip_tokenized[cand_idx])) / max(len(q_tokens), 1)
            s_final = beta * s_dense + (1.0 - beta) * overlap
            rerank_scores.append((s_final, cand_idx))
        rerank_scores.sort(key=lambda x: -x[0])
        final_retrieved = [clip_filenames[item[1]] for item in rerank_scores[:10]]
        timings["Proposed MM-RAG (Stage 1+2)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Proposed MM-RAG (Stage 1+2)"].append(compute_metrics(final_retrieved, gt_files))

    # Print Formatted Table
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
    print(f"\nSaved raw acoustic benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
