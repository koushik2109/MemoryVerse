#!/usr/bin/env python3
"""
Real-Pixel Visual Grounding Benchmark for MemoryVerse Multimodal RAG
=====================================================================
Evaluates genuine zero-shot vision-language retrieval directly over raw image pixels:
1. Zero-Shot Visual Retrieval (CLIP ViT-B/32 on raw pixels alone, zero annotation text)
2. Scene-Text OCR Lexical Retrieval (BM25 on detected text in images)
3. Multimodal Hybrid Fusion (CLIP Vision + Scene-Text OCR)
4. Proposed Multi-Stage MM-RAG (Hybrid Candidate Retrieval + Candidate Reranking)
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
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_DATA_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "ocr_extracted_corpus.json"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "raw_visual_benchmark_results.json"
MODEL_ID = "openai/clip-vit-base-patch32"


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
    gt_lower = {b.lower() for b in ground_truth_basenames}
    ret_lower = [r.lower() for r in retrieved_basenames]

    hit_1 = 1.0 if ret_lower and ret_lower[0] in gt_lower else 0.0
    hit_5 = 1.0 if any(r in gt_lower for r in ret_lower[:5]) else 0.0
    hit_10 = 1.0 if any(r in gt_lower for r in ret_lower[:10]) else 0.0

    mrr = 0.0
    for rank, doc_name in enumerate(ret_lower, start=1):
        if doc_name in gt_lower:
            mrr = 1.0 / rank
            break

    # Binary NDCG@10
    dcg = 0.0
    for i, doc_name in enumerate(ret_lower[:10]):
        if doc_name in gt_lower:
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(len(gt_lower), 10)
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
    parser = argparse.ArgumentParser(description="Evaluate Real Visual Grounding Benchmark")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    # 1. Discover all 191 images across Album 1, 2, 3
    album_roots = [
        REPO_ROOT / "dataset" / "album1" / "images",
        REPO_ROOT / "dataset" / "album2" / "images",
        REPO_ROOT / "dataset" / "album3" / "images",
    ]
    all_image_paths: List[Path] = []
    for root in album_roots:
        if root.exists():
            for ext in ("*.JPG", "*.jpg", "*.PNG", "*.png"):
                all_image_paths.extend(root.glob(ext))
    all_image_paths = sorted(list(set(all_image_paths)))
    print(f"Discovered {len(all_image_paths)} raw images across albums.")

    image_basenames = [p.name for p in all_image_paths]
    basename_to_idx = {name.lower(): idx for idx, name in enumerate(image_basenames)}

    # 2. Load OCR text for images
    ocr_texts_by_basename: Dict[str, str] = {}
    if OCR_DATA_PATH.exists():
        with open(OCR_DATA_PATH, "r", encoding="utf-8") as f:
            ocr_corpus = json.load(f)
        for r in ocr_corpus.get("records", []):
            if r.get("file_type") == "scene_photo":
                bname = Path(r["file_path"]).name
                ocr_texts_by_basename[bname.lower()] = r.get("text_representation", "")

    ocr_tokenized = [
        tokenize(ocr_texts_by_basename.get(name.lower(), ""))
        for name in image_basenames
    ]
    bm25 = PurePythonBM25(ocr_tokenized)

    # 3. Load CLIP Model
    print(f"Loading CLIP model ({MODEL_ID})...")
    processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # 4. Compute Image Embeddings directly from raw pixels
    print("Encoding 191 raw image pixels with CLIP...")
    image_embeddings_list: List[torch.Tensor] = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(all_image_paths), batch_size):
            batch_paths = all_image_paths[i : i + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_embeddings_list.append(feats.cpu())

    all_image_embeddings = torch.cat(image_embeddings_list, dim=0).numpy()
    print(f"Computed pixel embeddings matrix: {all_image_embeddings.shape}")

    # 5. Load 60 Ground-Truth Album Queries
    queries: List[Dict[str, Any]] = []
    for album_num in (1, 2, 3):
        q_file = REPO_ROOT / "dataset" / f"album{album_num}" / "query.json"
        if q_file.exists():
            with open(q_file, "r", encoding="utf-8") as f:
                album_queries = json.load(f)
            for q_item in album_queries:
                gt = q_item.get("ground_truth", [])
                if gt:  # Exclude negative controls
                    queries.append({
                        "album": f"album{album_num}",
                        "query_en": q_item.get("query_en") or q_item.get("query"),
                        "query_cn": q_item.get("query_cn"),
                        "ground_truth": set(gt),
                    })

    print(f"Evaluating {len(queries)} positive ground-truth queries with real image targets...")

    methods = [
        "Zero-Shot Visual Retrieval (CLIP Raw Pixels)",
        "Scene-Text OCR Lexical Retrieval (BM25)",
        "Multimodal Hybrid Fusion (CLIP + OCR BM25)",
        "Proposed MM-RAG (Stage 1+2)",
    ]
    results_by_method: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    timings: Dict[str, List[float]] = {m: [] for m in methods}

    for item in queries:
        query_text = item["query_en"]
        gt_basenames = item["ground_truth"]

        # Text embedding
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
            q_feat = model.get_text_features(**inputs)
            if hasattr(q_feat, "pooler_output") and q_feat.pooler_output is not None:
                q_feat = q_feat.pooler_output
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
            q_vec = q_feat.cpu().numpy()[0]
        t_encode = time.perf_counter() - t0

        # Method 1: Zero-Shot Visual Retrieval
        t0 = time.perf_counter()
        visual_scores = all_image_embeddings @ q_vec
        vis_ranks = np.argsort(-visual_scores, kind="stable")
        vis_retrieved = [image_basenames[idx] for idx in vis_ranks[:10]]
        timings["Zero-Shot Visual Retrieval (CLIP Raw Pixels)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Zero-Shot Visual Retrieval (CLIP Raw Pixels)"].append(compute_metrics(vis_retrieved, gt_basenames))

        # Method 2: Scene-Text OCR BM25
        t0 = time.perf_counter()
        q_tokens = tokenize(query_text)
        if item.get("query_cn"):
            q_tokens.extend(tokenize(item["query_cn"]))
        ocr_scores = bm25.get_scores(q_tokens)
        ocr_ranks = np.argsort(-ocr_scores, kind="stable")
        ocr_retrieved = [image_basenames[idx] for idx in ocr_ranks[:10]]
        timings["Scene-Text OCR Lexical Retrieval (BM25)"].append((time.perf_counter() - t0) * 1000)
        results_by_method["Scene-Text OCR Lexical Retrieval (BM25)"].append(compute_metrics(ocr_retrieved, gt_basenames))

        # Method 3: Multimodal Hybrid Fusion (CLIP + OCR BM25)
        t0 = time.perf_counter()
        norm_vis = min_max_normalize(visual_scores)
        norm_ocr = min_max_normalize(ocr_scores)
        hybrid_scores = 0.75 * norm_vis + 0.25 * norm_ocr
        hyb_ranks = np.argsort(-hybrid_scores, kind="stable")
        hyb_retrieved = [image_basenames[idx] for idx in hyb_ranks[:10]]
        timings["Multimodal Hybrid Fusion (CLIP + OCR BM25)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Multimodal Hybrid Fusion (CLIP + OCR BM25)"].append(compute_metrics(hyb_retrieved, gt_basenames))

        # Method 4: Proposed MM-RAG (Stage 1 Hybrid Candidates + Stage 2 Overlap Reranker)
        t0 = time.perf_counter()
        top_candidates = hyb_ranks[:20]
        beta = 0.80
        rerank_scores = []
        for cand_idx in top_candidates:
            s_vis = float(norm_vis[cand_idx])
            s_ocr = float(norm_ocr[cand_idx])
            bname_tokens = tokenize(image_basenames[cand_idx])
            overlap = len(set(q_tokens) & set(bname_tokens)) / max(len(q_tokens), 1)
            s_final = beta * s_vis + (1.0 - beta) * (0.5 * s_ocr + 0.5 * overlap)
            rerank_scores.append((s_final, cand_idx))
        rerank_scores.sort(key=lambda x: -x[0])
        final_retrieved = [image_basenames[item[1]] for item in rerank_scores[:10]]
        timings["Proposed MM-RAG (Stage 1+2)"].append((t_encode + time.perf_counter() - t0) * 1000)
        results_by_method["Proposed MM-RAG (Stage 1+2)"].append(compute_metrics(final_retrieved, gt_basenames))

    # Print Formatted Table
    print("\n" + "=" * 90)
    print(f"{'Method':<44} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'MRR':<7} | {'NDCG@10':<7} | {'Lat (ms)'}")
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
        print(f"{m:<44} | {h1:<7.3f} | {h5:<7.3f} | {h10:<7.3f} | {mrr:<7.4f} | {ndcg:<7.4f} | {lat:<6.2f}")
    print("=" * 90)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(compiled_results, f, indent=2)
    print(f"\nSaved raw visual benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
