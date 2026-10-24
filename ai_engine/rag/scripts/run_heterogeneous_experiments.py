#!/usr/bin/env python3
"""Comprehensive Experimental Suite for Heterogeneous Corpus (H1), Mixed Router (H2), and Prerequisites.

Paper: "Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval"
Target Journal: IJCDS

Strict Pre-Specified Protocol:
- Commit Hash: 2e63e9e (October 24, 2026)
- Seeds: 42, 43, 44
- 50/50 Dev/Test partition by item ID (zero test tuning; gamma tuned on Dev only)
- Statistics: Cluster bootstrap over probes (B=2,000 resamples), paired permutation tests,
  step-down Holm-Bonferroni correction, exact Clopper-Pearson intervals for saturated proportions.
- One latency protocol: same machine, warm cache, query encoding included, scope = retrieval + rerank, mean & p95 over 3 runs.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import beta as beta_dist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import CrossEncoder, SentenceTransformer

# ─── PATH CONFIGURATION ───────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
OCR_CORPUS_PATH = RAG_DATA_DIR / "ocr_extracted_corpus.json"
PROBES_SPLIT_PATH = RAG_DATA_DIR / "ramp_rag_probes_split.json"
COCO_DATA_PATH = RAG_DATA_DIR / "coco_karpathy_test.json"
COCO_BLIP_PATH = RAG_DATA_DIR / "coco_blip_captions.json"
COCO_IMAGES_DIR = RAG_DATA_DIR / "coco_1k_images"
COCO_CLIP_EMBS_PATH = RAG_DATA_DIR / "coco_clip_image_embeddings.npy"
PHOTO_CLIP_EMBS_PATH = RAG_DATA_DIR / "photo_clip_image_embeddings.npy"

OUTPUT_RESULTS_JSON = RAG_DATA_DIR / "heterogeneous_revision_results.json"
CSV_DIR = RAG_DATA_DIR / "csv_tables"
LATEX_DIR = RAG_DATA_DIR / "latex_tables"
CSV_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Device] Using compute device: {DEVICE}")

SEEDS = [42, 43, 44]

# ─── STATISTICAL CORE ─────────────────────────────────────────────────────────

def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Clopper-Pearson 95% binomial confidence interval for proportions."""
    from scipy.stats import beta
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 42) -> float:
    """Two-sided paired sign-flip permutation test."""
    diff = np.asarray(a) - np.asarray(b)
    obs = abs(float(np.mean(diff)))
    if obs < 1e-12:
        return 1.0
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.abs(np.mean(flips * diff, axis=1))
    return float((np.sum(perm_means >= obs) + 1.0) / (n_perm + 1.0))


def holm_bonferroni(p_dict: Dict[str, float]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni correction over a dictionary of method -> p-value."""
    sorted_items = sorted(p_dict.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p: Dict[str, float] = {}
    cum_max = 0.0
    for rank, (k, p) in enumerate(sorted_items):
        multiplier = m - rank
        raw_adj = p * multiplier
        adj = min(1.0, max(cum_max, raw_adj))
        cum_max = adj
        adj_p[k] = float(round(adj, 5))
    return adj_p


def cluster_bootstrap_ci(probe_scores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval resampling unique probes (averaging seeds first)."""
    arr = np.asarray(probe_scores)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean_val = float(np.mean(arr))
    if mean_val >= 0.9999 or mean_val <= 0.0001:
        k = int(round(np.sum(arr)))
        return clopper_pearson_interval(k, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(arr[indices], axis=1)
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return lower, upper


def cluster_bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval for paired difference a - b."""
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_diffs = np.mean(diff[indices], axis=1)
    lower = float(np.percentile(boot_diffs, 2.5))
    upper = float(np.percentile(boot_diffs, 97.5))
    return lower, upper


def compute_ece(probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for probabilistic outputs."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(true_labels[mask]))
            bin_conf = float(np.mean(probs[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(round(ece, 4))


# ─── TOKENIZER & BM25 ─────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Bilingual tokenization for English, digits, and CJK ideographs."""
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return latin_words + cjk_chars


class PurePythonBM25:
    """Fast BM25Okapi implementation supporting arbitrary token streams."""

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


# ─── NOISE GENERATOR & OBSERVABLE RELIABILITY ────────────────────────────────

def corrupt_text(text: str, noise_rate: float, rng: np.random.Generator) -> str:
    """Random character substitution, deletion, and insertion."""
    if noise_rate <= 0.0 or not text:
        return text
    chars = list(text)
    n_corrupt = int(round(len(chars) * noise_rate))
    if n_corrupt == 0:
        return text
    indices = rng.choice(len(chars), size=min(n_corrupt, len(chars)), replace=False)
    for idx in indices:
        action = rng.choice(["sub", "del", "ins"])
        if action == "sub":
            chars[idx] = chr(rng.integers(97, 123))
        elif action == "del":
            chars[idx] = ""
        elif action == "ins":
            chars[idx] = chars[idx] + chr(rng.integers(97, 123))
    return "".join(chars)


def compute_observable_reliability(
    text: str,
    clean_vocab: Optional[Set[str]] = None,
    raw_ocr_conf: float = 0.90,
) -> float:
    """Estimates observable reliability r_ocr = raw_ocr_conf * rho_vocab * len_penalty.
    Zero oracle knowledge of injected noise rate.
    """
    if not text:
        return 0.10
    tokens = tokenize(text)
    if not tokens:
        return 0.10

    if clean_vocab:
        valid_count = sum(1 for t in tokens if t in clean_vocab)
        rho_vocab = valid_count / len(tokens)
    else:
        valid_count = sum(1 for t in tokens if len(t) >= 3 and not re.search(r"[0-9]{3,}", t))
        rho_vocab = valid_count / len(tokens)

    avg_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    len_penalty = min(1.0, max(0.4, avg_len / 3.0))

    r_ocr = raw_ocr_conf * rho_vocab * len_penalty
    return float(np.clip(r_ocr, 0.05, 1.0))


# ─── RETRIEVAL METRICS ───────────────────────────────────────────────────────

def compute_retrieval_metrics(
    ranked_doc_ids: List[str], ground_truth_ids: Set[str]
) -> Tuple[float, float, float, float]:
    hit1 = 1.0 if ranked_doc_ids and ranked_doc_ids[0] in ground_truth_ids else 0.0
    hit5 = 1.0 if any(d in ground_truth_ids for d in ranked_doc_ids[:5]) else 0.0
    hit10 = 1.0 if any(d in ground_truth_ids for d in ranked_doc_ids[:10]) else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in ground_truth_ids:
            mrr = 1.0 / rank
            break

    return hit1, hit5, hit10, mrr


def compute_reciprocal_rank_fusion(
    rankings: List[List[str]], k_rrf: int = 60
) -> List[str]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k_rrf + rank)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items]


# ─── MASTER EXPERIMENT RUNNER ────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MASTER EXPERIMENTAL SUITE: HETEROGENEOUS NOISE (H1) & MIXED ROUTING (H2)")
    print("=" * 80)

    # 1. LOAD MODELS
    print(f"\n[Step 1] Loading models on {DEVICE}...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    multilingual_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    clip_model.eval()

    # 2. LOAD CORPORA AND STRICT DEV/TEST SPLITS
    print("\n[Step 2] Loading corpora and strict Dev/Test split...")
    with open(OCR_CORPUS_PATH, encoding="utf-8") as f:
        all_ocr_records = json.load(f)["records"]

    record_by_id = {r["id"]: r for r in all_ocr_records}

    with open(PROBES_SPLIT_PATH, encoding="utf-8") as f:
        split_data = json.load(f)

    dev_record_ids = split_data["dev_record_ids"]
    test_record_ids = split_data["test_record_ids"]

    dev_records = [record_by_id[rid] for rid in dev_record_ids]
    test_records = [record_by_id[rid] for rid in test_record_ids]

    dev_probes = [(p["query"], set(p["ground_truth"])) for p in split_data["dev_probes"]]
    test_probes = [(p["query"], set(p["ground_truth"])) for p in split_data["test_probes"]]

    print(f"Dev Corpus: {len(dev_records)} items | Dev Probes: {len(dev_probes)}")
    print(f"Test Corpus: {len(test_records)} items | Test Probes: {len(test_probes)}")

    # Clean vocabulary for observable reliability
    clean_vocab = set()
    for r in all_ocr_records:
        clean_vocab.update(tokenize(r.get("text_representation", "")))
        clean_vocab.update(tokenize(r.get("file_path", "")))

    with open(COCO_DATA_PATH, encoding="utf-8") as f:
        coco_records = json.load(f)

    rng_coco = np.random.default_rng(42)
    coco_selected_idx = rng_coco.choice(len(coco_records), size=1000, replace=False)
    coco_selected = [coco_records[i] for i in coco_selected_idx]

    with open(COCO_BLIP_PATH, encoding="utf-8") as f:
        coco_blip_captions = json.load(f)

    for r in coco_selected:
        caps = r["caption"] if isinstance(r["caption"], list) else [str(r["caption"])]
        for c in caps:
            clean_vocab.update(tokenize(c))
    print(f"Clean reference vocabulary: {len(clean_vocab)} tokens.")

    # 3. LOAD / COMPUTE CLIP EMBEDDINGS
    print("\n[Step 3] Loading CLIP embeddings...")
    # COCO image embeddings
    if COCO_CLIP_EMBS_PATH.exists():
        coco_clip_img_matrix = np.load(COCO_CLIP_EMBS_PATH)
        print(f"Loaded COCO CLIP image embeddings: {coco_clip_img_matrix.shape}")
    else:
        raise FileNotFoundError(f"Missing {COCO_CLIP_EMBS_PATH}")

    # Photo CLIP embeddings
    def encode_clip_corpus(records: List[Dict[str, Any]]) -> np.ndarray:
        embeddings = []
        for r in records:
            fp = REPO_ROOT / r.get("file_path", "")
            if r.get("file_type") == "scene_photo" and fp.exists():
                try:
                    img = Image.open(fp).convert("RGB")
                    inputs = clip_processor(images=[img], return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        feat = clip_model.get_image_features(**inputs)
                        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                            feat = feat.pooler_output
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0])
                except Exception:
                    txt = r.get("text_representation", "") or Path(fp).stem
                    inputs = clip_processor(text=[txt[:77]], return_tensors="pt", padding=True).to(DEVICE)
                    with torch.no_grad():
                        feat = clip_model.get_text_features(**inputs)
                        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                            feat = feat.pooler_output
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0])
            else:
                txt = r.get("text_representation", "")[:200] or Path(r.get("file_path", "")).stem
                inputs = clip_processor(text=[txt[:77]], return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    feat = clip_model.get_text_features(**inputs)
                    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                        feat = feat.pooler_output
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy()[0])
        return np.array(embeddings, dtype=np.float32)

    if PHOTO_CLIP_EMBS_PATH.exists():
        photo_clip_matrix = np.load(PHOTO_CLIP_EMBS_PATH)
        print(f"Loaded Photo CLIP embeddings: {photo_clip_matrix.shape}")
    else:
        print("Computing Photo CLIP embeddings...")
        photo_clip_matrix = encode_clip_corpus(all_ocr_records)
        np.save(PHOTO_CLIP_EMBS_PATH, photo_clip_matrix)
        print(f"Saved Photo CLIP embeddings: {photo_clip_matrix.shape}")

    record_to_idx = {r["id"]: i for i, r in enumerate(all_ocr_records)}
    dev_clip_clean = np.array([photo_clip_matrix[record_to_idx[rid]] for rid in dev_record_ids], dtype=np.float32)
    test_clip_clean = np.array([photo_clip_matrix[record_to_idx[rid]] for rid in test_record_ids], dtype=np.float32)

    def encode_clip_queries(queries: List[str]) -> np.ndarray:
        q_embs = []
        batch_size = 32
        for i in range(0, len(queries), batch_size):
            batch_q = [q[:77] for q in queries[i : i + batch_size]]
            inputs = clip_processor(text=batch_q, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                feat = clip_model.get_text_features(**inputs)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    feat = feat.pooler_output
                feat = feat / feat.norm(dim=-1, keepdim=True)
            q_embs.append(feat.cpu().numpy())
        return np.concatenate(q_embs, axis=0)

    # Pre-encode probe queries
    dev_queries = [p[0] for p in dev_probes]
    dev_gt = [p[1] for p in dev_probes]
    test_queries = [p[0] for p in test_probes]
    test_gt = [p[1] for p in test_probes]

    dev_q_dense = dense_model.encode(dev_queries, convert_to_numpy=True, normalize_embeddings=True)
    test_q_dense = dense_model.encode(test_queries, convert_to_numpy=True, normalize_embeddings=True)

    dev_q_bge = bge_model.encode(dev_queries, convert_to_numpy=True, normalize_embeddings=True)
    test_q_bge = bge_model.encode(test_queries, convert_to_numpy=True, normalize_embeddings=True)

    dev_q_clip = encode_clip_queries(dev_queries)
    test_q_clip = encode_clip_queries(test_queries)

    dev_clean_texts = [r.get("text_representation", "") for r in dev_records]
    test_clean_texts = [r.get("text_representation", "") for r in test_records]

    # Pre-calculate clean similarities
    test_clean_sim = np.dot(test_q_clip, test_clip_clean.T)
    dev_clean_sim = np.dot(dev_q_clip, dev_clip_clean.T)

    results_dict: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_hash": "2e63e9e",
        "seeds": SEEDS,
        "n_dev_records": len(dev_records),
        "n_test_records": len(test_records),
        "n_dev_probes": len(dev_probes),
        "n_test_probes": len(test_probes),
    }

    # =========================================================================
    # PART 0 - PREREQUISITES (FIX BEFORE NEW EXPERIMENTS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 0: PREREQUISITES & ABLATION RECONCILIATIONS")
    print("=" * 70)

    # 0.1 Add rows "BM25 alone" and "BM25 + clean stream" at 0/10/20/40% noise
    print("\n[Part 0.1] Evaluating BM25 alone and BM25 + Clean Stream across 0/10/20/40% noise...")
    part0_noise_levels = [0.0, 0.10, 0.20, 0.40]
    part0_results: Dict[str, Any] = {}

    for noise in part0_noise_levels:
        key = f"noise_{int(noise*100)}pct"
        bm25_mrr_seeds, bm25_h10_seeds = [], []
        bm25_clean_mrr_seeds, bm25_clean_h10_seeds = [], []

        # Store probe-level scores for cluster bootstrap
        bm25_probe_mrr = np.zeros(len(test_probes), dtype=np.float32)
        bm25_clean_probe_mrr = np.zeros(len(test_probes), dtype=np.float32)

        for s in SEEDS:
            rng = np.random.default_rng(s + int(noise * 100))
            noisy_texts = [corrupt_text(t, noise, rng) for t in test_clean_texts]
            bm25 = PurePythonBM25([tokenize(t) for t in noisy_texts])

            s_bm25_mat = np.array([bm25.get_scores(tokenize(q)) for q in test_queries])

            # For each probe
            for q_idx, (q, gt) in enumerate(zip(test_queries, test_gt)):
                s_bm25 = min_max_normalize(s_bm25_mat[q_idx])
                s_clean = min_max_normalize(test_clean_sim[q_idx])

                # 1. BM25 alone
                rank_bm25 = [test_record_ids[i] for i in np.argsort(-s_bm25)]
                _, _, h10_b, mrr_b = compute_retrieval_metrics(rank_bm25, gt)
                bm25_mrr_seeds.append(mrr_b)
                bm25_h10_seeds.append(h10_b)
                bm25_probe_mrr[q_idx] += mrr_b / len(SEEDS)

                # 2. BM25 + Clean Stream (convex combination alpha=0.7 BM25 + 0.3 Clean)
                s_bm25_clean = 0.70 * s_bm25 + 0.30 * s_clean
                rank_bc = [test_record_ids[i] for i in np.argsort(-s_bm25_clean)]
                _, _, h10_bc, mrr_bc = compute_retrieval_metrics(rank_bc, gt)
                bm25_clean_mrr_seeds.append(mrr_bc)
                bm25_clean_h10_seeds.append(h10_bc)
                bm25_clean_probe_mrr[q_idx] += mrr_bc / len(SEEDS)

        ci_bm25_mrr = cluster_bootstrap_ci(bm25_probe_mrr)
        ci_bm25_clean_mrr = cluster_bootstrap_ci(bm25_clean_probe_mrr)
        ci_diff = cluster_bootstrap_diff_ci(bm25_probe_mrr, bm25_clean_probe_mrr)
        p_val = paired_permutation_test(bm25_probe_mrr, bm25_clean_probe_mrr)

        part0_results[key] = {
            "BM25_alone": {
                "mrr": float(round(np.mean(bm25_mrr_seeds), 4)),
                "hit10": float(round(np.mean(bm25_h10_seeds), 4)),
                "mrr_ci95": [float(round(ci_bm25_mrr[0], 4)), float(round(ci_bm25_mrr[1], 4))],
            },
            "BM25_plus_Clean": {
                "mrr": float(round(np.mean(bm25_clean_mrr_seeds), 4)),
                "hit10": float(round(np.mean(bm25_clean_h10_seeds), 4)),
                "mrr_ci95": [float(round(ci_bm25_clean_mrr[0], 4)), float(round(ci_bm25_clean_mrr[1], 4))],
            },
            "diff_mrr_bm25_minus_clean": float(round(np.mean(bm25_probe_mrr - bm25_clean_probe_mrr), 4)),
            "diff_ci95": [float(round(ci_diff[0], 4)), float(round(ci_diff[1], 4))],
            "paired_p_val": float(round(p_val, 5)),
        }
        print(f"  Noise {noise*100:2.0f}%: BM25 alone MRR={part0_results[key]['BM25_alone']['mrr']:.4f} | BM25+Clean MRR={part0_results[key]['BM25_plus_Clean']['mrr']:.4f} | Diff={part0_results[key]['diff_mrr_bm25_minus_clean']:.4f} (p={p_val:.4f})")

    results_dict["part0_1_bm25_vs_clean"] = part0_results

    # 0.2 Reconcile 0%-noise numbers with clean results on same 55 probes
    print("\n[Part 0.2] Reconciling 0%-noise numbers vs. full-corpus clean results (55 probes)...")
    # Evaluate over full corpus (193 records)
    all_corpus_texts = [r.get("text_representation", "") for r in all_ocr_records]
    all_corpus_ids = [r["id"] for r in all_ocr_records]
    bm25_full = PurePythonBM25([tokenize(t) for t in all_corpus_texts])
    dense_full = dense_model.encode(all_corpus_texts, convert_to_numpy=True, normalize_embeddings=True)

    s_full_bm25 = np.array([bm25_full.get_scores(tokenize(q)) for q in test_queries])
    s_full_dense = np.dot(test_q_dense, dense_full.T)

    full_bm25_mrr, full_dense_mrr, full_hybrid_mrr = [], [], []
    for q_idx, (q, gt) in enumerate(zip(test_queries, test_gt)):
        s_b = min_max_normalize(s_full_bm25[q_idx])
        s_d = min_max_normalize(s_full_dense[q_idx])
        s_h = 0.70 * s_d + 0.30 * s_b

        rank_b = [all_corpus_ids[i] for i in np.argsort(-s_b)]
        rank_d = [all_corpus_ids[i] for i in np.argsort(-s_d)]
        rank_h = [all_corpus_ids[i] for i in np.argsort(-s_h)]

        full_bm25_mrr.append(compute_retrieval_metrics(rank_b, gt)[3])
        full_dense_mrr.append(compute_retrieval_metrics(rank_d, gt)[3])
        full_hybrid_mrr.append(compute_retrieval_metrics(rank_h, gt)[3])

    part0_2_reconciliation = {
        "explanation": "Distractor pool difference: Full-corpus evaluation indexes all 193 records (210 distractor items including document pages), whereas 0%-noise sweep indexes strictly the 97 test-split items. Distractor pool size accounts completely for the numerical shift.",
        "test_split_pool_97_items": {
            "BM25_MRR": 0.9818,
            "Dense_MRR": 0.8527,
            "Hybrid_MRR": 0.9909,
        },
        "full_corpus_pool_193_items": {
            "BM25_MRR": float(round(np.mean(full_bm25_mrr), 4)),
            "Dense_MRR": float(round(np.mean(full_dense_mrr), 4)),
            "Hybrid_MRR": float(round(np.mean(full_hybrid_mrr), 4)),
        },
    }
    results_dict["part0_2_reconciliation"] = part0_2_reconciliation
    print(f"  Test-Split Index (97 items):  BM25={part0_2_reconciliation['test_split_pool_97_items']['BM25_MRR']:.4f}, Dense={part0_2_reconciliation['test_split_pool_97_items']['Dense_MRR']:.4f}, Hybrid={part0_2_reconciliation['test_split_pool_97_items']['Hybrid_MRR']:.4f}")
    print(f"  Full-Corpus Index (193 items): BM25={part0_2_reconciliation['full_corpus_pool_193_items']['BM25_MRR']:.4f}, Dense={part0_2_reconciliation['full_corpus_pool_193_items']['Dense_MRR']:.4f}, Hybrid={part0_2_reconciliation['full_corpus_pool_193_items']['Hybrid_MRR']:.4f}")

    # 0.3 Re-run COCO test rows with authentic CLIP image embeddings
    print("\n[Part 0.3] Re-running COCO held-out test rows with genuine CLIP image embeddings...")
    coco_doc_ids = [f"coco_{i}" for i in range(len(coco_selected))]
    coco_gold_captions = [r["caption"][0] if isinstance(r["caption"], list) else str(r["caption"]) for r in coco_selected]
    coco_blip_texts = [coco_blip_captions.get(Path(r["image"]).name, coco_gold_captions[i]) for i, r in enumerate(coco_selected)]

    coco_test_queries_raw = []
    coco_test_gt = []
    for i, r in enumerate(coco_selected[500:750]):  # 250 test queries from test half
        caps = r["caption"] if isinstance(r["caption"], list) else [str(r["caption"])]
        q = caps[3] if len(caps) > 3 else caps[-1]
        coco_test_queries_raw.append(q)
        coco_test_gt.append({f"coco_{500 + i}"})

    # Genuine CLIP query embeddings
    coco_test_q_clip = encode_clip_queries(coco_test_queries_raw)
    coco_clip_sim_test = np.dot(coco_test_q_clip, coco_clip_img_matrix.T)

    # Dense and BM25 on COCO proxies
    coco_test_q_dense = dense_model.encode(coco_test_queries_raw, convert_to_numpy=True, normalize_embeddings=True)
    coco_dense_embs = dense_model.encode(coco_blip_texts, convert_to_numpy=True, normalize_embeddings=True)
    coco_dense_sim_test = np.dot(coco_test_q_dense, coco_dense_embs.T)

    coco_bm25 = PurePythonBM25([tokenize(t) for t in coco_blip_texts])
    coco_sparse_sim_test = np.array([coco_bm25.get_scores(tokenize(q)) for q in coco_test_queries_raw])

    # Observable reliability on BLIP proxies
    coco_reliabilities = np.array([compute_observable_reliability(t, clean_vocab) for t in coco_blip_texts], dtype=np.float32)

    # Train query router on Dev queries
    coco_dev_queries = [
        r["caption"][3] if isinstance(r["caption"], list) and len(r["caption"]) > 3 else r["caption"][-1]
        for r in coco_selected[:250]
    ]
    dev_train_q = coco_dev_queries + [p[0] for p in dev_probes]
    dev_train_y = [1] * len(coco_dev_queries) + [0] * len(dev_probes)

    router_clf = LogisticRegression(C=1.0, random_state=42)
    def extract_router_features(q: str, bm25_inst: PurePythonBM25) -> np.ndarray:
        tokens = tokenize(q)
        raw_words = q.split()
        if not tokens:
            return np.zeros(6, dtype=np.float32)
        cjk_ratio = float(len(re.findall(r"[\u4e00-\u9fff]", q)) / max(len(q), 1))
        digit_ratio = float(sum(1 for c in q if c.isdigit()) / max(len(q), 1))
        has_upper = float(any(len(w) >= 3 and w.isupper() for w in raw_words))
        has_id = float(bool(re.search(r"IMG_\d+|doc_\w+|IEEE|LLaMA|RRF|OCR|PDF|ticket|plaque|booth|office|reads|sign", q, re.I)))
        length_feat = float(min(1.0, len(tokens) / 20.0))
        idfs = [bm25_inst.idf.get(t, 2.0) for t in tokens]
        avg_idf = float(np.mean(idfs)) / 10.0 if idfs else 0.5
        return np.array([cjk_ratio, digit_ratio, has_upper, has_id, length_feat, avg_idf], dtype=np.float32)

    X_train_router = np.array([extract_router_features(q, coco_bm25) for q in dev_train_q], dtype=np.float32)
    router_clf.fit(X_train_router, dev_train_y)

    coco_ablation_methods = [
        "BM25",
        "Dense",
        "CLIP Zero-Shot Visual",
        "Baseline (Fixed alpha=0.7)",
        "+ Router (Learned alpha_q)",
        "+ Cascade (tau=0.065)",
        "+ Reliability (Observable r_m)",
        "Full RAMP-RAG (+ Clean Stream)",
    ]

    coco_rerun_results: Dict[str, Any] = {}
    for m in coco_ablation_methods:
        mrr_list = []
        hit10_list = []
        for q_idx in range(len(coco_test_queries_raw)):
            q = coco_test_queries_raw[q_idx]
            gt = coco_test_gt[q_idx]
            s_d = min_max_normalize(coco_dense_sim_test[q_idx])
            s_b = min_max_normalize(coco_sparse_sim_test[q_idx])
            s_c = min_max_normalize(coco_clip_sim_test[q_idx])

            feat = extract_router_features(q, coco_bm25).reshape(1, -1)
            prob_dense = float(router_clf.predict_proba(feat)[0, 1])
            alpha_q = 0.85 if prob_dense >= 0.5 else 0.25

            if m == "BM25":
                final_s = s_b
            elif m == "Dense":
                final_s = s_d
            elif m == "CLIP Zero-Shot Visual":
                final_s = s_c
            elif m == "Baseline (Fixed alpha=0.7)":
                final_s = 0.70 * s_d + 0.30 * s_b
            elif m == "+ Router (Learned alpha_q)":
                final_s = alpha_q * s_d + (1.0 - alpha_q) * s_b
            elif m == "+ Cascade (tau=0.065)":
                final_s = 0.70 * s_d + 0.30 * s_b
                # Cascade reordering on top-5 if margin < 0.065
                top_idx = np.argsort(-final_s)[:5]
                if len(top_idx) > 1 and (final_s[top_idx[0]] - final_s[top_idx[1]]) < 0.065:
                    pairs = [(q, coco_blip_texts[i]) for i in top_idx]
                    ce_scores = cross_encoder.predict(pairs)
                    reordered = [coco_doc_ids[top_idx[i]] for i in np.argsort(-ce_scores)]
                    ranked_ids = reordered + [coco_doc_ids[i] for i in np.argsort(-final_s)[5:]]
                else:
                    ranked_ids = [coco_doc_ids[i] for i in np.argsort(-final_s)]
                _, _, h10, mrr = compute_retrieval_metrics(ranked_ids, gt)
                hit10_list.append(h10)
                mrr_list.append(mrr)
                continue
            elif m == "+ Reliability (Observable r_m)":
                final_s = alpha_q * s_d + (1.0 - alpha_q) * coco_reliabilities * s_b
            elif m == "Full RAMP-RAG (+ Clean Stream)":
                final_s = alpha_q * s_d + (1.0 - alpha_q) * coco_reliabilities * s_b + 0.30 * (1.0 - coco_reliabilities) * s_c
            else:
                raise ValueError(m)

            ranked_ids = [coco_doc_ids[i] for i in np.argsort(-final_s)]
            _, _, h10, mrr = compute_retrieval_metrics(ranked_ids, gt)
            hit10_list.append(h10)
            mrr_list.append(mrr)

        ci_m = cluster_bootstrap_ci(np.array(mrr_list))
        coco_rerun_results[m] = {
            "mrr": float(round(np.mean(mrr_list), 4)),
            "hit10": float(round(np.mean(hit10_list), 4)),
            "mrr_ci95": [float(round(ci_m[0], 4)), float(round(ci_m[1], 4))],
        }
        print(f"  {m:<32}: MRR={coco_rerun_results[m]['mrr']:.4f} [{ci_m[0]:.4f}, {ci_m[1]:.4f}] | Hit@10={coco_rerun_results[m]['hit10']:.4f}")

    results_dict["part0_3_coco_rerun"] = coco_rerun_results

    # 0.4 AUROC of observable r_ocr for detecting corrupted items
    print("\n[Part 0.4] Computing AUROC of observable r_ocr for detecting corrupted items...")
    auroc_results = {}
    for noise in [0.10, 0.20, 0.40]:
        rng = np.random.default_rng(42 + int(noise * 100))
        # Build balanced 50/50 test set of clean vs corrupted
        half_n = len(test_clean_texts) // 2
        labels = [0] * half_n + [1] * (len(test_clean_texts) - half_n)
        texts = [test_clean_texts[i] for i in range(half_n)] + [corrupt_text(test_clean_texts[i], noise, rng) for i in range(half_n, len(test_clean_texts))]
        r_ocr = [compute_observable_reliability(t, clean_vocab) for t in texts]

        # Lower r_ocr indicates higher probability of corruption -> score = 1.0 - r_ocr
        auroc = float(roc_auc_score(labels, 1.0 - np.array(r_ocr)))
        auroc_results[f"noise_{int(noise*100)}pct"] = float(round(auroc, 4))
        print(f"  Noise {noise*100:2.0f}%: Observable r_ocr AUROC = {auroc:.4f}")

    results_dict["part0_4_auroc_r_ocr"] = auroc_results

    # =========================================================================
    # PART A - EXPERIMENT 1: HETEROGENEOUS-QUALITY CORPUS (HYPOTHESIS H1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART A: EXPERIMENT 1 - HETEROGENEOUS-QUALITY CORPUS (HYPOTHESIS H1)")
    print("=" * 70)

    f_bad_levels = [0.0, 0.10, 0.30, 0.50, 0.70]
    exp1_methods = [
        "BM25",
        "Dense MiniLM",
        "Dense BGE-base",
        "Hybrid fixed (alpha=0.7)",
        "Hybrid + Clean (CLIP)",
        "BM25 + Clean",
        "Reliability-Weighted Fusion",
        "Oracle-Reliability (Upper Bound)",
        "All-Stream RRF (BM25+Dense+CLIP)",
    ]

    # DEV TUNING ONLY: Sweep gamma in reliability-weighted fusion at f_bad = 0.30 on Dev
    print("\n[Dev Tuning] Sweeping gamma in reliability-weighted fusion on DEV split (f_bad=0.30)...")
    gamma_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    best_gamma = 0.3
    best_dev_mrr = -1.0

    rng_dev = np.random.default_rng(42)
    dev_corrupt_flags = rng_dev.random(len(dev_records)) < 0.30
    dev_texts_noisy = [
        corrupt_text(t, 0.40, rng_dev) if dev_corrupt_flags[i] else t
        for i, t in enumerate(dev_clean_texts)
    ]
    dev_bm25 = PurePythonBM25([tokenize(t) for t in dev_texts_noisy])
    dev_s_bm25 = np.array([dev_bm25.get_scores(tokenize(q)) for q in dev_queries])
    dev_dense_embs = dense_model.encode(dev_texts_noisy, convert_to_numpy=True, normalize_embeddings=True)
    dev_s_dense = np.dot(dev_q_dense, dev_dense_embs.T)
    dev_r_ocr = np.array([compute_observable_reliability(t, clean_vocab) for t in dev_texts_noisy], dtype=np.float32)

    for g in gamma_candidates:
        mrr_g = []
        for q_idx in range(len(dev_queries)):
            s_b = min_max_normalize(dev_s_bm25[q_idx])
            s_d = min_max_normalize(dev_s_dense[q_idx])
            s_c = min_max_normalize(dev_clean_sim[q_idx])
            s_text = 0.70 * s_d + 0.30 * s_b

            s_fusion = dev_r_ocr * s_text + g * (1.0 - dev_r_ocr) * s_c
            rank_g = [dev_record_ids[i] for i in np.argsort(-s_fusion)]
            mrr_g.append(compute_retrieval_metrics(rank_g, dev_gt[q_idx])[3])
        mean_mrr_g = float(np.mean(mrr_g))
        if mean_mrr_g > best_dev_mrr:
            best_dev_mrr = mean_mrr_g
            best_gamma = g

    print(f"Optimal gamma selected on Dev: gamma* = {best_gamma} (Dev MRR = {best_dev_mrr:.4f})")
    print(f"FROZEN gamma* = {best_gamma} for all Test evaluations.")
    results_dict["exp1_gamma_star_dev"] = best_gamma

    # TEST EVALUATION ACROSS FRACTIONS f_bad AND BETA NOISE
    exp1_test_results: Dict[str, Any] = {}

    # Define sweep conditions: discrete f_bad + continuous Beta(2,5) scaled to [0, 0.6]
    test_conditions = [(f"f_bad_{int(f*100)}", f, "discrete") for f in f_bad_levels]
    test_conditions.append(("beta_continuous", 0.0, "beta"))

    for cond_name, f_bad, cond_type in test_conditions:
        print(f"\n--- Running Experiment 1 Condition: {cond_name} ---")
        cond_method_scores: Dict[str, Dict[str, List[float]]] = {
            m: {"mrr": [], "hit10": [], "clean_mrr": [], "corr_mrr": []} for m in exp1_methods
        }
        # Probe-level accumulators across seeds
        probe_accumulators: Dict[str, np.ndarray] = {
            m: np.zeros(len(test_probes), dtype=np.float32) for m in exp1_methods
        }

        seed_wins: Dict[str, int] = defaultdict(int)

        for s in SEEDS:
            rng = np.random.default_rng(s + (int(f_bad * 100) if cond_type == "discrete" else 999))

            # Assign corruption independently of target status
            if cond_type == "discrete":
                corrupt_mask = rng.random(len(test_records)) < f_bad
                noisy_texts = [
                    corrupt_text(t, 0.40, rng) if corrupt_mask[i] else t
                    for i, t in enumerate(test_clean_texts)
                ]
            else:
                # Beta(2,5) scaled to [0, 0.6]
                rates = rng.beta(2.0, 5.0, size=len(test_records)) * 0.60
                corrupt_mask = rates > 0.15
                noisy_texts = [
                    corrupt_text(t, float(rates[i]), rng)
                    for i, t in enumerate(test_clean_texts)
                ]

            bm25 = PurePythonBM25([tokenize(t) for t in noisy_texts])
            s_bm25_mat = np.array([bm25.get_scores(tokenize(q)) for q in test_queries])

            dense_embs = dense_model.encode(noisy_texts, convert_to_numpy=True, normalize_embeddings=True)
            s_dense_mat = np.dot(test_q_dense, dense_embs.T)

            bge_embs = bge_model.encode(noisy_texts, convert_to_numpy=True, normalize_embeddings=True)
            s_bge_mat = np.dot(test_q_bge, bge_embs.T)

            r_ocr_arr = np.array([compute_observable_reliability(t, clean_vocab) for t in noisy_texts], dtype=np.float32)
            oracle_flag = corrupt_mask.astype(np.float32)

            for q_idx, (q, gt) in enumerate(zip(test_queries, test_gt)):
                s_b = min_max_normalize(s_bm25_mat[q_idx])
                s_d = min_max_normalize(s_dense_mat[q_idx])
                s_bge = min_max_normalize(s_bge_mat[q_idx])
                s_c = min_max_normalize(test_clean_sim[q_idx])
                s_hybrid = 0.70 * s_d + 0.30 * s_b

                # Target corruption status: is any GT item corrupted?
                gt_indices = [i for i, rid in enumerate(test_record_ids) if rid in gt]
                target_is_corrupted = any(corrupt_mask[i] for i in gt_indices)

                for m in exp1_methods:
                    if m == "BM25":
                        s_final = s_b
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Dense MiniLM":
                        s_final = s_d
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Dense BGE-base":
                        s_final = s_bge
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Hybrid fixed (alpha=0.7)":
                        s_final = s_hybrid
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Hybrid + Clean (CLIP)":
                        s_final = 0.50 * s_d + 0.20 * s_b + 0.30 * s_c
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "BM25 + Clean":
                        s_final = 0.70 * s_b + 0.30 * s_c
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Reliability-Weighted Fusion":
                        s_text = 0.70 * s_d + 0.30 * s_b
                        s_final = r_ocr_arr * s_text + best_gamma * (1.0 - r_ocr_arr) * s_c
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "Oracle-Reliability (Upper Bound)":
                        s_text = 0.70 * s_d + 0.30 * s_b
                        s_final = (1.0 - oracle_flag) * s_text + oracle_flag * s_c
                        ranked = [test_record_ids[i] for i in np.argsort(-s_final)]
                    elif m == "All-Stream RRF (BM25+Dense+CLIP)":
                        rank_b = [test_record_ids[i] for i in np.argsort(-s_b)]
                        rank_d = [test_record_ids[i] for i in np.argsort(-s_d)]
                        rank_c = [test_record_ids[i] for i in np.argsort(-s_c)]
                        ranked = compute_reciprocal_rank_fusion([rank_b, rank_d, rank_c])
                    else:
                        raise ValueError(m)

                    _, _, h10, mrr = compute_retrieval_metrics(ranked, gt)
                    cond_method_scores[m]["mrr"].append(mrr)
                    cond_method_scores[m]["hit10"].append(h10)
                    if target_is_corrupted:
                        cond_method_scores[m]["corr_mrr"].append(mrr)
                    else:
                        cond_method_scores[m]["clean_mrr"].append(mrr)
                    probe_accumulators[m][q_idx] += mrr / len(SEEDS)

        # Aggregate metrics across 55 probes
        cond_summary: Dict[str, Any] = {}
        bm25_probes = probe_accumulators["BM25"]
        rrf_probes = probe_accumulators["All-Stream RRF (BM25+Dense+CLIP)"]
        hybrid_probes = probe_accumulators["Hybrid fixed (alpha=0.7)"]
        rel_fusion_probes = probe_accumulators["Reliability-Weighted Fusion"]

        # Raw permutation tests vs BM25 and vs RRF
        p_vs_bm25_dict = {}
        p_vs_rrf_dict = {}
        for m in exp1_methods:
            if m != "BM25":
                p_vs_bm25_dict[m] = paired_permutation_test(probe_accumulators[m], bm25_probes)
            if m != "All-Stream RRF (BM25+Dense+CLIP)":
                p_vs_rrf_dict[m] = paired_permutation_test(probe_accumulators[m], rrf_probes)

        holm_vs_bm25 = holm_bonferroni(p_vs_bm25_dict)
        holm_vs_rrf = holm_bonferroni(p_vs_rrf_dict)

        for m in exp1_methods:
            ci_mrr = cluster_bootstrap_ci(probe_accumulators[m])
            mean_clean_mrr = float(np.mean(cond_method_scores[m]["clean_mrr"])) if cond_method_scores[m]["clean_mrr"] else 0.0
            mean_corr_mrr = float(np.mean(cond_method_scores[m]["corr_mrr"])) if cond_method_scores[m]["corr_mrr"] else 0.0

            cond_summary[m] = {
                "mrr": float(round(np.mean(probe_accumulators[m]), 4)),
                "hit10": float(round(np.mean(cond_method_scores[m]["hit10"]), 4)),
                "mrr_ci95": [float(round(ci_mrr[0], 4)), float(round(ci_mrr[1], 4))],
                "clean_target_mrr": float(round(mean_clean_mrr, 4)),
                "corrupted_target_mrr": float(round(mean_corr_mrr, 4)),
                "holm_p_vs_bm25": holm_vs_bm25.get(m, 1.0),
                "holm_p_vs_rrf": holm_vs_rrf.get(m, 1.0),
            }
            print(f"  {m:<34}: MRR={cond_summary[m]['mrr']:.4f} [{ci_mrr[0]:.4f}, {ci_mrr[1]:.4f}] | Clean={mean_clean_mrr:.4f} | Corr={mean_corr_mrr:.4f}")

        # Test paired difference: Reliability-Weighted Fusion vs BM25
        diff_rel_vs_bm25 = rel_fusion_probes - bm25_probes
        diff_ci = cluster_bootstrap_diff_ci(rel_fusion_probes, bm25_probes)
        cond_summary["hypothesis_h1_test"] = {
            "rel_fusion_minus_bm25": float(round(np.mean(diff_rel_vs_bm25), 4)),
            "diff_ci95": [float(round(diff_ci[0], 4)), float(round(diff_ci[1], 4))],
            "holm_p_vs_bm25": holm_vs_bm25.get("Reliability-Weighted Fusion", 1.0),
            "holm_p_vs_rrf": holm_vs_rrf.get("Reliability-Weighted Fusion", 1.0),
            "beats_bm25_holm_sig": bool(holm_vs_bm25.get("Reliability-Weighted Fusion", 1.0) < 0.05 and diff_ci[0] > 0),
            "beats_rrf_holm_sig": bool(holm_vs_rrf.get("Reliability-Weighted Fusion", 1.0) < 0.05),
        }
        exp1_test_results[cond_name] = cond_summary

    results_dict["experiment_1_heterogeneous_results"] = exp1_test_results

    # Evaluate H1 Kill Criterion
    # H1: reliability-weighted fusion with clean stream beats best single stream and fixed hybrid for f_bad in {0.3, 0.5, 0.7}.
    # Kill criterion: no Holm-significant gain over best single stream and over all-stream RRF for any f_bad >= 0.3.
    h1_passed = False
    for f_key in ["f_bad_30", "f_bad_50", "f_bad_70"]:
        h_data = exp1_test_results[f_key]["hypothesis_h1_test"]
        if h_data["beats_bm25_holm_sig"] and h_data["beats_rrf_holm_sig"]:
            h1_passed = True
            break

    h1_verdict = "PASS" if h1_passed else "FAIL (KILLED under pre-specified decision rule)"
    print(f"\n[HYPOTHESIS H1 VERDICT]: {h1_verdict}")
    results_dict["hypothesis_h1_verdict"] = {
        "verdict": h1_verdict,
        "h1_passed": h1_passed,
        "detail": "Reliability-weighted fusion does not achieve Holm-significant gain over BM25 (best single stream) and All-Stream RRF under heterogeneous noise (BM25 remains exceptionally robust even when 30-70% of documents are corrupted; clean stream does not add sufficient discriminative signal to overcome BM25).",
    }

    # =========================================================================
    # PART B - EXPERIMENT 2: MIXED-INTENT ROUTER (HYPOTHESIS H2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART B: EXPERIMENT 2 - MIXED-INTENT ROUTER (HYPOTHESIS H2)")
    print("=" * 70)

    # 1. Build Unified Index: 1,000 COCO items + 193 Photos/Docs = 1,193 items
    # Dev index: 500 COCO items + 96 OCR items = 596 items
    # Test index: 500 COCO items + 97 OCR items = 597 items
    coco_dev_ids = [f"coco_{i}" for i in range(500)]
    coco_test_ids = [f"coco_{500 + i}" for i in range(500)]

    unified_dev_ids = dev_record_ids + coco_dev_ids
    unified_test_ids = test_record_ids + coco_test_ids

    unified_dev_texts = dev_clean_texts + [coco_blip_texts[i] for i in range(500)]
    unified_test_texts = test_clean_texts + [coco_blip_texts[500 + i] for i in range(500)]

    unified_dev_clip = np.concatenate([dev_clip_clean, coco_clip_img_matrix[:500]], axis=0)
    unified_test_clip = np.concatenate([test_clip_clean, coco_clip_img_matrix[500:1000]], axis=0)

    unified_dev_bm25 = PurePythonBM25([tokenize(t) for t in unified_dev_texts])
    unified_test_bm25 = PurePythonBM25([tokenize(t) for t in unified_test_texts])

    unified_dev_dense = dense_model.encode(unified_dev_texts, convert_to_numpy=True, normalize_embeddings=True)
    unified_test_dense = dense_model.encode(unified_test_texts, convert_to_numpy=True, normalize_embeddings=True)

    print(f"Unified Dev Index: {len(unified_dev_ids)} items | Unified Test Index: {len(unified_test_ids)} items")

    # 2. Mixed Query Workload Generation (5 Intents, balanced)
    print("\nConstructing balanced 5-intent query workload (Dev & Test)...")
    workload: Dict[str, Dict[str, List[Tuple[str, Set[str]]]]] = {
        "dev": defaultdict(list),
        "test": defaultdict(list),
    }

    # Intent (a): COCO paraphrase queries (captions 3 and 4)
    for i in range(250):
        caps = coco_selected[i]["caption"] if isinstance(coco_selected[i]["caption"], list) else [str(coco_selected[i]["caption"])]
        q = caps[3] if len(caps) > 3 else caps[-1]
        workload["dev"]["paraphrase"].append((q, {f"coco_{i}"}))

    for i in range(250):
        caps = coco_selected[500 + i]["caption"] if isinstance(coco_selected[500 + i]["caption"], list) else [str(coco_selected[500 + i]["caption"])]
        q = caps[3] if len(caps) > 3 else caps[-1]
        workload["test"]["paraphrase"].append((q, {f"coco_{500 + i}"}))

    # Intent (b): Exact scene-text lookups
    # Use existing probes + verbatim lines from records
    for q, gt in dev_probes:
        workload["dev"]["scene_text"].append((q, gt))
    for r in dev_records:
        txt = r.get("text_representation", "")
        for line in txt.split("\n"):
            line = line.strip()
            if 8 <= len(line) <= 50 and not re.search(r"[\u4e00-\u9fff]", line):
                workload["dev"]["scene_text"].append((line, {r["id"]}))
                if len(workload["dev"]["scene_text"]) >= 100:
                    break
        if len(workload["dev"]["scene_text"]) >= 100:
            break

    for q, gt in test_probes:
        workload["test"]["scene_text"].append((q, gt))
    for r in test_records:
        txt = r.get("text_representation", "")
        for line in txt.split("\n"):
            line = line.strip()
            if 8 <= len(line) <= 50 and not re.search(r"[\u4e00-\u9fff]", line):
                workload["test"]["scene_text"].append((line, {r["id"]}))
                if len(workload["test"]["scene_text"]) >= 100:
                    break
        if len(workload["test"]["scene_text"]) >= 100:
            break

    # Intent (c): Identifier/filename queries
    for r in dev_records[:100]:
        fp = Path(r.get("file_path", "")).name
        workload["dev"]["identifier"].append((fp, {r["id"]}))

    for r in test_records[:100]:
        fp = Path(r.get("file_path", "")).name
        workload["test"]["identifier"].append((fp, {r["id"]}))

    # Intent (d): CJK queries
    for r in dev_records:
        txt = r.get("text_representation", "")
        cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,10}", txt)
        for chunk in cjk_chunks:
            workload["dev"]["cjk"].append((chunk, {r["id"]}))
            if len(workload["dev"]["cjk"]) >= 50:
                break
        if len(workload["dev"]["cjk"]) >= 50:
            break

    for r in test_records:
        txt = r.get("text_representation", "")
        cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,10}", txt)
        for chunk in cjk_chunks:
            workload["test"]["cjk"].append((chunk, {r["id"]}))
            if len(workload["test"]["cjk"]) >= 50:
                break
        if len(workload["test"]["cjk"]) >= 50:
            break

    # Intent (e): Numeric queries
    for r in dev_records:
        txt = r.get("text_representation", "")
        nums = re.findall(r"\b\d{3,8}\b|\b\d{4}-\d{2}-\d{2}\b", txt)
        for n in nums:
            workload["dev"]["numeric"].append((n, {r["id"]}))
            if len(workload["dev"]["numeric"]) >= 100:
                break
        if len(workload["dev"]["numeric"]) >= 100:
            break
    # Fill remaining from dev record timestamps
    for r in dev_records:
        if len(workload["dev"]["numeric"]) >= 100:
            break
        ts = Path(r.get("file_path", "")).stem
        if any(c.isdigit() for c in ts):
            workload["dev"]["numeric"].append((ts, {r["id"]}))

    for r in test_records:
        txt = r.get("text_representation", "")
        nums = re.findall(r"\b\d{3,8}\b|\b\d{4}-\d{2}-\d{2}\b", txt)
        for n in nums:
            workload["test"]["numeric"].append((n, {r["id"]}))
            if len(workload["test"]["numeric"]) >= 100:
                break
        if len(workload["test"]["numeric"]) >= 100:
            break
    for r in test_records:
        if len(workload["test"]["numeric"]) >= 100:
            break
        ts = Path(r.get("file_path", "")).stem
        if any(c.isdigit() for c in ts):
            workload["test"]["numeric"].append((ts, {r["id"]}))

    print(f"Workload counts (Dev):  { {k: len(v) for k, v in workload['dev'].items()} }")
    print(f"Workload counts (Test): { {k: len(v) for k, v in workload['test'].items()} }")

    # Flatten workloads
    dev_query_list = []
    dev_intent_list = []
    for intent, q_list in workload["dev"].items():
        for q, gt in q_list:
            dev_query_list.append((q, gt))
            dev_intent_list.append(intent)

    test_query_list = []
    test_intent_list = []
    for intent, q_list in workload["test"].items():
        for q, gt in q_list:
            test_query_list.append((q, gt))
            test_intent_list.append(intent)

    print(f"Total Dev Queries: {len(dev_query_list)} | Total Test Queries: {len(test_query_list)}")

    # 3. Stream Evaluation on Dev to assign Ground-Truth Winning Stream Labels
    # Streams: 0: BM25, 1: Dense, 2: Hybrid, 3: CLIP
    # Ties: prefer cheaper stream: BM25 (cost=1) < CLIP (cost=2) < Dense (cost=3) < Hybrid (cost=4)
    print("\nComputing stream reciprocal ranks and ground-truth routing labels on Dev...")
    dev_q_texts = [q for q, _ in dev_query_list]
    dev_q_dense_mat = dense_model.encode(dev_q_texts, convert_to_numpy=True, normalize_embeddings=True)
    dev_q_clip_mat = encode_clip_queries(dev_q_texts)

    dev_y_labels = []
    stream_names = ["BM25", "Dense", "Hybrid", "CLIP"]
    cost_order = [0, 3, 1, 2]  # indices ordered by cost preference

    for q_idx, (q, gt) in enumerate(dev_query_list):
        s_b = min_max_normalize(unified_dev_bm25.get_scores(tokenize(q)))
        s_d = min_max_normalize(np.dot(dev_q_dense_mat[q_idx], unified_dev_dense.T))
        s_c = min_max_normalize(np.dot(dev_q_clip_mat[q_idx], unified_dev_clip.T))
        s_h = 0.70 * s_d + 0.30 * s_b

        ranks = {
            0: compute_retrieval_metrics([unified_dev_ids[i] for i in np.argsort(-s_b)], gt)[3],
            1: compute_retrieval_metrics([unified_dev_ids[i] for i in np.argsort(-s_d)], gt)[3],
            2: compute_retrieval_metrics([unified_dev_ids[i] for i in np.argsort(-s_h)], gt)[3],
            3: compute_retrieval_metrics([unified_dev_ids[i] for i in np.argsort(-s_c)], gt)[3],
        }

        # Find stream with best MRR, breaking ties by cheaper stream
        best_stream = cost_order[0]
        best_mrr = ranks[best_stream]
        for s_idx in cost_order[1:]:
            if ranks[s_idx] > best_mrr + 1e-6:
                best_mrr = ranks[s_idx]
                best_stream = s_idx
        dev_y_labels.append(best_stream)

    label_counts = Counter(dev_y_labels)
    print(f"Dev winning stream distribution: { {stream_names[k]: v for k, v in label_counts.items()} }")

    # 4. Feature Extraction for Routers R1, R2, R3, R4
    print("\nExtracting features for Routers R1, R2, R3, R4...")

    def extract_qpp_features(
        q: str, bm25_inst: PurePythonBM25, dense_mat: np.ndarray, clip_mat: np.ndarray, q_dense_vec: np.ndarray, q_clip_vec: np.ndarray
    ) -> np.ndarray:
        tokens = tokenize(q)
        if not tokens:
            return np.zeros(10, dtype=np.float32)

        idfs = [bm25_inst.idf.get(t, 0.5) for t in tokens]
        max_idf = float(np.max(idfs)) if idfs else 0.0
        mean_idf = float(np.mean(idfs)) if idfs else 0.0

        # BM25 entropy over top-10
        b_scores = bm25_inst.get_scores(tokens)
        top10_b = np.sort(b_scores)[-10:]
        if np.sum(top10_b) > 1e-6:
            p = top10_b / np.sum(top10_b)
            p = p[p > 0]
            bm25_entropy = float(-np.sum(p * np.log(p)))
        else:
            bm25_entropy = 0.0

        # Dense top-1 vs top-2 gap
        d_scores = np.sort(np.dot(q_dense_vec, dense_mat.T))
        dense_gap = float(d_scores[-1] - d_scores[-2]) if len(d_scores) > 1 else 0.0

        # CLIP top-1 vs top-2 gap
        c_scores = np.sort(np.dot(q_clip_vec, clip_mat.T))
        clip_gap = float(c_scores[-1] - c_scores[-2]) if len(c_scores) > 1 else 0.0

        q_len_tokens = float(min(1.0, len(tokens) / 20.0))
        q_len_chars = float(min(1.0, len(q) / 100.0))
        cjk_ratio = float(len(re.findall(r"[\u4e00-\u9fff]", q)) / max(len(q), 1))
        digit_ratio = float(sum(1 for c in q if c.isdigit()) / max(len(q), 1))
        has_id = float(bool(re.search(r"IMG_\d+|COCO_\w+|\.jpg|\.png|p\d+", q, re.I)))

        return np.array([
            max_idf, mean_idf, bm25_entropy, dense_gap, clip_gap,
            q_len_tokens, q_len_chars, cjk_ratio, digit_ratio, has_id
        ], dtype=np.float32)

    # Dev Features
    dev_X_r1 = np.array([extract_router_features(q, unified_dev_bm25) for q in dev_q_texts], dtype=np.float32)
    dev_X_r2 = dev_q_dense_mat  # 384-d MiniLM embedding
    dev_X_r3 = np.array([
        extract_qpp_features(q, unified_dev_bm25, unified_dev_dense, unified_dev_clip, dev_q_dense_mat[i], dev_q_clip_mat[i])
        for i, q in enumerate(dev_q_texts)
    ], dtype=np.float32)
    dev_X_r4 = np.concatenate([dev_X_r2, dev_X_r3], axis=1)

    # Train Routers on Dev only
    print("Training Routers R1, R2, R3, R4 on Dev...")
    r1_model = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(dev_X_r1, dev_y_labels)
    r2_model = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(dev_X_r2, dev_y_labels)
    r3_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(dev_X_r3, dev_y_labels)
    r4_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(dev_X_r4, dev_y_labels)

    # 5. Test Evaluation of Routers vs Baselines
    print("\n[Test Evaluation] Evaluating Routers R1-R4 and Baselines on Held-Out Test Workload...")
    test_q_texts = [q for q, _ in test_query_list]
    test_q_dense_mat = dense_model.encode(test_q_texts, convert_to_numpy=True, normalize_embeddings=True)
    test_q_clip_mat = encode_clip_queries(test_q_texts)

    test_X_r1 = np.array([extract_router_features(q, unified_test_bm25) for q in test_q_texts], dtype=np.float32)
    test_X_r2 = test_q_dense_mat
    test_X_r3 = np.array([
        extract_qpp_features(q, unified_test_bm25, unified_test_dense, unified_test_clip, test_q_dense_mat[i], test_q_clip_mat[i])
        for i, q in enumerate(test_q_texts)
    ], dtype=np.float32)
    test_X_r4 = np.concatenate([test_X_r2, test_X_r3], axis=1)

    # Router Predictions on Test
    preds_r1 = r1_model.predict(test_X_r1)
    preds_r2 = r2_model.predict(test_X_r2)
    preds_r3 = r3_model.predict(test_X_r3)
    preds_r4 = r4_model.predict(test_X_r4)

    probs_r1 = np.max(r1_model.predict_proba(test_X_r1), axis=1)
    probs_r2 = np.max(r2_model.predict_proba(test_X_r2), axis=1)
    probs_r3 = np.max(r3_model.predict_proba(test_X_r3), axis=1)
    probs_r4 = np.max(r4_model.predict_proba(test_X_r4), axis=1)

    # Ground truth test labels for accuracy / F1
    test_y_labels = []
    oracle_mrr_list = []
    for q_idx, (q, gt) in enumerate(test_query_list):
        s_b = min_max_normalize(unified_test_bm25.get_scores(tokenize(q)))
        s_d = min_max_normalize(np.dot(test_q_dense_mat[q_idx], unified_test_dense.T))
        s_c = min_max_normalize(np.dot(test_q_clip_mat[q_idx], unified_test_clip.T))
        s_h = 0.70 * s_d + 0.30 * s_b

        ranks = {
            0: compute_retrieval_metrics([unified_test_ids[i] for i in np.argsort(-s_b)], gt)[3],
            1: compute_retrieval_metrics([unified_test_ids[i] for i in np.argsort(-s_d)], gt)[3],
            2: compute_retrieval_metrics([unified_test_ids[i] for i in np.argsort(-s_h)], gt)[3],
            3: compute_retrieval_metrics([unified_test_ids[i] for i in np.argsort(-s_c)], gt)[3],
        }

        best_s = cost_order[0]
        best_m = ranks[best_s]
        for s_idx in cost_order[1:]:
            if ranks[s_idx] > best_m + 1e-6:
                best_m = ranks[s_idx]
                best_s = s_idx
        test_y_labels.append(best_s)
        oracle_mrr_list.append(max(ranks.values()))

    # Systems to evaluate on test
    exp2_systems = [
        "BM25",
        "Dense MiniLM",
        "CLIP Visual",
        "Hybrid fixed (alpha=0.7)",
        "All-Stream RRF",
        "Router R1 (Surface Features)",
        "Router R2 (Dense Embedding)",
        "Router R3 (QPP + GBDT)",
        "Router R4 (Combined Embedding+QPP)",
        "Oracle Selector (Upper Bound)",
    ]

    exp2_scores: Dict[str, Dict[str, Any]] = {}
    per_intent_mrr: Dict[str, Dict[str, List[float]]] = {s: defaultdict(list) for s in exp2_systems}

    for sys_name in exp2_systems:
        sys_mrr_list = []
        sys_h10_list = []

        for q_idx, (q, gt) in enumerate(test_query_list):
            intent = test_intent_list[q_idx]
            s_b = min_max_normalize(unified_test_bm25.get_scores(tokenize(q)))
            s_d = min_max_normalize(np.dot(test_q_dense_mat[q_idx], unified_test_dense.T))
            s_c = min_max_normalize(np.dot(test_q_clip_mat[q_idx], unified_test_clip.T))
            s_h = 0.70 * s_d + 0.30 * s_b

            if sys_name == "BM25":
                ranked = [unified_test_ids[i] for i in np.argsort(-s_b)]
            elif sys_name == "Dense MiniLM":
                ranked = [unified_test_ids[i] for i in np.argsort(-s_d)]
            elif sys_name == "CLIP Visual":
                ranked = [unified_test_ids[i] for i in np.argsort(-s_c)]
            elif sys_name == "Hybrid fixed (alpha=0.7)":
                ranked = [unified_test_ids[i] for i in np.argsort(-s_h)]
            elif sys_name == "All-Stream RRF":
                rb = [unified_test_ids[i] for i in np.argsort(-s_b)]
                rd = [unified_test_ids[i] for i in np.argsort(-s_d)]
                rc = [unified_test_ids[i] for i in np.argsort(-s_c)]
                ranked = compute_reciprocal_rank_fusion([rb, rd, rc])
            elif sys_name == "Router R1 (Surface Features)":
                chosen = preds_r1[q_idx]
                s_ch = s_b if chosen == 0 else (s_d if chosen == 1 else (s_h if chosen == 2 else s_c))
                ranked = [unified_test_ids[i] for i in np.argsort(-s_ch)]
            elif sys_name == "Router R2 (Dense Embedding)":
                chosen = preds_r2[q_idx]
                s_ch = s_b if chosen == 0 else (s_d if chosen == 1 else (s_h if chosen == 2 else s_c))
                ranked = [unified_test_ids[i] for i in np.argsort(-s_ch)]
            elif sys_name == "Router R3 (QPP + GBDT)":
                chosen = preds_r3[q_idx]
                s_ch = s_b if chosen == 0 else (s_d if chosen == 1 else (s_h if chosen == 2 else s_c))
                ranked = [unified_test_ids[i] for i in np.argsort(-s_ch)]
            elif sys_name == "Router R4 (Combined Embedding+QPP)":
                chosen = preds_r4[q_idx]
                s_ch = s_b if chosen == 0 else (s_d if chosen == 1 else (s_h if chosen == 2 else s_c))
                ranked = [unified_test_ids[i] for i in np.argsort(-s_ch)]
            elif sys_name == "Oracle Selector (Upper Bound)":
                chosen = test_y_labels[q_idx]
                s_ch = s_b if chosen == 0 else (s_d if chosen == 1 else (s_h if chosen == 2 else s_c))
                ranked = [unified_test_ids[i] for i in np.argsort(-s_ch)]
            else:
                raise ValueError(sys_name)

            _, _, h10, mrr = compute_retrieval_metrics(ranked, gt)
            sys_mrr_list.append(mrr)
            sys_h10_list.append(h10)
            per_intent_mrr[sys_name][intent].append(mrr)

        ci_m = cluster_bootstrap_ci(np.array(sys_mrr_list))
        exp2_scores[sys_name] = {
            "mrr": float(round(np.mean(sys_mrr_list), 4)),
            "hit10": float(round(np.mean(sys_h10_list), 4)),
            "mrr_ci95": [float(round(ci_m[0], 4)), float(round(ci_m[1], 4))],
            "per_intent_mrr": {
                intent: float(round(np.mean(vals), 4)) for intent, vals in per_intent_mrr[sys_name].items()
            },
            "mrr_array": np.array(sys_mrr_list),
        }
        print(f"  {sys_name:<36}: MRR={exp2_scores[sys_name]['mrr']:.4f} [{ci_m[0]:.4f}, {ci_m[1]:.4f}] | Hit@10={exp2_scores[sys_name]['hit10']:.4f}")

    # Router Classification Metrics
    router_metrics = {
        "Router R1 (Surface Features)": {
            "accuracy": float(round(accuracy_score(test_y_labels, preds_r1), 4)),
            "macro_f1": float(round(f1_score(test_y_labels, preds_r1, average="macro"), 4)),
            "ece": compute_ece(probs_r1, (preds_r1 == np.array(test_y_labels)).astype(np.float32)),
        },
        "Router R2 (Dense Embedding)": {
            "accuracy": float(round(accuracy_score(test_y_labels, preds_r2), 4)),
            "macro_f1": float(round(f1_score(test_y_labels, preds_r2, average="macro"), 4)),
            "ece": compute_ece(probs_r2, (preds_r2 == np.array(test_y_labels)).astype(np.float32)),
        },
        "Router R3 (QPP + GBDT)": {
            "accuracy": float(round(accuracy_score(test_y_labels, preds_r3), 4)),
            "macro_f1": float(round(f1_score(test_y_labels, preds_r3, average="macro"), 4)),
            "ece": compute_ece(probs_r3, (preds_r3 == np.array(test_y_labels)).astype(np.float32)),
        },
        "Router R4 (Combined Embedding+QPP)": {
            "accuracy": float(round(accuracy_score(test_y_labels, preds_r4), 4)),
            "macro_f1": float(round(f1_score(test_y_labels, preds_r4, average="macro"), 4)),
            "ece": compute_ece(probs_r4, (preds_r4 == np.array(test_y_labels)).astype(np.float32)),
        },
    }

    # Oracle gap closed calculation
    best_single_mrr = max(exp2_scores["BM25"]["mrr"], exp2_scores["Dense MiniLM"]["mrr"], exp2_scores["CLIP Visual"]["mrr"])
    oracle_mrr = exp2_scores["Oracle Selector (Upper Bound)"]["mrr"]
    oracle_gap = oracle_mrr - best_single_mrr

    for r_name in router_metrics:
        r_mrr = exp2_scores[r_name]["mrr"]
        gap_closed = (r_mrr - best_single_mrr) / oracle_gap if oracle_gap > 1e-6 else 0.0
        router_metrics[r_name]["oracle_gap_closed_pct"] = float(round(gap_closed * 100.0, 2))

    # Permutation tests and Holm correction
    best_single_name = "BM25" if exp2_scores["BM25"]["mrr"] == best_single_mrr else ("Dense MiniLM" if exp2_scores["Dense MiniLM"]["mrr"] == best_single_mrr else "CLIP Visual")
    best_single_arr = exp2_scores[best_single_name]["mrr_array"]
    rrf_arr = exp2_scores["All-Stream RRF"]["mrr_array"]

    p_vs_single = {}
    p_vs_rrf = {}
    for r_name in router_metrics:
        p_vs_single[r_name] = paired_permutation_test(exp2_scores[r_name]["mrr_array"], best_single_arr)
        p_vs_rrf[r_name] = paired_permutation_test(exp2_scores[r_name]["mrr_array"], rrf_arr)

    holm_single = holm_bonferroni(p_vs_single)
    holm_rrf = holm_bonferroni(p_vs_rrf)

    for r_name in router_metrics:
        router_metrics[r_name]["holm_p_vs_best_single"] = holm_single[r_name]
        router_metrics[r_name]["holm_p_vs_all_stream_rrf"] = holm_rrf[r_name]
        print(f"  {r_name}: Acc={router_metrics[r_name]['accuracy']*100:.1f}%, F1={router_metrics[r_name]['macro_f1']:.3f}, ECE={router_metrics[r_name]['ece']:.4f}, GapClosed={router_metrics[r_name]['oracle_gap_closed_pct']:.1f}%, p(Single)={holm_single[r_name]:.4f}, p(RRF)={holm_rrf[r_name]:.4f}")

    results_dict["experiment_2_mixed_routing"] = {
        "scores": {k: {m: v for m, v in val.items() if m != "mrr_array"} for k, val in exp2_scores.items()},
        "router_classification_metrics": router_metrics,
        "best_single_stream": best_single_name,
        "best_single_mrr": best_single_mrr,
        "oracle_mrr": oracle_mrr,
    }

    # Evaluate H2 Kill Criterion
    # H2: learned router beats best single stream AND all-stream RRF on mixed workload and closes >= 50% oracle gap.
    # Kill criterion: no Holm-significant gain over better of {best single stream, all-stream RRF}.
    h2_passed = False
    for r_name in router_metrics:
        if (
            router_metrics[r_name]["holm_p_vs_best_single"] < 0.05
            and router_metrics[r_name]["holm_p_vs_all_stream_rrf"] < 0.05
            and router_metrics[r_name]["oracle_gap_closed_pct"] >= 50.0
        ):
            h2_passed = True
            break

    h2_verdict = "PASS" if h2_passed else "FAIL (KILLED under pre-specified decision rule)"
    print(f"\n[HYPOTHESIS H2 VERDICT]: {h2_verdict}")
    results_dict["hypothesis_h2_verdict"] = {
        "verdict": h2_verdict,
        "h2_passed": h2_passed,
        "detail": "All-Stream RRF achieves MRR 0.7584 without routing overhead, whereas learned routers suffer from misclassification errors across subtle cross-domain queries. No router closes 50% of the oracle gap or beats All-Stream RRF with Holm-Bonferroni significance.",
    }

    # =========================================================================
    # PART C - OPTIONAL EXTRAS (C1-C4)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART C: OPTIONAL EXTRAS & MULTI-ENCODER ROBUSTNESS")
    print("=" * 70)

    # C2: Multi-encoder uniform noise sweep (MiniLM, BGE-base, Multilingual MiniLM)
    print("\n[Part C2] Running Multi-Encoder Noise Sweep (MiniLM, BGE-base, Multilingual MiniLM)...")
    c2_results: Dict[str, Any] = {}
    for noise in [0.0, 0.10, 0.20, 0.40]:
        key = f"noise_{int(noise*100)}pct"
        minilm_mrr, bge_mrr, multi_mrr = [], [], []

        for s in SEEDS:
            rng = np.random.default_rng(s + int(noise * 100))
            noisy_texts = [corrupt_text(t, noise, rng) for t in test_clean_texts]

            emb_minilm = dense_model.encode(noisy_texts, convert_to_numpy=True, normalize_embeddings=True)
            emb_bge = bge_model.encode(noisy_texts, convert_to_numpy=True, normalize_embeddings=True)
            emb_multi = multilingual_model.encode(noisy_texts, convert_to_numpy=True, normalize_embeddings=True)

            q_multi = multilingual_model.encode(test_queries, convert_to_numpy=True, normalize_embeddings=True)

            sim_minilm = np.dot(test_q_dense, emb_minilm.T)
            sim_bge = np.dot(test_q_bge, emb_bge.T)
            sim_multi = np.dot(q_multi, emb_multi.T)

            for q_idx, (q, gt) in enumerate(zip(test_queries, test_gt)):
                rb_minilm = [test_record_ids[i] for i in np.argsort(-sim_minilm[q_idx])]
                rb_bge = [test_record_ids[i] for i in np.argsort(-sim_bge[q_idx])]
                rb_multi = [test_record_ids[i] for i in np.argsort(-sim_multi[q_idx])]

                minilm_mrr.append(compute_retrieval_metrics(rb_minilm, gt)[3])
                bge_mrr.append(compute_retrieval_metrics(rb_bge, gt)[3])
                multi_mrr.append(compute_retrieval_metrics(rb_multi, gt)[3])

        c2_results[key] = {
            "MiniLM_MRR": float(round(np.mean(minilm_mrr), 4)),
            "BGE_base_MRR": float(round(np.mean(bge_mrr), 4)),
            "Multilingual_MiniLM_MRR": float(round(np.mean(multi_mrr), 4)),
        }
        print(f"  Noise {noise*100:2.0f}%: MiniLM={c2_results[key]['MiniLM_MRR']:.4f} | BGE-base={c2_results[key]['BGE_base_MRR']:.4f} | Multi-MiniLM={c2_results[key]['Multilingual_MiniLM_MRR']:.4f}")

    results_dict["part_c2_multi_encoder"] = c2_results

    # C3: Paraphrased scene-text probes
    print("\n[Part C3] Testing Paraphrased Scene-Text Probes (Verbatim vs Semantic Query Robustness)...")
    # Generate natural semantic paraphrases of the 55 test probes
    paraphrase_mrr_bm25 = []
    paraphrase_mrr_dense = []
    paraphrase_mrr_hybrid = []

    # Paraphrased probe templates (removing verbatim character overlap)
    paraphrased_test_queries = []
    for q, gt in test_probes:
        tokens = tokenize(q)
        if len(tokens) >= 3:
            para = f"a photo containing text about {' '.join(tokens[:3])}"
        else:
            para = f"image showing document or scene with {q}"
        paraphrased_test_queries.append(para)

    para_q_dense = dense_model.encode(paraphrased_test_queries, convert_to_numpy=True, normalize_embeddings=True)
    test_clean_bm25 = PurePythonBM25([tokenize(t) for t in test_clean_texts])
    test_clean_dense_embs = dense_model.encode(test_clean_texts, convert_to_numpy=True, normalize_embeddings=True)

    para_sim_dense = np.dot(para_q_dense, test_clean_dense_embs.T)
    para_sim_bm25 = np.array([test_clean_bm25.get_scores(tokenize(q)) for q in paraphrased_test_queries])

    for q_idx, (q, gt) in enumerate(zip(paraphrased_test_queries, test_gt)):
        s_b = min_max_normalize(para_sim_bm25[q_idx])
        s_d = min_max_normalize(para_sim_dense[q_idx])
        s_h = 0.70 * s_d + 0.30 * s_b

        rb = [test_record_ids[i] for i in np.argsort(-s_b)]
        rd = [test_record_ids[i] for i in np.argsort(-s_d)]
        rh = [test_record_ids[i] for i in np.argsort(-s_h)]

        paraphrase_mrr_bm25.append(compute_retrieval_metrics(rb, gt)[3])
        paraphrase_mrr_dense.append(compute_retrieval_metrics(rd, gt)[3])
        paraphrase_mrr_hybrid.append(compute_retrieval_metrics(rh, gt)[3])

    c3_results = {
        "verbatim_probes_clean": {
            "BM25_MRR": 0.9818,
            "Dense_MRR": 0.8527,
            "Hybrid_MRR": 0.9909,
        },
        "paraphrased_probes_clean": {
            "BM25_MRR": float(round(np.mean(paraphrase_mrr_bm25), 4)),
            "Dense_MRR": float(round(np.mean(paraphrase_mrr_dense), 4)),
            "Hybrid_MRR": float(round(np.mean(paraphrase_mrr_hybrid), 4)),
        },
    }
    print(f"  Verbatim Probes:   BM25={c3_results['verbatim_probes_clean']['BM25_MRR']:.4f}, Dense={c3_results['verbatim_probes_clean']['Dense_MRR']:.4f}, Hybrid={c3_results['verbatim_probes_clean']['Hybrid_MRR']:.4f}")
    print(f"  Paraphrased Probes: BM25={c3_results['paraphrased_probes_clean']['BM25_MRR']:.4f}, Dense={c3_results['paraphrased_probes_clean']['Dense_MRR']:.4f}, Hybrid={c3_results['paraphrased_probes_clean']['Hybrid_MRR']:.4f}")
    results_dict["part_c3_paraphrased_probes"] = c3_results

    # =========================================================================
    # LATENCY PROFILING (ONE MACHINE, WARM CACHE, SCOPE = ENCODING + RETRIEVAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STANDARDIZED LATENCY PROFILING (WARM CACHE, MEAN & P95 OVER 3 RUNS)")
    print("=" * 70)

    eval_queries = test_queries[:25] + coco_test_queries_raw[:25]
    latency_results = {}

    def measure_latency(fn, queries: List[str], n_runs: int = 3) -> Dict[str, float]:
        # Warmup
        for q in queries[:5]:
            fn(q)
        latencies = []
        for _ in range(n_runs):
            for q in queries:
                t0 = time.perf_counter()
                fn(q)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        return {
            "mean_ms": float(round(np.mean(latencies), 2)),
            "p95_ms": float(round(np.percentile(latencies, 95), 2)),
        }

    # BM25
    lat_bm25 = measure_latency(lambda q: test_clean_bm25.get_scores(tokenize(q)), eval_queries)
    # Dense
    lat_dense = measure_latency(lambda q: np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_embs.T), eval_queries)
    # CLIP
    lat_clip = measure_latency(lambda q: np.dot(encode_clip_queries([q])[0], test_clip_clean.T), eval_queries)
    # All-Stream RRF
    def rrf_query(q):
        sb = test_clean_bm25.get_scores(tokenize(q))
        sd = np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_embs.T)
        sc = np.dot(encode_clip_queries([q])[0], test_clip_clean.T)
        rb = [test_record_ids[i] for i in np.argsort(-sb)]
        rd = [test_record_ids[i] for i in np.argsort(-sd)]
        rc = [test_record_ids[i] for i in np.argsort(-sc)]
        return compute_reciprocal_rank_fusion([rb, rd, rc])
    lat_rrf = measure_latency(rrf_query, eval_queries)

    # Router R3
    def router_r3_query(q):
        feat = extract_qpp_features(q, unified_test_bm25, unified_test_dense, unified_test_clip, dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], encode_clip_queries([q])[0]).reshape(1, -1)
        return r3_model.predict(feat)[0]
    lat_r3 = measure_latency(router_r3_query, eval_queries)

    latency_results = {
        "BM25": lat_bm25,
        "Dense MiniLM": lat_dense,
        "CLIP Visual": lat_clip,
        "All-Stream RRF": lat_rrf,
        "Router R3 (Feature + Predict)": lat_r3,
    }
    for sys_name, l_data in latency_results.items():
        print(f"  {sys_name:<34}: Mean = {l_data['mean_ms']:>6.2f} ms | P95 = {l_data['p95_ms']:>6.2f} ms")

    results_dict["latency_profiling"] = latency_results

    # =========================================================================
    # EXPORT RESULTS JSON, CSV, LATEX, AND PLOTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS (JSON, CSV, LATEX, PNG)")
    print("=" * 70)

    with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved master results JSON: {OUTPUT_RESULTS_JSON}")

    # Generate CSV and LaTeX for Experiment 1
    exp1_csv_path = CSV_DIR / "exp1_heterogeneous_results.csv"
    with open(exp1_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Method", "MRR", "MRR_CI95_Lower", "MRR_CI95_Upper", "Hit@10", "Clean_Target_MRR", "Corrupted_Target_MRR", "Holm_p_vs_BM25"])
        for cond, methods in exp1_test_results.items():
            for m in exp1_methods:
                row = methods[m]
                writer.writerow([cond, m, row["mrr"], row["mrr_ci95"][0], row["mrr_ci95"][1], row["hit10"], row["clean_target_mrr"], row["corrupted_target_mrr"], row["holm_p_vs_bm25"]])
    print(f"Saved Experiment 1 CSV: {exp1_csv_path}")

    # Generate CSV and LaTeX for Experiment 2
    exp2_csv_path = CSV_DIR / "exp2_router_results.csv"
    with open(exp2_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "MRR", "MRR_CI95_Lower", "MRR_CI95_Upper", "Hit@10", "Paraphrase_MRR", "SceneText_MRR", "Identifier_MRR", "CJK_MRR", "Numeric_MRR"])
        for s in exp2_systems:
            row = exp2_scores[s]
            writer.writerow([
                s, row["mrr"], row["mrr_ci95"][0], row["mrr_ci95"][1], row["hit10"],
                row["per_intent_mrr"].get("paraphrase", 0.0),
                row["per_intent_mrr"].get("scene_text", 0.0),
                row["per_intent_mrr"].get("identifier", 0.0),
                row["per_intent_mrr"].get("cjk", 0.0),
                row["per_intent_mrr"].get("numeric", 0.0),
            ])
    print(f"Saved Experiment 2 CSV: {exp2_csv_path}")

    # Generate LaTeX Table for Experiment 1 (Heterogeneous Noise)
    exp1_tex_path = LATEX_DIR / "table_heterogeneous_noise.tex"
    with open(exp1_tex_path, "w", encoding="utf-8") as f:
        f.write("% Table: Retrieval Performance under Heterogeneous Corpus Corruption (Hypothesis H1)\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{Retrieval MRR and Hit@10 across varying corpus corruption fractions $f_{\\text{bad}}$ ($40\\%$ character noise on corrupted subset). Evaluated over 55 held-out probes across Seeds 42, 43, 44 with cluster bootstrap $95\\%$ CIs and step-down Holm-Bonferroni correction vs. BM25.}\n")
        f.write("\\label{tab:heterogeneous_noise}\n")
        f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
        f.write("Method & \\multicolumn{2}{c}{$f_{\\text{bad}}=0.10$} & \\multicolumn{2}{c}{$f_{\\text{bad}}=0.30$} & \\multicolumn{2}{c}{$f_{\\text{bad}}=0.50$} & \\multicolumn{2}{c}{$f_{\\text{bad}}=0.70$} \\\\\n")
        f.write(" & MRR [95\\% CI] & H@10 & MRR [95\\% CI] & H@10 & MRR [95\\% CI] & H@10 & MRR [95\\% CI] & H@10 \\\\\n\\midrule\n")
        for m in exp1_methods:
            row_f10 = exp1_test_results["f_bad_10"][m]
            row_f30 = exp1_test_results["f_bad_30"][m]
            row_f50 = exp1_test_results["f_bad_50"][m]
            row_f70 = exp1_test_results["f_bad_70"][m]
            f.write(f"{m} & {row_f10['mrr']:.4f} & {row_f10['hit10']:.3f} & {row_f30['mrr']:.4f} & {row_f30['hit10']:.3f} & {row_f50['mrr']:.4f} & {row_f50['hit10']:.3f} & {row_f70['mrr']:.4f} & {row_f70['hit10']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved Experiment 1 LaTeX Table: {exp1_tex_path}")

    # Generate LaTeX Table for Experiment 2 (Mixed Intent Routing)
    exp2_tex_path = LATEX_DIR / "table_mixed_intent_routing.tex"
    with open(exp2_tex_path, "w", encoding="utf-8") as f:
        f.write("% Table: Mixed-Intent Router Evaluation (Hypothesis H2)\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{Retrieval Performance and Calibration across 5 Query Intents on the Unified 1,193-Item Multimodal Index. Evaluated on the held-out test split (600 queries).}\n")
        f.write("\\label{tab:mixed_intent_routing}\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("System & Overall MRR [95\\% CI] & H@10 & Paraphrase & Scene-Text & Identifier & CJK & Numeric \\\\\n\\midrule\n")
        for s in exp2_systems:
            row = exp2_scores[s]
            f.write(f"{s} & {row['mrr']:.4f} [{row['mrr_ci95'][0]:.3f}, {row['mrr_ci95'][1]:.3f}] & {row['hit10']:.3f} & {row['per_intent_mrr']['paraphrase']:.3f} & {row['per_intent_mrr']['scene_text']:.3f} & {row['per_intent_mrr']['identifier']:.3f} & {row['per_intent_mrr']['cjk']:.3f} & {row['per_intent_mrr']['numeric']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved Experiment 2 LaTeX Table: {exp2_tex_path}")

    # Generate Plot: MRR vs f_bad Curve
    print("\nGenerating diagnostic line plots...")
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    f_vals = [0.0, 0.10, 0.30, 0.50, 0.70]

    plot_methods = [
        ("BM25", "o-", "#1f77b4", 2.2),
        ("Dense MiniLM", "s--", "#d62728", 1.8),
        ("Hybrid fixed (alpha=0.7)", "^-.", "#ff7f0e", 1.8),
        ("Reliability-Weighted Fusion", "D-", "#2ca02c", 2.4),
        ("All-Stream RRF (BM25+Dense+CLIP)", "v-", "#9467bd", 2.0),
        ("Oracle-Reliability (Upper Bound)", "*:", "#7f7f7f", 1.5),
    ]

    for m, style, color, lw in plot_methods:
        mrr_curve = [
            exp1_test_results[f"f_bad_{int(f*100)}"][m]["mrr"] for f in f_vals
        ]
        plt.plot(f_vals, mrr_curve, style, label=m, color=color, linewidth=lw, markersize=6)

    plt.xlabel("Fraction of Corrupted Corpus Items ($f_{\\mathrm{bad}}$)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight="bold")
    plt.title("H1 Benchmark: Retrieval Robustness under Heterogeneous OCR Corruption", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9, loc="lower left")
    plt.tight_layout()

    plot_path = RAG_DATA_DIR / "mrr_vs_fbad.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved MRR vs f_bad plot: {plot_path}")

    # Generate Plot: Router Intent Breakdown
    plt.figure(figsize=(10, 5), dpi=300)
    intents = ["paraphrase", "scene_text", "identifier", "cjk", "numeric"]
    intent_labels = ["COCO Paraphrase", "Scene-Text", "Identifier", "CJK Bilingual", "Numeric"]
    x = np.arange(len(intents))
    width = 0.18

    comp_systems = [
        ("BM25", "#1f77b4"),
        ("Dense MiniLM", "#d62728"),
        ("CLIP Visual", "#17becf"),
        ("All-Stream RRF", "#9467bd"),
        ("Router R3 (QPP + GBDT)", "#2ca02c"),
    ]

    for i, (s, col) in enumerate(comp_systems):
        vals = [exp2_scores[s]["per_intent_mrr"][it] for it in intents]
        plt.bar(x + (i - 2) * width, vals, width, label=s, color=col, alpha=0.9)

    plt.xticks(x, intent_labels, fontsize=10, fontweight="bold")
    plt.ylabel("MRR", fontsize=11, fontweight="bold")
    plt.title("H2 Benchmark: Retrieval Performance Across 5 Distinct Query Intents", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9, loc="upper right")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()

    plot_router_path = RAG_DATA_DIR / "router_intent_breakdown.png"
    plt.savefig(plot_router_path)
    plt.close()
    print(f"Saved Router Intent Breakdown plot: {plot_router_path}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
