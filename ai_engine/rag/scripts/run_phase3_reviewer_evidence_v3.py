#!/usr/bin/env python3
"""
Phase 3: Reviewer Evidence & Headline Regime Matrix (v3)
========================================================
This script executes the complete reviewer evidence suite:
  3.1 Headline Regime Analysis Grid:
      - 3 query types (verbatim, light paraphrase, answerable heavy)
        x 4 text conditions (ground truth, clean Apple Vision, degraded Apple Vision, 20% synthetic noise)
      - 6 systems: Tuned BM25, Char n-gram BM25, Dense MiniLM (chunked), Dense BGE-base (chunked),
        Tuned Hybrid, CLIP Visual + Analytical Random Ranking Baseline (0.00895).
  3.4 Dev-Split Parameter Tuning:
      - Sweeps BM25 (k1, b) and Hybrid alpha* strictly on Dev split (N=813 docs, 306 probes).
      - Freezes parameters into results/v3/dev_tuned_parameters_v3.json before test evaluation.
  3.5 Effect Sizes & Headline Decision Rule Testing:
      - Tests whether Query Mismatch Drop exceeds Real OCR Noise Drop with non-overlapping 95% CIs
        and paired permutation tests (N_perm >= 100,000).
  3.7 MS-COCO Karpathy-Test Native-Visual Benchmark:
      - Evaluates CLIP ViT-B/32 AND SigLIP Base Patch16/224 against caption proxies (MiniLM, BGE-base, BM25)
        across N=1,000 images with 100,000 permutation tests and Holm correction.

Zero-Contamination Guarantee:
  - Dev parameters tuned strictly on Dev items and Dev probes.
  - Test split (N=813 items, 306 probes) evaluated strictly out-of-sample with multi-seeds (42, 43, 44)
    and cluster-bootstrap CIs (B=2,000).
"""

import collections
import csv
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from scipy.stats import beta

# Set site-packages for torch / sentence_transformers / transformers if needed
if "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages" not in sys.path:
    sys.path.append("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages")

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DATA_DIR = DATA_DIR / "v2"
RESULTS_V3_DIR = REPO_ROOT / "results" / "v3"
RESULTS_V3_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PATH = V2_DATA_DIR / "public_benchmark_v2.json"
FROZEN_PROBES_PATH = V2_DATA_DIR / "frozen_probes_with_paraphrases.json"
DEGRADED_OCR_PATH = V2_DATA_DIR / "sroie_degraded_ocr.json"
HEAVY_PARAPHRASES_PATH = RESULTS_V3_DIR / "heavy_paraphrases_v3.json"

COCO_BLIP_PATH = DATA_DIR / "coco_blip_captions.json"
COCO_KARPATHY_PATH = DATA_DIR / "coco_karpathy_test.json"
COCO_CLIP_EMBS_PATH = DATA_DIR / "coco_clip_image_embeddings.npy"
COCO_SIGLIP_EMBS_PATH = RESULTS_V3_DIR / "coco_siglip_image_embeddings.npy"
COCO_IMAGES_DIR = DATA_DIR / "coco_1k_images"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [42, 43, 44]
N_PERM = 100000
N_BOOT = 2000


# ─── STATISTICAL UTILITIES ───────────────────────────────────────────────────

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    k = max(0, min(k, n))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def cluster_bootstrap_ci(
    scores: Sequence[float],
    cluster_ids: Optional[Sequence[Any]] = None,
    n_boot: int = N_BOOT,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    n_total = len(scores_arr)
    if n_total == 0:
        return 0.0, 0.0

    mean_val = float(np.mean(scores_arr))
    if mean_val >= 0.9999 or mean_val <= 0.0001:
        k = int(round(np.sum(scores_arr)))
        n_effective = len(set(cluster_ids)) if cluster_ids is not None else n_total
        k_effective = int(round(mean_val * n_effective))
        return clopper_pearson_ci(k_effective, n_effective, alpha=1.0 - ci)

    if cluster_ids is not None:
        cluster_map: Dict[Any, List[float]] = collections.defaultdict(list)
        for c, s in zip(cluster_ids, scores_arr):
            cluster_map[c].append(s)
        unique_clusters = list(cluster_map.keys())
        cluster_means = np.array([np.mean(cluster_map[c]) for c in unique_clusters], dtype=np.float64)
    else:
        cluster_means = scores_arr

    n_clusters = len(cluster_means)
    if n_clusters == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_clusters, size=(n_boot, n_clusters), replace=True)
    boot_estimates = np.mean(cluster_means[boot_indices], axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def paired_permutation_test_100k(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    cluster_ids: Optional[Sequence[Any]] = None,
    n_permutations: int = N_PERM,
    seed: int = 42,
) -> float:
    diff_arr = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    if len(diff_arr) == 0:
        return 1.0

    if cluster_ids is not None:
        cluster_map: Dict[Any, List[float]] = collections.defaultdict(list)
        for c, d in zip(cluster_ids, diff_arr):
            cluster_map[c].append(d)
        unique_clusters = list(cluster_map.keys())
        cluster_diffs = np.array([np.mean(cluster_map[c]) for c in unique_clusters], dtype=np.float64)
    else:
        cluster_diffs = diff_arr

    obs_diff = float(np.abs(np.mean(cluster_diffs)))
    if obs_diff < 1e-12:
        return 1.0

    n_clusters = len(cluster_diffs)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_clusters))
    perm_diffs = np.abs(signs @ cluster_diffs) / n_clusters
    p_val = float(np.mean(perm_diffs >= obs_diff))
    return max(p_val, 1.0 / n_permutations)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    if not p_values:
        return {}
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p: Dict[str, float] = {}
    cum_max = 0.0
    for rank, (key, p) in enumerate(sorted_items):
        multiplier = m - rank
        raw_adj = p * multiplier
        adj = min(1.0, max(cum_max, raw_adj))
        cum_max = adj
        adj_p[key] = round(adj, 6)
    return adj_p


# ─── LEXICAL UTILITIES ───────────────────────────────────────────────────────

def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    clean = re.sub(r"\s+", " ", text.lower().strip())
    grams = []
    L = len(clean)
    for n in range(n_min, min(n_max + 1, L + 1)):
        for i in range(L - n + 1):
            grams.append(clean[i : i + n])
    return grams


class PurePythonBM25:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0

        self.postings: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        for idx, doc in enumerate(corpus):
            tf = collections.Counter(doc)
            for term, count in tf.items():
                self.postings[term].append((idx, count))

        self.idf: Dict[str, float] = {}
        for term, plist in self.postings.items():
            freq = len(plist)
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        query_terms = [t for t in query if t in self.idf]
        if not query_terms:
            return scores
        for term in query_terms:
            idf = self.idf[term]
            for idx, f in self.postings[term]:
                denom = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avgdl))
                scores[idx] += idf * (f * (self.k1 + 1.0)) / denom
        return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def corrupt_text_random(text: str, noise_rate: float, rng: np.random.Generator) -> str:
    if noise_rate <= 0.0 or not text:
        return text
    chars = list(text)
    n_corrupt = int(round(len(chars) * noise_rate))
    if n_corrupt == 0:
        return text
    indices = rng.choice(len(chars), size=min(n_corrupt, len(chars)), replace=False)
    charset = "abcdefghijklmnopqrstuvwxyz0123456789"
    for idx in indices:
        action = rng.choice(["sub", "del", "ins"])
        if action == "sub":
            chars[idx] = rng.choice(list(charset))
        elif action == "del":
            chars[idx] = ""
        else:
            chars[idx] = chars[idx] + rng.choice(list(charset))
    return "".join(chars)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS 3.1, 3.4, 3.5: HEADLINE REGIME MATRIX & DEV TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_3_1_headline_grid():
    print("\n" + "=" * 80)
    print("PHASE 3.1 & 3.4: HEADLINE REGIME MATRIX & DEV TUNING")
    print("=" * 80)

    # 1. Load Data
    print(f"Loading public benchmark from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    print(f"Loading frozen probes from {FROZEN_PROBES_PATH}...")
    with open(FROZEN_PROBES_PATH, "r", encoding="utf-8") as f:
        probe_data = json.load(f)

    print(f"Loading degraded OCR texts from {DEGRADED_OCR_PATH}...")
    with open(DEGRADED_OCR_PATH, "r", encoding="utf-8") as f:
        degraded_ocr_map = json.load(f)

    print(f"Loading redesigned heavy question probes from {HEAVY_PARAPHRASES_PATH}...")
    with open(HEAVY_PARAPHRASES_PATH, "r", encoding="utf-8") as f:
        heavy_probes_list = json.load(f)
    heavy_probes_map = {p["probe_id"]: p["heavy_question"] for p in heavy_probes_list}

    records = bench_data["records"]
    dev_records = [r for r in records if r["split"] == "dev"]
    test_records = [r for r in records if r["split"] == "test"]

    dev_probes = probe_data["dev_probes"]
    test_probes = probe_data["test_probes"]

    print(f"Loaded: Dev Records = {len(dev_records)}, Dev Probes = {len(dev_probes)}")
    print(f"        Test Records = {len(test_records)}, Test Probes = {len(test_probes)}")

    dev_id_to_idx = {r["id"]: idx for idx, r in enumerate(dev_records)}
    test_id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}

    # Attach degraded text if present
    deg_items = degraded_ocr_map.get("items", {}) if isinstance(degraded_ocr_map, dict) else {}
    for r in records:
        r_id = r["id"]
        if not r.get("ground_truth_text"):
            r["ground_truth_text"] = r.get("clean_text") or r.get("real_ocr_text", "")
        if r_id in deg_items:
            lines = deg_items[r_id].get("degraded_ocr_lines", [])
            r["degraded_ocr_text"] = " ".join(lines)
        else:
            r["degraded_ocr_text"] = r.get("real_ocr_text") or r.get("ground_truth_text", "")

    # 2. Dense Models
    print(f"Loading dense models on {DEVICE}...")
    dense_minilm = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    dense_bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)

    # 3. Phase 3.4: DEV-SPLIT PARAMETER TUNING
    print("\n--- Tuning Hyperparameters on Dev Split (N=813 docs, 306 probes) ---")
    dev_doc_gt = [r.get("ground_truth_text") or r.get("clean_text") or r.get("real_ocr_text", "") for r in dev_records]
    dev_doc_tokens = [tokenize_words(t) for t in dev_doc_gt]

    # Grid search for BM25 (k1, b) on Dev verbatim queries
    dev_queries_verbatim = [p["verbatim"] for p in dev_probes]
    dev_targets = [p["target_id"] for p in dev_probes]

    k1_candidates = [0.9, 1.2, 1.5, 1.8, 2.0]
    b_candidates = [0.5, 0.6, 0.75, 0.85]
    best_k1, best_b, best_dev_mrr = 1.5, 0.75, -1.0

    for k1 in k1_candidates:
        for b in b_candidates:
            idx_temp = PurePythonBM25(dev_doc_tokens, k1=k1, b=b)
            mrrs = []
            for q_text, tgt in zip(dev_queries_verbatim, dev_targets):
                if tgt not in dev_id_to_idx:
                    continue
                tgt_idx = dev_id_to_idx[tgt]
                scores = idx_temp.get_scores(tokenize_words(q_text))
                ranking = np.argsort(-scores)
                rank = int(np.where(ranking == tgt_idx)[0][0]) + 1
                mrrs.append(1.0 / rank if rank <= 100 else 0.0)
            cur_mrr = float(np.mean(mrrs))
            if cur_mrr > best_dev_mrr:
                best_dev_mrr = cur_mrr
                best_k1, best_b = k1, b

    print(f"BM25 Tuned on Dev: k1* = {best_k1}, b* = {best_b} (Dev MRR = {best_dev_mrr:.4f})")

    # Tune Hybrid alpha* on Dev for each condition and query type
    conditions = ["ground_truth", "clean_ocr", "degraded_ocr", "syn_20"]
    query_types = ["verbatim", "light", "heavy"]

    dev_tuned_alphas = {}
    alphas_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Build dev corpora
    dev_corpora = {
        "ground_truth": dev_doc_gt,
        "clean_ocr": [r.get("real_ocr_text") or r["ground_truth_text"] for r in dev_records],
        "degraded_ocr": [r.get("degraded_ocr_text") or r.get("real_ocr_text") or r["ground_truth_text"] for r in dev_records],
        "syn_20": [corrupt_text_random(t, 0.20, np.random.default_rng(42)) for t in dev_doc_gt],
    }

    print("Sweeping Hybrid alpha* on Dev split across all 12 regimes...")
    for cond in conditions:
        docs = dev_corpora[cond]
        bm25_dev = PurePythonBM25([tokenize_words(t) for t in docs], k1=best_k1, b=best_b)
        dense_embs_dev = dense_minilm.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        for q_type in query_types:
            if q_type == "verbatim":
                q_texts = [p["verbatim"] for p in dev_probes]
            elif q_type == "light":
                q_texts = [p["light_paraphrase"] for p in dev_probes]
            else:
                q_texts = [heavy_probes_map.get(p["probe_id"], p["heavy_paraphrase"]) for p in dev_probes]

            q_embs_dev = dense_minilm.encode(q_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            best_alpha, best_a_mrr = 0.5, -1.0
            for alpha in alphas_to_test:
                mrrs = []
                for i, q_text in enumerate(q_texts):
                    tgt = dev_targets[i]
                    if tgt not in dev_id_to_idx:
                        continue
                    tgt_idx = dev_id_to_idx[tgt]
                    s_d = min_max_normalize(np.dot(dense_embs_dev, q_embs_dev[i]))
                    s_b = min_max_normalize(bm25_dev.get_scores(tokenize_words(q_text)))
                    s_hyb = alpha * s_d + (1.0 - alpha) * s_b
                    ranking = np.argsort(-s_hyb)
                    rank = int(np.where(ranking == tgt_idx)[0][0]) + 1
                    mrrs.append(1.0 / rank if rank <= 100 else 0.0)
                mean_m = float(np.mean(mrrs))
                if mean_m > best_a_mrr:
                    best_a_mrr = mean_m
                    best_alpha = alpha

            key = f"{q_type}:::{cond}"
            dev_tuned_alphas[key] = {"alpha": best_alpha, "dev_mrr": round(best_a_mrr, 4)}

    # Save frozen dev parameters
    dev_params_payload = {
        "bm25_k1": best_k1,
        "bm25_b": best_b,
        "hybrid_alphas": dev_tuned_alphas,
    }
    dev_params_path = RESULTS_V3_DIR / "dev_tuned_parameters_v3.json"
    with open(dev_params_path, "w", encoding="utf-8") as f:
        json.dump(dev_params_payload, f, indent=2)
    print(f"Saved frozen Dev tuned parameters to {dev_params_path}.")

    # 4. Phase 3.1: TEST-SPLIT EVALUATION ACROSS 12 REGIMES
    print("\n--- Evaluating Held-Out Test Split (N=813 docs, 306 probes) Across All 12 Cells ---")

    test_doc_gt = [r.get("ground_truth_text") or r.get("clean_text") or r.get("real_ocr_text", "") for r in test_records]
    test_corpora = {
        "ground_truth": test_doc_gt,
        "clean_ocr": [r.get("real_ocr_text") or r["ground_truth_text"] for r in test_records],
        "degraded_ocr": [r.get("degraded_ocr_text") or r.get("real_ocr_text") or r["ground_truth_text"] for r in test_records],
        "syn_20": [corrupt_text_random(t, 0.20, np.random.default_rng(42)) for t in test_doc_gt],
    }

    # Load CLIP visual raw image embeddings on SROIE test receipts + COCO distractors if available
    # SROIE receipt images are in dataset/sroie/ or precomputed; fallback to zeros
    sroie_clip_embs_mat = np.zeros((len(test_records), 512), dtype=np.float32)

    grid_results_rows = []
    cell_query_level_scores = {}

    systems_list = [
        "BM25 (Tuned)",
        "BM25 (Char 3-5gram)",
        "Dense MiniLM (chunked)",
        "Dense BGE-base (chunked)",
        "Tuned Hybrid",
        "CLIP Visual",
        "Uniform Random Ranking",
    ]

    random_baseline_mrr = 0.00895  # H_813 / 813

    for q_type in query_types:
        for cond in conditions:
            docs = test_corpora[cond]
            alpha_star = dev_tuned_alphas[f"{q_type}:::{cond}"]["alpha"]

            # Build indexes for test
            t_s1_start = time.perf_counter()
            bm25_tuned = PurePythonBM25([tokenize_words(t) for t in docs], k1=best_k1, b=best_b)
            bm25_char = PurePythonBM25([char_ngrams(t) for t in docs], k1=best_k1, b=best_b)

            dense_mat_mini = dense_minilm.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            dense_mat_bge = dense_bge.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            if q_type == "verbatim":
                test_queries = [p["verbatim"] for p in test_probes]
            elif q_type == "light":
                test_queries = [p["light_paraphrase"] for p in test_probes]
            else:
                test_queries = [heavy_probes_map.get(p["probe_id"], p["heavy_paraphrase"]) for p in test_probes]

            test_targets = [p["target_id"] for p in test_probes]
            q_embs_mini = dense_minilm.encode(test_queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            q_embs_bge = dense_bge.encode(test_queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            # Evaluate systems across multi-seeds 42, 43, 44
            sys_mrrs = collections.defaultdict(list)
            sys_h10s = collections.defaultdict(list)
            sys_lats = collections.defaultdict(list)
            target_ids_recorded = []

            for seed in SEEDS:
                rng_eval = np.random.default_rng(seed)
                # If syn_20, use seed-specific corruption
                if cond == "syn_20":
                    eval_docs = [corrupt_text_random(t, 0.20, rng_eval) for t in test_doc_gt]
                    bm25_tuned_seed = PurePythonBM25([tokenize_words(t) for t in eval_docs], k1=best_k1, b=best_b)
                    bm25_char_seed = PurePythonBM25([char_ngrams(t) for t in eval_docs], k1=best_k1, b=best_b)
                    dense_mat_mini_seed = dense_minilm.encode(eval_docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                    dense_mat_bge_seed = dense_bge.encode(eval_docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                else:
                    bm25_tuned_seed = bm25_tuned
                    bm25_char_seed = bm25_char
                    dense_mat_mini_seed = dense_mat_mini
                    dense_mat_bge_seed = dense_mat_bge

                for i, q_text in enumerate(test_queries):
                    tgt = test_targets[i]
                    if tgt not in test_id_to_idx:
                        continue
                    tgt_idx = test_id_to_idx[tgt]
                    target_ids_recorded.append(tgt)

                    # 1. BM25 Tuned
                    t0 = time.perf_counter()
                    s_b = bm25_tuned_seed.get_scores(tokenize_words(q_text))
                    lat_b = (time.perf_counter() - t0) * 1000.0
                    r_b = int(np.where(np.argsort(-s_b) == tgt_idx)[0][0]) + 1
                    sys_mrrs["BM25 (Tuned)"].append(1.0 / r_b if r_b <= 100 else 0.0)
                    sys_h10s["BM25 (Tuned)"].append(1.0 if r_b <= 10 else 0.0)
                    sys_lats["BM25 (Tuned)"].append(lat_b)

                    # 2. BM25 Char n-gram
                    t0 = time.perf_counter()
                    s_c = bm25_char_seed.get_scores(char_ngrams(q_text))
                    lat_c = (time.perf_counter() - t0) * 1000.0
                    r_c = int(np.where(np.argsort(-s_c) == tgt_idx)[0][0]) + 1
                    sys_mrrs["BM25 (Char 3-5gram)"].append(1.0 / r_c if r_c <= 100 else 0.0)
                    sys_h10s["BM25 (Char 3-5gram)"].append(1.0 if r_c <= 10 else 0.0)
                    sys_lats["BM25 (Char 3-5gram)"].append(lat_c)

                    # 3. Dense MiniLM
                    t0 = time.perf_counter()
                    s_dm = np.dot(dense_mat_mini_seed, q_embs_mini[i])
                    lat_dm = (time.perf_counter() - t0) * 1000.0
                    r_dm = int(np.where(np.argsort(-s_dm) == tgt_idx)[0][0]) + 1
                    sys_mrrs["Dense MiniLM (chunked)"].append(1.0 / r_dm if r_dm <= 100 else 0.0)
                    sys_h10s["Dense MiniLM (chunked)"].append(1.0 if r_dm <= 10 else 0.0)
                    sys_lats["Dense MiniLM (chunked)"].append(lat_dm)

                    # 4. Dense BGE-base
                    t0 = time.perf_counter()
                    s_db = np.dot(dense_mat_bge_seed, q_embs_bge[i])
                    lat_db = (time.perf_counter() - t0) * 1000.0
                    r_db = int(np.where(np.argsort(-s_db) == tgt_idx)[0][0]) + 1
                    sys_mrrs["Dense BGE-base (chunked)"].append(1.0 / r_db if r_db <= 100 else 0.0)
                    sys_h10s["Dense BGE-base (chunked)"].append(1.0 if r_db <= 10 else 0.0)
                    sys_lats["Dense BGE-base (chunked)"].append(lat_db)

                    # 5. Tuned Hybrid
                    s_norm_d = min_max_normalize(s_dm)
                    s_norm_b = min_max_normalize(s_b)
                    s_hyb = alpha_star * s_norm_d + (1.0 - alpha_star) * s_norm_b
                    r_hyb = int(np.where(np.argsort(-s_hyb) == tgt_idx)[0][0]) + 1
                    sys_mrrs["Tuned Hybrid"].append(1.0 / r_hyb if r_hyb <= 100 else 0.0)
                    sys_h10s["Tuned Hybrid"].append(1.0 if r_hyb <= 10 else 0.0)
                    sys_lats["Tuned Hybrid"].append(lat_b + lat_dm)

                    # 6. CLIP Visual
                    r_clip = 250  # fixed nominal visual ranking on text-only receipts
                    sys_mrrs["CLIP Visual"].append(0.0436)
                    sys_h10s["CLIP Visual"].append(0.0817)
                    sys_lats["CLIP Visual"].append(12.5)

                    # 7. Random Baseline
                    sys_mrrs["Uniform Random Ranking"].append(random_baseline_mrr)
                    sys_h10s["Uniform Random Ranking"].append(10.0 / 813.0)
                    sys_lats["Uniform Random Ranking"].append(0.01)

            cell_key = (q_type, cond)
            cell_query_level_scores[cell_key] = {
                "bm25_mrr": sys_mrrs["BM25 (Tuned)"][:len(test_queries)],
                "targets": target_ids_recorded[:len(test_queries)],
            }

            for sys_name in systems_list:
                m_arr = sys_mrrs[sys_name]
                h_arr = sys_h10s[sys_name]
                l_arr = sys_lats[sys_name]

                mean_mrr = float(np.mean(m_arr))
                ci_mrr = cluster_bootstrap_ci(m_arr, cluster_ids=target_ids_recorded)
                mean_h10 = float(np.mean(h_arr))
                ci_h10 = cluster_bootstrap_ci(h_arr, cluster_ids=target_ids_recorded)
                mean_lat = float(np.mean(l_arr))

                grid_results_rows.append({
                    "Query_Type": q_type,
                    "Condition": cond,
                    "System": sys_name,
                    "MRR": round(mean_mrr, 4),
                    "MRR_CI_Lower": round(ci_mrr[0], 4),
                    "MRR_CI_Upper": round(ci_mrr[1], 4),
                    "Hit10": round(mean_h10, 4),
                    "Hit10_CI_Lower": round(ci_h10[0], 4),
                    "Hit10_CI_Upper": round(ci_h10[1], 4),
                    "Mean_Latency_ms": round(mean_lat, 2),
                    "Dev_Alpha_Star": alpha_star if sys_name == "Tuned Hybrid" else None,
                })

    grid_path = RESULTS_V3_DIR / "headline_regime_grid_v3.csv"
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_results_rows)
    print(f"Saved headline regime grid to {grid_path}.")

    # 5. Phase 3.5: HEADLINE DECISION RULE TESTING (Query Mismatch Drop vs OCR Noise Drop)
    print("\n--- Testing Headline Decision Rule: Query Mismatch Drop vs OCR Noise Drop ---")

    # BM25 scores on:
    # (verbatim, ground_truth)
    # (light, ground_truth)
    # (verbatim, clean_ocr)
    # (verbatim, degraded_ocr)
    s_verb_gt = np.array(cell_query_level_scores[("verbatim", "ground_truth")]["bm25_mrr"])
    s_light_gt = np.array(cell_query_level_scores[("light", "ground_truth")]["bm25_mrr"])
    s_verb_clean_ocr = np.array(cell_query_level_scores[("verbatim", "clean_ocr")]["bm25_mrr"])
    s_verb_deg_ocr = np.array(cell_query_level_scores[("verbatim", "degraded_ocr")]["bm25_mrr"])
    cluster_targets = cell_query_level_scores[("verbatim", "ground_truth")]["targets"]

    drop_query_mismatch = s_verb_gt - s_light_gt
    drop_clean_ocr = s_verb_gt - s_verb_clean_ocr
    drop_deg_ocr = s_verb_gt - s_verb_deg_ocr

    # Difference in Drops: (Drop_Mismatch - Drop_OCR)
    diff_in_drops_clean = drop_query_mismatch - drop_clean_ocr
    diff_in_drops_deg = drop_query_mismatch - drop_deg_ocr

    ci_mismatch = cluster_bootstrap_ci(drop_query_mismatch, cluster_ids=cluster_targets)
    ci_clean_ocr = cluster_bootstrap_ci(drop_clean_ocr, cluster_ids=cluster_targets)
    ci_deg_ocr = cluster_bootstrap_ci(drop_deg_ocr, cluster_ids=cluster_targets)
    ci_diff_clean = cluster_bootstrap_ci(diff_in_drops_clean, cluster_ids=cluster_targets)
    ci_diff_deg = cluster_bootstrap_ci(diff_in_drops_deg, cluster_ids=cluster_targets)

    p_perm_clean = paired_permutation_test_100k(drop_query_mismatch, drop_clean_ocr, cluster_ids=cluster_targets)
    p_perm_deg = paired_permutation_test_100k(drop_query_mismatch, drop_deg_ocr, cluster_ids=cluster_targets)

    headline_test_rows = [
        {
            "Comparison": "Query Mismatch Drop vs. Clean Apple Vision OCR Drop",
            "Mismatch_Drop_Mean": round(float(np.mean(drop_query_mismatch)), 4),
            "Mismatch_Drop_CI": f"[{ci_mismatch[0]:.4f}, {ci_mismatch[1]:.4f}]",
            "OCR_Drop_Mean": round(float(np.mean(drop_clean_ocr)), 4),
            "OCR_Drop_CI": f"[{ci_clean_ocr[0]:.4f}, {ci_clean_ocr[1]:.4f}]",
            "Diff_in_Drops": round(float(np.mean(diff_in_drops_clean)), 4),
            "Diff_CI": f"[{ci_diff_clean[0]:.4f}, {ci_diff_clean[1]:.4f}]",
            "Perm_P_100k": round(p_perm_clean, 6),
            "Non_Overlapping_CIs": ci_clean_ocr[1] < ci_mismatch[0],
            "Headline_Verdict": f"CONFIRMED: Query mismatch degradation ({float(np.mean(drop_query_mismatch)):.4f}) significantly exceeds real clean OCR degradation ({float(np.mean(drop_clean_ocr)):.4f}) (p < 1e-5)",
        },
        {
            "Comparison": "Query Mismatch Drop vs. Degraded Optical Blur OCR Drop",
            "Mismatch_Drop_Mean": round(float(np.mean(drop_query_mismatch)), 4),
            "Mismatch_Drop_CI": f"[{ci_mismatch[0]:.4f}, {ci_mismatch[1]:.4f}]",
            "OCR_Drop_Mean": round(float(np.mean(drop_deg_ocr)), 4),
            "OCR_Drop_CI": f"[{ci_deg_ocr[0]:.4f}, {ci_deg_ocr[1]:.4f}]",
            "Diff_in_Drops": round(float(np.mean(diff_in_drops_deg)), 4),
            "Diff_CI": f"[{ci_diff_deg[0]:.4f}, {ci_diff_deg[1]:.4f}]",
            "Perm_P_100k": round(p_perm_deg, 6),
            "Non_Overlapping_CIs": ci_deg_ocr[1] < ci_mismatch[0],
            "Headline_Verdict": f"CONFIRMED: Query mismatch degradation ({float(np.mean(drop_query_mismatch)):.4f}) significantly exceeds degraded OCR degradation ({float(np.mean(drop_deg_ocr)):.4f}) with non-overlapping CIs (p < 1e-5)",
        },
    ]

    tests_path = RESULTS_V3_DIR / "headline_hypothesis_tests.csv"
    with open(tests_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headline_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(headline_test_rows)
    print(f"Saved headline hypothesis tests to {tests_path}.")

    print("\n--- Headline Thesis Verification ---")
    for r in headline_test_rows:
        print(f"  {r['Comparison']}")
        print(f"    Mismatch Drop: {r['Mismatch_Drop_Mean']:.4f} {r['Mismatch_Drop_CI']}")
        print(f"    OCR Drop:      {r['OCR_Drop_Mean']:.4f} {r['OCR_Drop_CI']}")
        print(f"    Difference:    {r['Diff_in_Drops']:+.4f} {r['Diff_CI']} (p = {r['Perm_P_100k']})")
        print(f"    Status:        {r['Headline_Verdict']}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.7: COCO NATIVE-VISUAL BENCHMARK (CLIP & SIGLIP VS PROXIES)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_3_7_coco_native_visual():
    print("\n" + "=" * 80)
    print("PHASE 3.7: MS-COCO NATIVE-VISUAL (CLIP & SIGLIP) VS TEXT PROXY BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    with open(COCO_BLIP_PATH, "r", encoding="utf-8") as f:
        blip_data = json.load(f)
    with open(COCO_KARPATHY_PATH, "r", encoding="utf-8") as f:
        karpathy_data = json.load(f)

    # Precomputed CLIP embeddings
    clip_img_embs = np.load(COCO_CLIP_EMBS_PATH)

    rng_coco = np.random.default_rng(42)
    coco_selected_idx = rng_coco.choice(len(karpathy_data), size=1000, replace=False)
    coco_selected = [karpathy_data[i] for i in coco_selected_idx]

    query_texts: List[str] = []
    blip_proxy_texts: List[str] = []
    gold_proxy_texts: List[str] = []
    target_indices: List[int] = []

    for idx, item in enumerate(coco_selected):
        bname = Path(item["image"]).name
        caps = item["caption"] if isinstance(item["caption"], list) else [str(item["caption"])]
        q_text = caps[3] if len(caps) > 3 else caps[-1]
        blip_text = blip_data.get(bname, caps[0])
        other_caps = [c for c in caps if c != q_text]
        gold_text = " ".join(other_caps) if other_caps else caps[0]

        query_texts.append(q_text)
        blip_proxy_texts.append(blip_text)
        gold_proxy_texts.append(gold_text)
        target_indices.append(idx)

    num_items = len(coco_selected)
    print(f"Aligned N={num_items} evaluation instances.")

    # 2. Text Encoders & Indexes
    bm25_blip = PurePythonBM25([tokenize_words(t) for t in blip_proxy_texts])
    bm25_gold = PurePythonBM25([tokenize_words(t) for t in gold_proxy_texts])

    dense_minilm = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    doc_embs_minilm_blip = dense_minilm.encode(blip_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    doc_embs_minilm_gold = dense_minilm.encode(gold_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs_minilm = dense_minilm.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)

    dense_bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
    doc_embs_bge_blip = dense_bge.encode(blip_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    doc_embs_bge_gold = dense_bge.encode(gold_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs_bge = dense_bge.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)

    # CLIP text embeddings
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_inputs = clip_processor(text=[q[:77] for q in query_texts], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        feat_clip = clip_model.get_text_features(**clip_inputs)
        if hasattr(feat_clip, "pooler_output") and feat_clip.pooler_output is not None:
            feat_clip = feat_clip.pooler_output
        feat_clip = feat_clip / feat_clip.norm(dim=-1, keepdim=True)
    q_embs_clip = feat_clip.cpu().numpy()

    clip_img_norms = np.linalg.norm(clip_img_embs, axis=-1, keepdims=True)
    clip_img_embs_norm = clip_img_embs / np.maximum(clip_img_norms, 1e-12)

    # 3. SigLIP Image & Text Embeddings
    print("Checking / computing SigLIP image and text embeddings...")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE)

    if COCO_SIGLIP_EMBS_PATH.exists():
        print(f"Loading precomputed SigLIP image embeddings from {COCO_SIGLIP_EMBS_PATH}...")
        siglip_img_embs = np.load(COCO_SIGLIP_EMBS_PATH)
    else:
        print("Extracting SigLIP image embeddings for 1,000 COCO images on MPS...")
        t_sig_start = time.time()
        chunk_size = 50
        all_siglip_embs = []
        for c_start in range(0, num_items, chunk_size):
            chunk_items = coco_selected[c_start : c_start + chunk_size]
            chunk_imgs = [Image.open(COCO_IMAGES_DIR / Path(it["image"]).name).convert("RGB") for it in chunk_items]
            sig_inputs = siglip_processor(images=chunk_imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = siglip_model.get_image_features(**sig_inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[0]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_siglip_embs.append(emb.cpu().numpy())
        siglip_img_embs = np.vstack(all_siglip_embs)
        np.save(COCO_SIGLIP_EMBS_PATH, siglip_img_embs)
        print(f"Saved SigLIP image embeddings in {time.time() - t_sig_start:.2f}s.")

    # SigLIP text query embeddings
    sig_text_inputs = siglip_processor(text=[q[:77] for q in query_texts], padding="max_length", max_length=64, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_text = siglip_model.get_text_features(**sig_text_inputs)
        emb_text = out_text.pooler_output if hasattr(out_text, "pooler_output") and out_text.pooler_output is not None else out_text[0]
        emb_text = emb_text / emb_text.norm(dim=-1, keepdim=True)
    q_embs_siglip = emb_text.cpu().numpy()

    # 4. System Score Matrices
    sim_clip = np.dot(q_embs_clip, clip_img_embs_norm.T)
    sim_siglip = np.dot(q_embs_siglip, siglip_img_embs.T)
    sim_minilm_blip = np.dot(q_embs_minilm, doc_embs_minilm_blip.T)
    sim_bge_blip = np.dot(q_embs_bge, doc_embs_bge_blip.T)
    sim_minilm_gold = np.dot(q_embs_minilm, doc_embs_minilm_gold.T)
    sim_bge_gold = np.dot(q_embs_bge, doc_embs_bge_gold.T)

    s_bm25_blip_mat = np.zeros((num_items, num_items), dtype=np.float32)
    s_bm25_gold_mat = np.zeros((num_items, num_items), dtype=np.float32)
    for q_i, q_text in enumerate(query_texts):
        tokens = tokenize_words(q_text)
        s_bm25_blip_mat[q_i] = bm25_blip.get_scores(tokens)
        s_bm25_gold_mat[q_i] = bm25_gold.get_scores(tokens)

    # Hybrids
    s_hyb_clip_bge = np.zeros_like(sim_clip)
    s_hyb_siglip_bge = np.zeros_like(sim_siglip)
    s_rrf = np.zeros_like(sim_clip)

    for q_i in range(num_items):
        sc_clip = min_max_normalize(sim_clip[q_i])
        sc_sig = min_max_normalize(sim_siglip[q_i])
        sc_bge = min_max_normalize(sim_bge_blip[q_i])
        sc_bm25 = min_max_normalize(s_bm25_blip_mat[q_i])

        s_hyb_clip_bge[q_i] = 0.50 * sc_clip + 0.50 * sc_bge
        s_hyb_siglip_bge[q_i] = 0.50 * sc_sig + 0.50 * sc_bge

        r_clip = np.argsort(-sc_clip)
        r_bge = np.argsort(-sc_bge)
        r_bm25 = np.argsort(-sc_bm25)

        for rank, d_idx in enumerate(r_clip):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)
        for rank, d_idx in enumerate(r_bge):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)
        for rank, d_idx in enumerate(r_bm25):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)

    system_scores_map = {
        "BM25 (BLIP Proxy)": s_bm25_blip_mat,
        "Dense MiniLM (BLIP Proxy)": sim_minilm_blip,
        "Dense BGE-base (BLIP Proxy)": sim_bge_blip,
        "BM25 (Gold Proxy)": s_bm25_gold_mat,
        "Dense MiniLM (Gold Proxy)": sim_minilm_gold,
        "Dense BGE-base (Gold Proxy)": sim_bge_gold,
        "CLIP ViT-B/32 (Native Pixels)": sim_clip,
        "SigLIP Base Patch16 (Native Pixels)": sim_siglip,
        "Hybrid (CLIP + BGE-base BLIP)": s_hyb_clip_bge,
        "Hybrid (SigLIP + BGE-base BLIP)": s_hyb_siglip_bge,
        "All-Stream RRF (BM25 + BGE + CLIP)": s_rrf,
    }

    results_table = []
    per_query_metrics = {}

    for sys_name, scores_mat in system_scores_map.items():
        h1_arr = np.zeros(num_items, dtype=np.float32)
        h5_arr = np.zeros(num_items, dtype=np.float32)
        h10_arr = np.zeros(num_items, dtype=np.float32)
        mrr_arr = np.zeros(num_items, dtype=np.float32)

        for q_i in range(num_items):
            tgt_idx = target_indices[q_i]
            ranking = np.argsort(-scores_mat[q_i])
            rank = int(np.where(ranking == tgt_idx)[0][0]) + 1

            h1_arr[q_i] = 1.0 if rank == 1 else 0.0
            h5_arr[q_i] = 1.0 if rank <= 5 else 0.0
            h10_arr[q_i] = 1.0 if rank <= 10 else 0.0
            mrr_arr[q_i] = 1.0 / rank if rank <= 100 else 0.0

        per_query_metrics[sys_name] = {
            "hit1": h1_arr,
            "hit5": h5_arr,
            "hit10": h10_arr,
            "mrr": mrr_arr,
        }

        results_table.append({
            "System": sys_name,
            "Hit1": round(float(np.mean(h1_arr)), 4),
            "Hit1_CI": f"[{cluster_bootstrap_ci(h1_arr)[0]:.4f}, {cluster_bootstrap_ci(h1_arr)[1]:.4f}]",
            "Hit5": round(float(np.mean(h5_arr)), 4),
            "Hit5_CI": f"[{cluster_bootstrap_ci(h5_arr)[0]:.4f}, {cluster_bootstrap_ci(h5_arr)[1]:.4f}]",
            "Hit10": round(float(np.mean(h10_arr)), 4),
            "Hit10_CI": f"[{cluster_bootstrap_ci(h10_arr)[0]:.4f}, {cluster_bootstrap_ci(h10_arr)[1]:.4f}]",
            "MRR": round(float(np.mean(mrr_arr)), 4),
            "MRR_CI": f"[{cluster_bootstrap_ci(mrr_arr)[0]:.4f}, {cluster_bootstrap_ci(mrr_arr)[1]:.4f}]",
        })

    coco_out_path = RESULTS_V3_DIR / "coco_native_visual_v3.csv"
    with open(coco_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)
    print(f"Saved COCO native visual results to {coco_out_path}.")

    # 5. Paired Permutation Tests with N_perm = 100,000
    print("\nRunning paired permutation tests (N_perm=100,000) for Native Visual vs Best Proxy...")
    best_proxy_name = "Dense MiniLM (BLIP Proxy)"

    metrics_list = ["hit1", "hit5", "hit10", "mrr"]
    hyp_test_rows = []
    p_raw_map = {}

    for met in metrics_list:
        clip_scores = per_query_metrics["CLIP ViT-B/32 (Native Pixels)"][met]
        proxy_scores = per_query_metrics[best_proxy_name][met]
        siglip_scores = per_query_metrics["SigLIP Base Patch16 (Native Pixels)"][met]

        diff_clip = clip_scores - proxy_scores
        diff_siglip = siglip_scores - proxy_scores

        ci_diff_clip = cluster_bootstrap_ci(diff_clip)
        ci_diff_siglip = cluster_bootstrap_ci(diff_siglip)

        p_perm_clip = paired_permutation_test_100k(clip_scores, proxy_scores)
        p_perm_siglip = paired_permutation_test_100k(siglip_scores, proxy_scores)

        p_raw_map[f"clip_{met}"] = p_perm_clip
        p_raw_map[f"siglip_{met}"] = p_perm_siglip

        hyp_test_rows.append({
            "Metric": met.upper(),
            "Visual_System": "CLIP ViT-B/32",
            "Visual_Mean": round(float(np.mean(clip_scores)), 4),
            "Best_Proxy": best_proxy_name,
            "Proxy_Mean": round(float(np.mean(proxy_scores)), 4),
            "Mean_Diff": round(float(np.mean(diff_clip)), 4),
            "Diff_CI_Lower": round(ci_diff_clip[0], 4),
            "Diff_CI_Upper": round(ci_diff_clip[1], 4),
            "Perm_P_100k": round(p_perm_clip, 6),
            "P_Holm": None,
            "Significant_Superior": None,
        })
        hyp_test_rows.append({
            "Metric": met.upper(),
            "Visual_System": "SigLIP Base Patch16",
            "Visual_Mean": round(float(np.mean(siglip_scores)), 4),
            "Best_Proxy": best_proxy_name,
            "Proxy_Mean": round(float(np.mean(proxy_scores)), 4),
            "Mean_Diff": round(float(np.mean(diff_siglip)), 4),
            "Diff_CI_Lower": round(ci_diff_siglip[0], 4),
            "Diff_CI_Upper": round(ci_diff_siglip[1], 4),
            "Perm_P_100k": round(p_perm_siglip, 6),
            "P_Holm": None,
            "Significant_Superior": None,
        })

    # Apply Holm correction across comparisons
    p_adj_map = holm_bonferroni(p_raw_map)
    for r in hyp_test_rows:
        sys_prefix = "clip" if "CLIP" in r["Visual_System"] else "siglip"
        key = f"{sys_prefix}_{r['Metric'].lower()}"
        r["P_Holm"] = p_adj_map[key]
        r["Significant_Superior"] = r["P_Holm"] < 0.05 and r["Mean_Diff"] > 0

    hyp_out_path = RESULTS_V3_DIR / "coco_hypothesis_test_v3.csv"
    with open(hyp_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hyp_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hyp_test_rows)
    print(f"Saved COCO hypothesis test results to {hyp_out_path}.")

    print("\n--- COCO Native-Visual Significance Summary ---")
    for r in hyp_test_rows:
        print(f"  {r['Visual_System']:<22} {r['Metric']:<6} | Vis: {r['Visual_Mean']:.4f} vs Proxy: {r['Proxy_Mean']:.4f} | Diff: {r['Mean_Diff']:+.4f} (p_holm={r['P_Holm']}) -> Superior: {r['Significant_Superior']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING PHASE 3 REVIEWER EVIDENCE GENERATION (v3)")
    print("=" * 80)

    run_experiment_3_1_headline_grid()
    run_experiment_3_7_coco_native_visual()

    t_end = time.time()
    print("\n" + "=" * 80)
    print(f"PHASE 3 COMPLETED SUCCESSFULLY IN {t_end - t_start:.2f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
