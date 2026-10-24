#!/usr/bin/env python3
"""
Phase 5: Conformal Cascade Calibration & Cost-Quality Pareto Frontier (v2)

This script implements rigorous, pre-registered conformal risk control for the
margin-triggered cascade ranker on a pooled benchmark of N=2,754 queries across:
  - 306 probe clusters
  - 3 query variants: verbatim, light paraphrase, heavy paraphrase
  - 3 representation conditions: clean OCR, synthetic 20% noise, real Apple Vision OCR

Protocol (Pre-Registered):
  - 200 random 50/50 cluster-stratified splits (seeds 1000..1199).
  - For each split:
      1. Calibrate threshold tau on calibration half targeting non-escalated miss rate epsilon <= 0.05.
      2. Evaluate empirical miss rate and escalation rate on held-out test half.
  - Report:
      - Fraction of splits satisfying test miss rate <= 0.05 (empirical coverage).
      - Mean, 5th, 50th, 95th percentiles of test miss rate.
      - Mean escalation rate and distribution.
      - Cost-quality Pareto curve: Conformal vs. Fixed-threshold vs. Always-escalate vs. Never-escalate
        across (Mean Latency / Cost) x (MRR / Hit@10).
"""

import collections
import csv
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Set site-packages for matplotlib / transformers / torch if needed
if "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages" not in sys.path:
    sys.path.append("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages")

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DIR = DATA_DIR / "v2"
RESULTS_V2_DIR = REPO_ROOT / "results" / "v2"
LATEX_DIR = REPO_ROOT / "latex_fragments"

RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PATH = V2_DIR / "public_benchmark_v2.json"
FROZEN_PROBES_PATH = V2_DIR / "frozen_probes_with_paraphrases.json"
DEV_PARAMS_CACHE = V2_DIR / "dev_tuned_params_cache.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = list(range(1000, 1200))  # 200 seeds: 1000..1199


# ─── LEXICAL & NORMALIZATION UTILITIES ─────────────────────────────────────────
def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


class PurePythonBM25:
    """Inverted Index Okapi BM25 implementation."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
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

    def get_scores(self, query: List[str]) -> np.ndarray:
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


def wilson_score_upper_bound(p_hat: float, n: int, confidence: float = 0.95) -> float:
    """Computes Wilson score upper bound for binomial proportion."""
    if n == 0:
        return 1.0
    z = 1.959963984540054  # for 95%
    denom = 1.0 + (z**2) / n
    center = p_hat + (z**2) / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)
    return min(1.0, (center + spread) / denom)


# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
def run_phase5_conformal_cascade():
    t_start = time.time()
    print("=" * 80)
    print("STARTING PHASE 5: CONFORMAL CASCADE CALIBRATION & PARETO FRONTIER")
    print("=" * 80)

    # 1. Load Datasets
    print(f"Loading benchmark from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    print(f"Loading frozen probes from {FROZEN_PROBES_PATH}...")
    with open(FROZEN_PROBES_PATH, "r", encoding="utf-8") as f:
        probe_data = json.load(f)

    records = bench_data["records"]
    test_records = [r for r in records if r["split"] == "test"]
    test_probes = probe_data["test_probes"]

    test_id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}
    num_test_records = len(test_records)
    print(f"Loaded: Test Index Records = {num_test_records}, Test Probes = {len(test_probes)}")

    # Load tuned parameters cache
    raw_cache = {}
    if DEV_PARAMS_CACHE.exists():
        with open(DEV_PARAMS_CACHE, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
    print(f"Loaded {len(raw_cache)} parameter sets from Dev cache.")

    # 2. Models
    print(f"Loading dense bi-encoder ('all-MiniLM-L6-v2') on {DEVICE}...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print(f"Loading cross-encoder ('BAAI/bge-reranker-base') on {DEVICE}...")
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

    # 3. Build Representation Corpora
    # Conditions: 'clean', 'syn_20', 'real_ocr'
    conditions = ["clean", "syn_20", "real_ocr"]
    rng_syn = np.random.default_rng(42)

    condition_corpora: Dict[str, List[str]] = {}
    condition_bm25: Dict[str, PurePythonBM25] = {}
    condition_dense_mat: Dict[str, np.ndarray] = {}

    for cond in conditions:
        print(f"Indexing corpus under condition: '{cond}'...")
        doc_texts: List[str] = []
        for r in test_records:
            if cond == "clean":
                t = r.get("clean_text") or r.get("real_ocr_text", "")
            elif cond == "syn_20":
                base_t = r.get("clean_text") or r.get("real_ocr_text", "")
                t = corrupt_text_random(base_t, noise_rate=0.20, rng=rng_syn)
            elif cond == "real_ocr":
                t = r.get("real_ocr_text") or r.get("clean_text", "")
            else:
                t = r.get("clean_text", "")
            doc_texts.append(t)
        condition_corpora[cond] = doc_texts
        tokens = [tokenize_words(t) for t in doc_texts]
        condition_bm25[cond] = PurePythonBM25(tokens)
        condition_dense_mat[cond] = dense_model.encode(
            doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    # 4. Construct Pooled Query Evaluations (N = 306 probes x 3 variants x 3 conditions = 2,754 queries)
    query_variants = ["verbatim", "light_paraphrase", "heavy_paraphrase"]
    pooled_evals = []

    print("\nPrecomputing Stage 1 retrieval and Stage 2 reranking for all pooled queries...")
    t_stage1_total = 0.0
    ce_pairs_to_score = []
    eval_metadata = []

    for cond in conditions:
        bm25_idx = condition_bm25[cond]
        dense_mat = condition_dense_mat[cond]
        doc_texts = condition_corpora[cond]

        for q_var in query_variants:
            var_key = "verbatim" if "verbatim" in q_var else ("light" if "light" in q_var else "heavy")
            cache_key = f"{var_key}:::{cond}"
            alpha_star = raw_cache.get(cache_key, {}).get("alpha", 0.50)

            queries = [p[q_var] for p in test_probes]
            targets = [p["target_id"] for p in test_probes]
            probe_ids = [p["probe_id"] for p in test_probes]

            # Batch encode queries
            t_s1_start = time.perf_counter()
            q_embs = dense_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            t_s1_embed = (time.perf_counter() - t_s1_start) / len(queries)

            for i, q_text in enumerate(queries):
                target_id = targets[i]
                if target_id not in test_id_to_idx:
                    continue
                tgt_idx = test_id_to_idx[target_id]

                t_query_start = time.perf_counter()
                s_d = min_max_normalize(np.dot(dense_mat, q_embs[i]))
                s_b = min_max_normalize(bm25_idx.get_scores(tokenize_words(q_text)))
                s_hybrid = alpha_star * s_d + (1.0 - alpha_star) * s_b
                t_query_s1 = (time.perf_counter() - t_query_start) + t_s1_embed
                t_stage1_total += t_query_s1

                sorted_indices = np.argsort(-s_hybrid)
                s1 = float(s_hybrid[sorted_indices[0]])
                s2 = float(s_hybrid[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
                margin = float(s1 - s2)

                top1_idx = int(sorted_indices[0])
                s1_hit1 = 1 if top1_idx == tgt_idx else 0
                s1_rank = int(np.where(sorted_indices == tgt_idx)[0][0]) + 1
                s1_mrr = 1.0 / s1_rank if s1_rank <= 100 else 0.0
                s1_hit10 = 1 if s1_rank <= 10 else 0

                top5_indices = sorted_indices[:5].tolist()
                pair_start_idx = len(ce_pairs_to_score)
                for cand_idx in top5_indices:
                    ce_pairs_to_score.append((q_text, doc_texts[cand_idx][:300]))

                eval_metadata.append({
                    "cluster_id": target_id,
                    "probe_id": probe_ids[i],
                    "condition": cond,
                    "query_variant": q_var,
                    "tgt_idx": tgt_idx,
                    "margin": margin,
                    "s1_hit1": s1_hit1,
                    "s1_rank": s1_rank,
                    "s1_mrr": s1_mrr,
                    "s1_hit10": s1_hit10,
                    "top5_indices": top5_indices,
                    "pair_start_idx": pair_start_idx,
                })

    num_pooled = len(eval_metadata)
    mean_s1_latency_ms = (t_stage1_total / max(1, num_pooled)) * 1000.0
    print(f"Stage 1 precomputed for N={num_pooled} queries (Mean Latency: {mean_s1_latency_ms:.2f} ms/query).")

    # Batch run Stage 2 Cross-Encoder reranking in safe chunks of 250 pairs
    print(f"Running Cross-Encoder over {len(ce_pairs_to_score)} pairs in chunks of 250...")
    t_ce_start = time.perf_counter()
    chunk_size = 250
    all_ce_scores_list = []
    for c_start in range(0, len(ce_pairs_to_score), chunk_size):
        chunk = ce_pairs_to_score[c_start : c_start + chunk_size]
        c_scores = cross_encoder.predict(chunk, batch_size=32, show_progress_bar=False)
        all_ce_scores_list.extend(c_scores.tolist())
    all_ce_scores = np.array(all_ce_scores_list, dtype=np.float32)
    t_ce_total = time.perf_counter() - t_ce_start
    mean_s2_latency_ms = (t_ce_total / max(1, num_pooled)) * 1000.0
    print(f"Stage 2 cross-encoder finished in {t_ce_total:.2f}s (Mean Rerank Latency: {mean_s2_latency_ms:.2f} ms/query).")

    # Combine Stage 1 and Stage 2 precomputed stats into fast arrays
    margins = np.zeros(num_pooled, dtype=np.float32)
    s1_hit1s = np.zeros(num_pooled, dtype=np.int32)
    s1_mrrs = np.zeros(num_pooled, dtype=np.float32)
    s1_hit10s = np.zeros(num_pooled, dtype=np.float32)

    s2_hit1s = np.zeros(num_pooled, dtype=np.int32)
    s2_mrrs = np.zeros(num_pooled, dtype=np.float32)
    s2_hit10s = np.zeros(num_pooled, dtype=np.float32)

    cluster_ids = []

    for idx, meta in enumerate(eval_metadata):
        margins[idx] = meta["margin"]
        s1_hit1s[idx] = meta["s1_hit1"]
        s1_mrrs[idx] = meta["s1_mrr"]
        s1_hit10s[idx] = meta["s1_hit10"]
        cluster_ids.append(meta["cluster_id"])

        p_start = meta["pair_start_idx"]
        ce_scores = all_ce_scores[p_start : p_start + len(meta["top5_indices"])]
        reordered_top5 = [meta["top5_indices"][k] for k in np.argsort(-ce_scores)]

        tgt_idx = meta["tgt_idx"]
        s2_hit1 = 1 if reordered_top5[0] == tgt_idx else 0
        if tgt_idx in reordered_top5:
            s2_rank = reordered_top5.index(tgt_idx) + 1
        else:
            s2_rank = meta["s1_rank"]

        s2_mrr = 1.0 / s2_rank if s2_rank <= 100 else 0.0
        s2_hit10 = 1 if s2_rank <= 10 else 0

        s2_hit1s[idx] = s2_hit1
        s2_mrrs[idx] = s2_mrr
        s2_hit10s[idx] = s2_hit10

    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    cluster_to_indices: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, c in enumerate(cluster_ids):
        cluster_to_indices[c].append(idx)

    print(f"Grouped {num_pooled} queries into {num_clusters} probe clusters.")

    # ─── 5. 200 RANDOM CLUSTER-STRATIFIED 50/50 SPLITS (SEEDS 1000..1199) ─────────
    print(f"\nExecuting 200 random cluster-stratified splits (seeds 1000..1199)...")
    tau_sweep = np.linspace(0.0, 0.40, 201)

    split_results = []
    target_epsilon = 0.05

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled_clusters = rng.permutation(unique_clusters)
        split_point = len(shuffled_clusters) // 2

        cal_clusters = set(shuffled_clusters[:split_point])
        test_clusters = set(shuffled_clusters[split_point:])

        cal_indices = [idx for c in cal_clusters for idx in cluster_to_indices[c]]
        test_indices = [idx for c in test_clusters for idx in cluster_to_indices[c]]

        cal_m = margins[cal_indices]
        cal_s1_h1 = s1_hit1s[cal_indices]

        test_m = margins[test_indices]
        test_s1_h1 = s1_hit1s[test_indices]
        test_s1_mrr = s1_mrrs[test_indices]
        test_s1_h10 = s1_hit10s[test_indices]
        test_s2_mrr = s2_mrrs[test_indices]
        test_s2_h10 = s2_hit10s[test_indices]

        # 5.1 Calibrate on Calibration Half
        # Sweep tau to find smallest tau bounding non-escalated miss rate <= 0.05
        best_tau_emp = 0.40
        best_tau_ucb = 0.40
        found_emp = False
        found_ucb = False

        for tau in tau_sweep:
            non_esc_mask = cal_m >= tau
            n_non_esc = int(np.sum(non_esc_mask))
            if n_non_esc == 0:
                risk_emp = 0.0
                risk_ucb = 0.0
            else:
                misses = int(np.sum(1 - cal_s1_h1[non_esc_mask]))
                risk_emp = misses / n_non_esc
                risk_ucb = wilson_score_upper_bound(risk_emp, n_non_esc, confidence=0.95)

            if not found_emp and risk_emp <= target_epsilon:
                best_tau_emp = float(tau)
                found_emp = True

            if not found_ucb and risk_ucb <= target_epsilon:
                best_tau_ucb = float(tau)
                found_ucb = True

        # 5.2 Evaluate on Held-Out Test Half for Empirical Tau
        def eval_test_at_tau(tau: float) -> Dict[str, float]:
            esc_mask = test_m < tau
            non_esc_mask = ~esc_mask
            n_test = len(test_indices)
            n_non_esc = int(np.sum(non_esc_mask))
            n_esc = int(np.sum(esc_mask))

            test_misses = int(np.sum(1 - test_s1_h1[non_esc_mask]))
            test_miss_rate = (test_misses / n_non_esc) if n_non_esc > 0 else 0.0
            escalation_rate = n_esc / n_test

            # Retrieval effectiveness: escalated queries get S2, non-escalated get S1
            final_mrr = float(np.mean(np.where(esc_mask, test_s2_mrr, test_s1_mrr)))
            final_h10 = float(np.mean(np.where(esc_mask, test_s2_h10, test_s1_h10)))
            final_latency = mean_s1_latency_ms + escalation_rate * mean_s2_latency_ms

            return {
                "tau": tau,
                "miss_rate": test_miss_rate,
                "escalation_rate": escalation_rate,
                "mrr": final_mrr,
                "hit10": final_h10,
                "latency_ms": final_latency,
                "satisfied": 1 if test_miss_rate <= target_epsilon else 0,
            }

        res_emp = eval_test_at_tau(best_tau_emp)
        res_ucb = eval_test_at_tau(best_tau_ucb)

        split_results.append({
            "seed": seed,
            "tau_emp": best_tau_emp,
            "test_miss_emp": res_emp["miss_rate"],
            "test_esc_emp": res_emp["escalation_rate"],
            "test_mrr_emp": res_emp["mrr"],
            "test_h10_emp": res_emp["hit10"],
            "test_lat_emp": res_emp["latency_ms"],
            "satisfied_emp": res_emp["satisfied"],
            "tau_ucb": best_tau_ucb,
            "test_miss_ucb": res_ucb["miss_rate"],
            "test_esc_ucb": res_ucb["escalation_rate"],
            "test_mrr_ucb": res_ucb["mrr"],
            "test_h10_ucb": res_ucb["hit10"],
            "test_lat_ucb": res_ucb["latency_ms"],
            "satisfied_ucb": res_ucb["satisfied"],
        })

    # ─── 6. FIXED THRESHOLDS PARETO SWEEP ──────────────────────────────────────────
    fixed_taus = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    pareto_data = []

    # Baselines: Never Escalate (tau = 0.0), Always Escalate (tau -> inf)
    all_n = len(margins)
    s1_only_mrr = float(np.mean(s1_mrrs))
    s1_only_h10 = float(np.mean(s1_hit10s))
    s1_only_miss = float(np.mean(1 - s1_hit1s))
    s1_only_lat = mean_s1_latency_ms

    always_esc_mrr = float(np.mean(s2_mrrs))
    always_esc_h10 = float(np.mean(s2_hit10s))
    always_esc_miss = 0.0  # 0 non-escalated queries
    always_esc_lat = mean_s1_latency_ms + mean_s2_latency_ms

    pareto_data.append({
        "policy": "Never Escalate (Stage-1 Only)",
        "tau": 0.0,
        "escalation_rate": 0.0,
        "non_esc_miss_rate": s1_only_miss,
        "mrr": s1_only_mrr,
        "hit10": s1_only_h10,
        "mean_latency_ms": s1_only_lat,
    })

    for f_tau in fixed_taus:
        esc_m = margins < f_tau
        non_esc_m = ~esc_m
        n_non_esc = int(np.sum(non_esc_m))
        miss_r = float(np.sum(1 - s1_hit1s[non_esc_m]) / n_non_esc) if n_non_esc > 0 else 0.0
        esc_r = float(np.mean(esc_m))
        cur_mrr = float(np.mean(np.where(esc_m, s2_mrrs, s1_mrrs)))
        cur_h10 = float(np.mean(np.where(esc_m, s2_hit10s, s1_hit10s)))
        cur_lat = mean_s1_latency_ms + esc_r * mean_s2_latency_ms

        pareto_data.append({
            "policy": f"Fixed tau = {f_tau:.2f}",
            "tau": f_tau,
            "escalation_rate": esc_r,
            "non_esc_miss_rate": miss_r,
            "mrr": cur_mrr,
            "hit10": cur_h10,
            "mean_latency_ms": cur_lat,
        })

    pareto_data.append({
        "policy": "Always Escalate (All Queries)",
        "tau": 99.0,
        "escalation_rate": 1.0,
        "non_esc_miss_rate": always_esc_miss,
        "mrr": always_esc_mrr,
        "hit10": always_esc_h10,
        "mean_latency_ms": always_esc_lat,
    })

    # Add Conformal Cascade average
    mean_tau_conf = float(np.mean([r["tau_emp"] for r in split_results]))
    mean_esc_conf = float(np.mean([r["test_esc_emp"] for r in split_results]))
    mean_miss_conf = float(np.mean([r["test_miss_emp"] for r in split_results]))
    mean_mrr_conf = float(np.mean([r["test_mrr_emp"] for r in split_results]))
    mean_h10_conf = float(np.mean([r["test_h10_emp"] for r in split_results]))
    mean_lat_conf = float(np.mean([r["test_lat_emp"] for r in split_results]))

    pareto_data.append({
        "policy": "Conformal Cascade (Empirical, epsilon=0.05)",
        "tau": round(mean_tau_conf, 3),
        "escalation_rate": mean_esc_conf,
        "non_esc_miss_rate": mean_miss_conf,
        "mrr": mean_mrr_conf,
        "hit10": mean_h10_conf,
        "mean_latency_ms": mean_lat_conf,
    })

    # Add Conformal UCB average
    mean_tau_ucb = float(np.mean([r["tau_ucb"] for r in split_results]))
    mean_esc_ucb = float(np.mean([r["test_esc_ucb"] for r in split_results]))
    mean_miss_ucb = float(np.mean([r["test_miss_ucb"] for r in split_results]))
    mean_mrr_ucb = float(np.mean([r["test_mrr_ucb"] for r in split_results]))
    mean_h10_ucb = float(np.mean([r["test_h10_ucb"] for r in split_results]))
    mean_lat_ucb = float(np.mean([r["test_lat_ucb"] for r in split_results]))

    pareto_data.append({
        "policy": "Conformal Cascade (Conservative 95% UCB)",
        "tau": round(mean_tau_ucb, 3),
        "escalation_rate": mean_esc_ucb,
        "non_esc_miss_rate": mean_miss_ucb,
        "mrr": mean_mrr_ucb,
        "hit10": mean_h10_ucb,
        "mean_latency_ms": mean_lat_ucb,
    })

    # ─── 7. COMPUTE SUMMARY METRICS & DISTRIBUTIONS ────────────────────────────────
    test_misses_emp = np.array([r["test_miss_emp"] for r in split_results])
    test_escs_emp = np.array([r["test_esc_emp"] for r in split_results])
    sat_frac_emp = float(np.mean([r["satisfied_emp"] for r in split_results]))

    test_misses_ucb = np.array([r["test_miss_ucb"] for r in split_results])
    test_escs_ucb = np.array([r["test_esc_ucb"] for r in split_results])
    sat_frac_ucb = float(np.mean([r["satisfied_ucb"] for r in split_results]))

    summary_rows = [
        {
            "method": "Conformal Cascade (Empirical Calibration)",
            "target_epsilon": target_epsilon,
            "coverage_fraction": sat_frac_emp,
            "miss_rate_mean": float(np.mean(test_misses_emp)),
            "miss_rate_p05": float(np.percentile(test_misses_emp, 5)),
            "miss_rate_p50": float(np.percentile(test_misses_emp, 50)),
            "miss_rate_p95": float(np.percentile(test_misses_emp, 95)),
            "escalation_rate_mean": float(np.mean(test_escs_emp)),
            "escalation_rate_p05": float(np.percentile(test_escs_emp, 5)),
            "escalation_rate_p50": float(np.percentile(test_escs_emp, 50)),
            "escalation_rate_p95": float(np.percentile(test_escs_emp, 95)),
            "test_mrr_mean": mean_mrr_conf,
            "test_hit10_mean": mean_h10_conf,
            "latency_ms_mean": mean_lat_conf,
        },
        {
            "method": "Conformal Cascade (Conservative 95% UCB)",
            "target_epsilon": target_epsilon,
            "coverage_fraction": sat_frac_ucb,
            "miss_rate_mean": float(np.mean(test_misses_ucb)),
            "miss_rate_p05": float(np.percentile(test_misses_ucb, 5)),
            "miss_rate_p50": float(np.percentile(test_misses_ucb, 50)),
            "miss_rate_p95": float(np.percentile(test_misses_ucb, 95)),
            "escalation_rate_mean": float(np.mean(test_escs_ucb)),
            "escalation_rate_p05": float(np.percentile(test_escs_ucb, 5)),
            "escalation_rate_p50": float(np.percentile(test_escs_ucb, 50)),
            "escalation_rate_p95": float(np.percentile(test_escs_ucb, 95)),
            "test_mrr_mean": mean_mrr_ucb,
            "test_hit10_mean": mean_h10_ucb,
            "latency_ms_mean": mean_lat_ucb,
        },
    ]

    print("\n" + "=" * 80)
    print("PHASE 5 CONFORMAL CASCADE CALIBRATION RESULTS (200 SPLITS):")
    print("=" * 80)
    print(f"Empirical Calibration: Target epsilon = {target_epsilon:.2f}")
    print(f"  Coverage (splits with Test Miss Rate <= 0.05): {sat_frac_emp * 100:.1f}%")
    print(f"  Test Miss Rate Mean: {np.mean(test_misses_emp):.4f} [5th: {np.percentile(test_misses_emp, 5):.4f}, 50th: {np.percentile(test_misses_emp, 50):.4f}, 95th: {np.percentile(test_misses_emp, 95):.4f}]")
    print(f"  Test Escalation Rate Mean: {np.mean(test_escs_emp) * 100:.1f}% [5th: {np.percentile(test_escs_emp, 5) * 100:.1f}%, 95th: {np.percentile(test_escs_emp, 95) * 100:.1f}%]")
    print(f"  Test MRR Mean: {mean_mrr_conf:.4f}, Mean Latency: {mean_lat_conf:.2f} ms")
    print("-" * 80)
    print(f"Conservative 95% UCB Calibration:")
    print(f"  Coverage (splits with Test Miss Rate <= 0.05): {sat_frac_ucb * 100:.1f}%")
    print(f"  Test Miss Rate Mean: {np.mean(test_misses_ucb):.4f} [5th: {np.percentile(test_misses_ucb, 5):.4f}, 50th: {np.percentile(test_misses_ucb, 50):.4f}, 95th: {np.percentile(test_misses_ucb, 95):.4f}]")
    print(f"  Test Escalation Rate Mean: {np.mean(test_escs_ucb) * 100:.1f}%")

    # ─── 8. WRITE CSV ARTIFACTS ───────────────────────────────────────────────────
    out_splits_path = RESULTS_V2_DIR / "conformal_cascade_results.csv"
    with open(out_splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_results[0].keys()))
        writer.writeheader()
        writer.writerows(split_results)
    print(f"\nWrote split-by-split results to: {out_splits_path}")

    out_summary_path = RESULTS_V2_DIR / "conformal_splits_summary.csv"
    with open(out_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote calibration summary to: {out_summary_path}")

    out_pareto_path = RESULTS_V2_DIR / "conformal_pareto_curve.csv"
    with open(out_pareto_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pareto_data[0].keys()))
        writer.writeheader()
        writer.writerows(pareto_data)
    print(f"Wrote Pareto curve data to: {out_pareto_path}")

    # ─── 9. GENERATE PARETO FRONTIER FIGURE ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Latency vs MRR
    fixed_points = [p for p in pareto_data if p["policy"].startswith("Fixed")]
    f_lats = [p["mean_latency_ms"] for p in fixed_points]
    f_mrrs = [p["mrr"] for p in fixed_points]

    ax1.plot(f_lats, f_mrrs, "o--", color="#6c757d", alpha=0.7, label="Fixed-Threshold Sweep (tau)")
    for p in fixed_points:
        if p["tau"] in [0.04, 0.08, 0.15, 0.25]:
            ax1.annotate(f"tau={p['tau']:.2f}", (p["mean_latency_ms"], p["mrr"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Never Escalate
    ax1.scatter([s1_only_lat], [s1_only_mrr], color="#2b5c8f", s=120, zorder=5, marker="s", label=f"Never Escalate (Stage-1: {s1_only_mrr:.4f})")
    # Always Escalate
    ax1.scatter([always_esc_lat], [always_esc_mrr], color="#d9534f", s=140, zorder=5, marker="^", label=f"Always Escalate (Stage-2: {always_esc_mrr:.4f})")
    # Conformal Empirical
    ax1.scatter([mean_lat_conf], [mean_mrr_conf], color="#28a745", s=180, zorder=6, marker="*", label=f"Conformal (tau*={mean_tau_conf:.3f}, Esc={mean_esc_conf*100:.1f}%)")
    # Conformal UCB
    ax1.scatter([mean_lat_ucb], [mean_mrr_ucb], color="#fd7e14", s=140, zorder=6, marker="D", label=f"Conformal 95% UCB (Esc={mean_esc_ucb*100:.1f}%)")

    ax1.set_xlabel("Mean Latency per Query (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Retrieval MRR", fontsize=11, fontweight="bold")
    ax1.set_title("A. Cost-Quality Pareto Frontier (Latency vs. MRR)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel B: Test Non-Escalated Miss Rate Distribution across 200 Splits
    bins = np.linspace(0.0, 0.10, 30)
    ax2.hist(test_misses_emp, bins=bins, color="#28a745", alpha=0.65, edgecolor="black", label=f"Empirical Calib (Coverage={sat_frac_emp*100:.1f}%)")
    ax2.hist(test_misses_ucb, bins=bins, color="#fd7e14", alpha=0.5, edgecolor="black", label=f"95% UCB Calib (Coverage={sat_frac_ucb*100:.1f}%)")
    ax2.axvline(target_epsilon, color="red", linestyle="--", lw=2, label=f"Target Risk Bound (epsilon = {target_epsilon:.2f})")

    ax2.set_xlabel("Held-Out Test Miss Rate (Non-Escalated)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Split Count (out of 200)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Risk Control Verification across 200 Random Splits", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path = RESULTS_V2_DIR / "fig_conformal_pareto.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Pareto figure to: {fig_path}")

    # ─── 10. GENERATE LATEX TABLE FRAGMENT ────────────────────────────────────────
    latex_path = LATEX_DIR / "table_conformal_v2.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Conformal Cascade Calibration & Pareto Frontier (N=2,754 pooled queries, 200 random splits)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Conformal cascade risk control and latency-quality trade-offs evaluated across 200 cluster-stratified 50/50 splits (seeds 1000--1199) on $N=2{,}754$ pooled benchmark queries. Risk bound target $\\epsilon = 0.05$.}\n")
        f.write("\\label{tab:conformal_cascade_v2}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Cascade Policy & Calibrated $\\tau$ & Escalation Rate & Test Miss Rate & Coverage ($\\le 0.05$) & Test MRR & Latency (ms) \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Never Escalate (Stage-1 Only) & --- & 0.0\\% & {s1_only_miss:.4f} & 0.0\\% & {s1_only_mrr:.4f} & {s1_only_lat:.2f} \\\\\n")
        f.write(f"Fixed Threshold ($\\tau=0.04$) & 0.040 & {pareto_data[2]['escalation_rate']*100:.1f}\\% & {pareto_data[2]['non_esc_miss_rate']:.4f} & --- & {pareto_data[2]['mrr']:.4f} & {pareto_data[2]['mean_latency_ms']:.2f} \\\\\n")
        f.write(f"Fixed Threshold ($\\tau=0.08$) & 0.080 & {pareto_data[4]['escalation_rate']*100:.1f}\\% & {pareto_data[4]['non_esc_miss_rate']:.4f} & --- & {pareto_data[4]['mrr']:.4f} & {pareto_data[4]['mean_latency_ms']:.2f} \\\\\n")
        f.write(f"Fixed Threshold ($\\tau=0.15$) & 0.150 & {pareto_data[7]['escalation_rate']*100:.1f}\\% & {pareto_data[7]['non_esc_miss_rate']:.4f} & --- & {pareto_data[7]['mrr']:.4f} & {pareto_data[7]['mean_latency_ms']:.2f} \\\\\n")
        f.write(f"\\textbf{{Conformal (Empirical)}} & \\textbf{{{mean_tau_conf:.3f}}} & \\textbf{{{mean_esc_conf*100:.1f}\\%}} & \\textbf{{{mean_miss_conf:.4f}}} & \\textbf{{{sat_frac_emp*100:.1f}\\%}} & \\textbf{{{mean_mrr_conf:.4f}}} & \\textbf{{{mean_lat_conf:.2f}}} \\\\\n")
        f.write(f"\\textbf{{Conformal (95\\% UCB)}} & \\textbf{{{mean_tau_ucb:.3f}}} & \\textbf{{{mean_esc_ucb*100:.1f}\\%}} & \\textbf{{{mean_miss_ucb:.4f}}} & \\textbf{{{sat_frac_ucb*100:.1f}\\%}} & \\textbf{{{mean_mrr_ucb:.4f}}} & \\textbf{{{mean_lat_ucb:.2f}}} \\\\\n")
        f.write(f"Always Escalate (Stage-2 Full) & --- & 100.0\\% & 0.0000 & 100.0\\% & {always_esc_mrr:.4f} & {always_esc_lat:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX fragment to: {latex_path}")

    elapsed = time.time() - t_start
    print(f"\nPhase 5 completed successfully in {elapsed:.2f}s.")


if __name__ == "__main__":
    run_phase5_conformal_cascade()
