#!/usr/bin/env python3
"""
Phase 6: Query-Side Remedies Evaluation (CPU)
==============================================
Implements and evaluates:
(b) Rule-based keyword/entity extraction from questions (LLM-free).
(c) Document expansion from OCR key tokens (appending predicted vendor/date/total).
(d) Query + OCR-text Pseudo-Relevance Feedback (RM3 / PRF).
Evaluates MRR gain with paired cluster CIs, latency, and cost on Light and Heavy queries.
Decision rule: Remedy is effective ONLY if paired MRR gain CI strictly excludes zero (> 0).
Emits:
- results/v6/remedies_summary.csv
- results/v6/remedies_queries_v6.json
- latex_fragments_v6/table_remedies_v6.tex
"""

import collections
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DATA_DIR = DATA_DIR / "v2"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"
V6_RESULTS_DIR = REPO_ROOT / "results" / "v6"
LATEX_DIR = REPO_ROOT / "latex_fragments_v6"

def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]

def cluster_bootstrap_ci(scores: np.ndarray, cluster_ids: List[str], B: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_c = len(unique_clusters)
    cluster_to_idx = collections.defaultdict(list)
    for idx, c in enumerate(cluster_ids):
        cluster_to_idx[c].append(idx)

    boot_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        sampled_c = rng.choice(unique_clusters, size=n_c, replace=True)
        sampled_idx = [i for c in sampled_c for i in cluster_to_idx[c]]
        boot_means[b] = np.mean(scores[sampled_idx])

    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD BENCHMARK AND TEST PROBES
# ─────────────────────────────────────────────────────────────────────────────
print("=== Phase 6 Setup: Loading Benchmark and Probes ===")
with open(V2_DATA_DIR / "public_benchmark_v2.json", "r", encoding="utf-8") as f:
    bench_data = json.load(f)

records = bench_data["records"]
test_records = [r for r in records if r["split"] == "test"]
test_id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}
assert len(test_records) == 813

with open(V2_DATA_DIR / "frozen_probes_with_paraphrases.json", "r", encoding="utf-8") as f:
    probe_data = json.load(f)
test_probes = probe_data["test_probes"]
assert len(test_probes) == 306

with open(V3_RESULTS_DIR / "heavy_paraphrases_v3.json", "r", encoding="utf-8") as f:
    heavy_probes_list = json.load(f)
heavy_map = {p["probe_id"]: p.get("heavy_question", p.get("heavy_paraphrase", "")) for p in heavy_probes_list}

test_doc_clean_ocr = [r.get("real_ocr_text") or r.get("ground_truth_text", "") for r in test_records]
target_indices = [test_id_to_idx[p["target_id"]] for p in test_probes]
cluster_targets = [p["target_id"] for p in test_probes]

# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPLEMENT QUERY-SIDE REMEDIES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Phase 6.1: Implementing Query Remedies ===")

QUESTION_STOPWORDS = {
    "which", "what", "where", "when", "who", "how", "issued", "receipt", "date",
    "dated", "on", "for", "with", "total", "amount", "charge", "store", "vendor",
    "enterprise", "company", "location", "outlet", "shop", "the", "a", "an", "is",
    "was", "appear", "appears", "costing", "show", "showing", "given", "give", "from"
}

def extract_keywords_rule_based(query: str) -> str:
    # 1. Preserve financial and date regex patterns
    financial_tokens = re.findall(r"rm\s*\d+(?:\.\d{2})?|\$\d+(?:\.\d{2})?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", query, re.I)
    # 2. Extract capitalized or alphanumeric terms
    all_words = tokenize_words(query)
    filtered = [w for w in all_words if w not in QUESTION_STOPWORDS and len(w) >= 2]
    combined = list(dict.fromkeys(financial_tokens + filtered))
    return " ".join(combined) if combined else query

def extract_predicted_doc_fields(doc_text: str) -> str:
    # Rule-based field extraction for document expansion
    lines = [l.strip() for l in doc_text.splitlines() if l.strip()]
    vendor = lines[0] if lines else ""
    dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", doc_text)
    totals = re.findall(r"(?:total|amount|rm|\$)\s*[:=]?\s*(\d+\.\d{2})", doc_text, re.I)
    expansion = f"Vendor: {vendor} Date: {' '.join(dates[:2])} Total: {' '.join(totals[:2])}"
    return expansion

# Build expanded corpus
expanded_clean_corpus = [
    d + "\n" + extract_predicted_doc_fields(d) for d in test_doc_clean_ocr
]

# Pre-index base and expanded corpora
corpus_w_base = [tokenize_words(d) for d in test_doc_clean_ocr]
corpus_w_exp = [tokenize_words(d) for d in expanded_clean_corpus]

bm25_base = BM25Okapi(corpus_w_base, k1=1.5, b=0.75)
bm25_expanded = BM25Okapi(corpus_w_exp, k1=1.5, b=0.75)

# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE REMEDIES ACROSS LIGHT AND HEAVY QUERIES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Phase 6.2: Evaluating Remedies against Baseline BM25 ===")

remedy_eval_records = []
queries_cache = []

for q_type, q_raw_list in [("Light Paraphrase", [p["light_paraphrase"] for p in test_probes]),
                           ("Answerable Heavy", [heavy_map.get(p["probe_id"], p.get("heavy_paraphrase", p["verbatim"])) for p in test_probes])]:

    # 1. Baseline BM25 (no remedy)
    t0 = time.time()
    mrr_base = []
    for i in range(len(test_probes)):
        sw = bm25_base.get_scores(tokenize_words(q_raw_list[i]))
        r = int(np.where(np.argsort(-sw) == target_indices[i])[0][0]) + 1
        mrr_base.append(1.0 / r)
    lat_base = ((time.time() - t0) / len(test_probes)) * 1000.0

    # 2. Remedy B: Rule-based Keyword Extraction
    t0 = time.time()
    mrr_kw = []
    q_kw_list = [extract_keywords_rule_based(q) for q in q_raw_list]
    for i in range(len(test_probes)):
        sw = bm25_base.get_scores(tokenize_words(q_kw_list[i]))
        r = int(np.where(np.argsort(-sw) == target_indices[i])[0][0]) + 1
        mrr_kw.append(1.0 / r)
    lat_kw = ((time.time() - t0) / len(test_probes)) * 1000.0

    # 3. Remedy C: Document Expansion (Appended Predicted Fields)
    t0 = time.time()
    mrr_exp = []
    for i in range(len(test_probes)):
        sw = bm25_expanded.get_scores(tokenize_words(q_raw_list[i]))
        r = int(np.where(np.argsort(-sw) == target_indices[i])[0][0]) + 1
        mrr_exp.append(1.0 / r)
    lat_exp = ((time.time() - t0) / len(test_probes)) * 1000.0

    # 4. Remedy D: Pseudo-Relevance Feedback (RM3 / PRF, top-3 docs, alpha=0.6)
    t0 = time.time()
    mrr_prf = []
    q_prf_list = []
    for i in range(len(test_probes)):
        q_toks = tokenize_words(q_raw_list[i])
        sw = bm25_base.get_scores(q_toks)
        top_doc_indices = np.argsort(-sw)[:3]
        # Extract prominent terms from top-3 docs not in original query
        feedback_terms = []
        for d_idx in top_doc_indices:
            d_words = corpus_w_base[d_idx]
            for w in d_words:
                if len(w) >= 3 and w not in q_toks and w not in QUESTION_STOPWORDS:
                    feedback_terms.append(w)
        counts = collections.Counter(feedback_terms)
        top_exp_terms = [term for term, _ in counts.most_common(5)]
        expanded_query = q_raw_list[i] + " " + " ".join(top_exp_terms)
        q_prf_list.append(expanded_query)

        sw_prf = bm25_base.get_scores(tokenize_words(expanded_query))
        r = int(np.where(np.argsort(-sw_prf) == target_indices[i])[0][0]) + 1
        mrr_prf.append(1.0 / r)
    lat_prf = ((time.time() - t0) / len(test_probes)) * 1000.0

    # Save queries to cache
    for i in range(len(test_probes)):
        queries_cache.append({
            "probe_id": test_probes[i]["probe_id"],
            "query_type": q_type,
            "raw_query": q_raw_list[i],
            "keyword_extracted": q_kw_list[i],
            "prf_expanded": q_prf_list[i],
        })

    # Paired comparisons against baseline
    for rem_name, rem_mrr, rem_lat, cost in [
        ("Keyword Extraction (b)", mrr_kw, lat_kw, "$0.00"),
        ("Document Expansion (c)", mrr_exp, lat_exp, "$0.00"),
        ("PRF RM3 Expansion (d)", mrr_prf, lat_prf, "$0.00"),
    ]:
        diff_arr = np.array(rem_mrr) - np.array(mrr_base)
        diff_ci = cluster_bootstrap_ci(diff_arr, cluster_targets)
        mean_gain = float(np.mean(diff_arr))

        # Pre-registered decision rule: effective only if lower CI > 0
        is_effective = diff_ci[0] > 0.0
        verdict = "Effective (CI > 0)" if is_effective else ("Negative (CI < 0)" if diff_ci[1] < 0.0 else "Inconclusive / Tie (CI spans 0)")

        remedy_eval_records.append({
            "Query_Regime": q_type,
            "Remedy": rem_name,
            "Baseline_BM25_MRR": round(float(np.mean(mrr_base)), 4),
            "Remedy_MRR": round(float(np.mean(rem_mrr)), 4),
            "Paired_MRR_Gain": round(mean_gain, 4),
            "Gain_CI_Lower": round(diff_ci[0], 4),
            "Gain_CI_Upper": round(diff_ci[1], 4),
            "Added_Latency_ms": round(rem_lat - lat_base, 2),
            "Cost_Per_Query": cost,
            "PreRegistered_Verdict": verdict,
        })

df_remedies = pd.DataFrame(remedy_eval_records)
df_remedies.to_csv(V6_RESULTS_DIR / "remedies_summary.csv", index=False)
with open(V6_RESULTS_DIR / "remedies_queries_v6.json", "w", encoding="utf-8") as f:
    json.dump(queries_cache, f, indent=2)

print(f"Saved remedies evaluation to {V6_RESULTS_DIR / 'remedies_summary.csv'}")
for r in remedy_eval_records:
    print(f"  [{r['Query_Regime'][:5]} | {r['Remedy']:<22}]: Baseline={r['Baseline_BM25_MRR']:.4f} -> Remedy={r['Remedy_MRR']:.4f} (Gain={r['Paired_MRR_Gain']:+.4f} [{r['Gain_CI_Lower']:+.4f}, {r['Gain_CI_Upper']:+.4f}]) -> {r['PreRegistered_Verdict']}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE LATEX TABLE FRAGMENT
# ─────────────────────────────────────────────────────────────────────────────
with open(LATEX_DIR / "table_remedies_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/remedies_summary.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Query-Side and Document-Side Remedy Evaluation on Clean OCR ($N=306$). Paired MRR gains with 95\\% cluster-bootstrap CIs, latency, and costs.}\\label{tab:remedies_v6}\n")
    f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
    f.write("Regime & Remedy Strategy & Baseline MRR & Remedy MRR & Paired Gain [95\\% CI] & Added Lat. & Cost \\\\\n\\midrule\n")
    cur_reg = ""
    for _, r in df_remedies.iterrows():
        reg_label = r["Query_Regime"] if r["Query_Regime"] != cur_reg else ""
        cur_reg = r["Query_Regime"]
        gain_str = f"{r['Paired_MRR_Gain']:+.4f} [{r['Gain_CI_Lower']:+.4f}, {r['Gain_CI_Upper']:+.4f}]"
        f.write(f"{reg_label} & {r['Remedy']} & {r['Baseline_BM25_MRR']:.4f} & {r['Remedy_MRR']:.4f} & {gain_str} & {r['Added_Latency_ms']:+.1f}\\,ms & {r['Cost_Per_Query']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print(f"Generated LaTeX fragment {LATEX_DIR / 'table_remedies_v6.tex'}")

print("\n================================================================================")
print("Phase 6 Execution Finished Successfully!")
print("================================================================================")
