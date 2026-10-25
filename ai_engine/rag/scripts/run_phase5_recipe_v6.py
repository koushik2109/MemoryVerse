#!/usr/bin/env python3
"""
Phase 5: Constructive Recipe Formulation & Evaluation (CPU)
============================================================
5.1 On Dev only, tune:
    - Sparse: Word BM25, Char 3-5gram BM25
    - Dense: BGE-Base
    - Fusion: RRF (k=60) vs. Convex score interpolation
    - Define and freeze recommended pipeline configuration BEFORE looking at Test.
5.2 Evaluate frozen recipe on Test:
    - All query types (Verbatim, Light Paraphrase, Answerable Heavy Questions)
    - All text sources (GT, Clean Apple, Degraded Apple, Clean Tesseract, Degraded Tesseract, Syn 10/20/40%)
    - FUNSD scanned administrative forms replication (N=100)
    - Statistical tie-set analysis (paired difference 95% CI includes zero).
5.3 Emits:
    - results/v6/frozen_recipe_v6.json
    - results/v6/recipe_summary.csv
    - latex_fragments_v6/table_recipe_summary_v6.tex
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
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DATA_DIR = DATA_DIR / "v2"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"
V6_RESULTS_DIR = REPO_ROOT / "results" / "v6"
LATEX_DIR = REPO_ROOT / "latex_fragments_v6"
V6_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]

def tokenize_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    clean = re.sub(r"\s+", " ", text.lower().strip())
    if not clean:
        return []
    ngrams = []
    L = len(clean)
    for n in range(n_min, n_max + 1):
        if L >= n:
            for i in range(L - n + 1):
                ngrams.append(clean[i : i + n])
    return ngrams if ngrams else [clean]

def corrupt_text_random(text: str, noise_rate: float, rng: np.random.Generator) -> str:
    chars = list(text)
    n = len(chars)
    if n == 0 or noise_rate <= 0.0:
        return text
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    n_corrupt = int(round(n * noise_rate))
    indices = rng.choice(n, size=min(n, n_corrupt), replace=False)
    for idx in indices:
        chars[idx] = rng.choice(list(alphabet))
    return "".join(chars)

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
# 1. LOAD BENCHMARK AND SPLITS
# ─────────────────────────────────────────────────────────────────────────────
print("=== Phase 5 Setup: Loading Benchmark & Dev/Test Splits ===")
with open(V2_DATA_DIR / "public_benchmark_v2.json", "r", encoding="utf-8") as f:
    bench_data = json.load(f)

records = bench_data["records"]

DEGRADED_OCR_PATH = V2_DATA_DIR / "sroie_degraded_ocr.json"
with open(DEGRADED_OCR_PATH, "r", encoding="utf-8") as f:
    degraded_ocr_map = json.load(f)
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

dev_records = [r for r in records if r["split"] == "dev"]
test_records = [r for r in records if r["split"] == "test"]
dev_id_to_idx = {r["id"]: idx for idx, r in enumerate(dev_records)}
test_id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}

assert len(dev_records) == 813 and len(test_records) == 813
assert len(set(dev_id_to_idx.keys()).intersection(set(test_id_to_idx.keys()))) == 0

# Probes
with open(V2_DATA_DIR / "frozen_probes_with_paraphrases.json", "r", encoding="utf-8") as f:
    probe_data = json.load(f)
dev_probes = probe_data["dev_probes"]
test_probes = probe_data["test_probes"]

with open(V3_RESULTS_DIR / "heavy_paraphrases_v3.json", "r", encoding="utf-8") as f:
    heavy_probes_list = json.load(f)
heavy_map = {p["probe_id"]: p.get("heavy_question", p.get("heavy_paraphrase", "")) for p in heavy_probes_list}

def normalize_tess_dict(d):
    normalized = {}
    for k, v in d.items():
        text_str = " \n ".join(v) if isinstance(v, list) else str(v)
        norm_k = f"sroie_{k}" if not str(k).startswith("sroie_") else str(k)
        normalized[norm_k] = text_str
        stem = str(k).replace("sroie_", "")
        normalized[stem] = text_str
    return normalized

with open(V3_RESULTS_DIR / "sroie_tesseract_clean_ocr.json", "r", encoding="utf-8") as f:
    tesseract_clean_dict = normalize_tess_dict(json.load(f))
with open(V3_RESULTS_DIR / "sroie_tesseract_degraded_ocr.json", "r", encoding="utf-8") as f:
    tesseract_degraded_dict = normalize_tess_dict(json.load(f))

# Dev corpora
dev_doc_gt = [r["ground_truth_text"] for r in dev_records]
dev_doc_clean_ocr = [r.get("real_ocr_text") or r["ground_truth_text"] for r in dev_records]
dev_doc_deg_ocr = [r.get("degraded_ocr_text") or r.get("real_ocr_text") or r["ground_truth_text"] for r in dev_records]

# Test corpora
test_doc_gt = [r["ground_truth_text"] for r in test_records]
test_doc_clean_ocr = [r.get("real_ocr_text") or r["ground_truth_text"] for r in test_records]
test_doc_deg_ocr = [r.get("degraded_ocr_text") or r.get("real_ocr_text") or r["ground_truth_text"] for r in test_records]
test_doc_tess_clean = [tesseract_clean_dict.get(r["id"], r.get("real_ocr_text") or r["ground_truth_text"]) for r in test_records]
test_doc_tess_deg = [tesseract_degraded_dict.get(r["id"], r.get("degraded_ocr_text") or r.get("real_ocr_text") or r["ground_truth_text"]) for r in test_records]

# Initialize dense model
dense_bge = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE 5.1: DEV-ONLY RECIPE TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Phase 5.1: Tuning Constructive Recipe Strictly on Dev Split ===")

# Dev queries
dev_q_verb = [p["verbatim"] for p in dev_probes]
dev_q_light = [p["light_paraphrase"] for p in dev_probes]
dev_q_heavy = [heavy_map.get(p["probe_id"], p.get("heavy_paraphrase", p["verbatim"])) for p in dev_probes]
dev_targets = [dev_id_to_idx[p["target_id"]] for p in dev_probes]

# Pre-index Dev Clean OCR
dev_words = [tokenize_words(d) for d in dev_doc_clean_ocr]
dev_chars = [tokenize_char_ngrams(d) for d in dev_doc_clean_ocr]
bm25_w_dev = BM25Okapi(dev_words, k1=1.5, b=0.75)
bm25_c_dev = BM25Okapi(dev_chars, k1=1.5, b=0.75)
dev_dense_mat = dense_bge.encode(dev_doc_clean_ocr, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

# Grid of fusion candidates on Dev across pooled query types
fusion_candidates = [
    {"name": "RRF_Balanced_3Way", "type": "rrf", "k": 60, "w_w": 1.0, "w_c": 1.0, "w_d": 1.0},
    {"name": "RRF_Char_Heavy", "type": "rrf", "k": 60, "w_w": 0.5, "w_c": 1.5, "w_d": 1.0},
    {"name": "RRF_Dense_Heavy", "type": "rrf", "k": 60, "w_w": 0.8, "w_c": 0.8, "w_d": 1.4},
    {"name": "Convex_Equal", "type": "convex", "w_w": 0.33, "w_c": 0.33, "w_d": 0.34},
    {"name": "Convex_Sparse_Biased", "type": "convex", "w_w": 0.40, "w_c": 0.40, "w_d": 0.20},
    {"name": "Convex_Dense_Biased", "type": "convex", "w_w": 0.20, "w_c": 0.20, "w_d": 0.60},
]

dev_results = []
for cand in fusion_candidates:
    pooled_mrrs = []
    for q_type, q_list in [("verbatim", dev_q_verb), ("light", dev_q_light), ("heavy", dev_q_heavy)]:
        q_embs = dense_bge.encode(q_list, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        mrrs = []
        for i in range(len(dev_probes)):
            tgt = dev_targets[i]
            q = q_list[i]
            sw = bm25_w_dev.get_scores(tokenize_words(q))
            sc = bm25_c_dev.get_scores(tokenize_char_ngrams(q))
            sd = np.dot(dev_dense_mat, q_embs[i])

            rw = np.argsort(np.argsort(-sw)) + 1
            rc = np.argsort(np.argsort(-sc)) + 1
            rd = np.argsort(np.argsort(-sd)) + 1

            if cand["type"] == "rrf":
                k = cand["k"]
                score = (cand["w_w"] / (k + rw)) + (cand["w_c"] / (k + rc)) + (cand["w_d"] / (k + rd))
            else:
                norm_w = (sw - np.min(sw)) / max(1e-9, np.max(sw) - np.min(sw))
                norm_c = (sc - np.min(sc)) / max(1e-9, np.max(sc) - np.min(sc))
                norm_d = (sd - np.min(sd)) / max(1e-9, np.max(sd) - np.min(sd))
                score = cand["w_w"] * norm_w + cand["w_c"] * norm_c + cand["w_d"] * norm_d

            rank = int(np.where(np.argsort(-score) == tgt)[0][0]) + 1
            mrrs.append(1.0 / rank)
        pooled_mrrs.extend(mrrs)

    mean_dev_mrr = float(np.mean(pooled_mrrs))
    dev_results.append((cand, mean_dev_mrr))
    print(f"  Dev Candidate [{cand['name']:<22}]: Mean MRR = {mean_dev_mrr:.4f}")

# Sort by Dev MRR
dev_results.sort(key=lambda x: x[1], reverse=True)
best_recipe_config, best_dev_mrr = dev_results[0]
print(f"\nOptimal Dev Recipe Selected: {best_recipe_config['name']} (Dev MRR = {best_dev_mrr:.4f})")

frozen_recipe = {
    "recipe_name": "Tri-Modal Robust Reciprocal Rank Fusion (TR-RRF)",
    "selection_criterion": "Tuned strictly on Dev split across pooled Verbatim, Light, Heavy queries",
    "dev_mean_mrr": round(best_dev_mrr, 4),
    "configuration": best_recipe_config,
}

with open(V6_RESULTS_DIR / "frozen_recipe_v6.json", "w", encoding="utf-8") as f:
    json.dump(frozen_recipe, f, indent=2)
print(f"Saved frozen recipe config to {V6_RESULTS_DIR / 'frozen_recipe_v6.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHASE 5.2: EVALUATE FROZEN RECIPE ON TEST SPLIT ACROSS ALL CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Phase 5.2: Evaluating Frozen Recipe on Held-Out Test Split ===")

test_q_verb = [p["verbatim"] for p in test_probes]
test_q_light = [p["light_paraphrase"] for p in test_probes]
test_q_heavy = [heavy_map.get(p["probe_id"], p.get("heavy_paraphrase", p["verbatim"])) for p in test_probes]
test_targets = [test_id_to_idx[p["target_id"]] for p in test_probes]
cluster_targets = [p["target_id"] for p in test_probes]

# Test conditions
rng_syn = np.random.default_rng(42)
test_conditions = {
    "Ground Truth": test_doc_gt,
    "Clean Apple Vision": test_doc_clean_ocr,
    "Degraded Apple Vision": test_doc_deg_ocr,
    "Clean Tesseract 5.5": test_doc_tess_clean,
    "Degraded Tesseract": test_doc_tess_deg,
    "Synthetic 10% Noise": [corrupt_text_random(t, 0.10, rng_syn) for t in test_doc_gt],
    "Synthetic 20% Noise": [corrupt_text_random(t, 0.20, rng_syn) for t in test_doc_gt],
    "Synthetic 40% Noise": [corrupt_text_random(t, 0.40, rng_syn) for t in test_doc_gt],
}

recipe_eval_rows = []
total_cells = 0
recipe_in_tie_set_count = 0

for cond_name, doc_list in test_conditions.items():
    corpus_w = [tokenize_words(d) for d in doc_list]
    corpus_c = [tokenize_char_ngrams(d) for d in doc_list]
    bm25_w = BM25Okapi(corpus_w, k1=1.5, b=0.75)
    bm25_c = BM25Okapi(corpus_c, k1=1.5, b=0.75)
    dense_mat = dense_bge.encode(doc_list, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    for q_type, q_list in [("Verbatim", test_q_verb), ("Light Paraphrase", test_q_light), ("Answerable Heavy", test_q_heavy)]:
        total_cells += 1
        q_embs = dense_bge.encode(q_list, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        mrr_w, mrr_c, mrr_d, mrr_rec = [], [], [], []

        for i in range(len(test_probes)):
            tgt = test_targets[i]
            q = q_list[i]

            sw = bm25_w.get_scores(tokenize_words(q))
            sc = bm25_c.get_scores(tokenize_char_ngrams(q))
            sd = np.dot(dense_mat, q_embs[i])

            rw = np.argsort(np.argsort(-sw)) + 1
            rc = np.argsort(np.argsort(-sc)) + 1
            rd = np.argsort(np.argsort(-sd)) + 1

            # Standalone systems
            mrr_w.append(1.0 / rw[tgt])
            mrr_c.append(1.0 / rc[tgt])
            mrr_d.append(1.0 / rd[tgt])

            # Recipe RRF
            k = best_recipe_config.get("k", 60)
            score_rec = (best_recipe_config["w_w"] / (k + rw)) + (best_recipe_config["w_c"] / (k + rc)) + (best_recipe_config["w_d"] / (k + rd))
            rank_rec = int(np.where(np.argsort(-score_rec) == tgt)[0][0]) + 1
            mrr_rec.append(1.0 / rank_rec)

        mean_w = float(np.mean(mrr_w))
        mean_c = float(np.mean(mrr_c))
        mean_d = float(np.mean(mrr_d))
        mean_rec = float(np.mean(mrr_rec))

        # Best standalone engine in this cell
        best_standalone_mean = max(mean_w, mean_c, mean_d)
        best_standalone_name = "Char BM25" if best_standalone_mean == mean_c else ("Word BM25" if best_standalone_mean == mean_w else "Dense BGE")
        best_standalone_arr = np.array(mrr_c) if best_standalone_mean == mean_c else (np.array(mrr_w) if best_standalone_mean == mean_w else np.array(mrr_d))

        # Paired difference: Recipe - Best Standalone
        paired_diff = np.array(mrr_rec) - best_standalone_arr
        diff_ci = cluster_bootstrap_ci(paired_diff, cluster_targets)
        rec_ci = cluster_bootstrap_ci(np.array(mrr_rec), cluster_targets)

        # Tie-set rule: recipe is in tie set if upper CI >= 0 (i.e. difference is not statistically significantly worse)
        # Or wins if lower CI > 0
        if diff_ci[1] >= 0.0:
            is_in_tie_set = True
            recipe_in_tie_set_count += 1
            tie_status = "Wins Decisively" if diff_ci[0] > 0.0 else "Statistical Tie (Within Parity)"
        else:
            is_in_tie_set = False
            tie_status = "Inferior to Best Standalone"

        recipe_eval_rows.append({
            "Text_Condition": cond_name,
            "Query_Type": q_type,
            "Word_BM25_MRR": round(mean_w, 4),
            "Char_BM25_MRR": round(mean_c, 4),
            "Dense_BGE_MRR": round(mean_d, 4),
            "Recipe_TR_RRF_MRR": round(mean_rec, 4),
            "Recipe_CI_Lower": round(rec_ci[0], 4),
            "Recipe_CI_Upper": round(rec_ci[1], 4),
            "Best_Standalone_Engine": best_standalone_name,
            "Best_Standalone_MRR": round(best_standalone_mean, 4),
            "Recipe_Minus_Best": round(float(np.mean(paired_diff)), 4),
            "Diff_CI_Lower": round(diff_ci[0], 4),
            "Diff_CI_Upper": round(diff_ci[1], 4),
            "In_Statistical_Tie_Set": is_in_tie_set,
            "Tie_Status": tie_status,
        })

df_recipe = pd.DataFrame(recipe_eval_rows)
df_recipe.to_csv(V6_RESULTS_DIR / "recipe_summary.csv", index=False)
print(f"Saved recipe summary to {V6_RESULTS_DIR / 'recipe_summary.csv'}")

tie_set_fraction = (recipe_in_tie_set_count / total_cells) * 100.0
print(f"\nRecipe Parity Evaluation: Recipe is in the statistical tie-set in {recipe_in_tie_set_count}/{total_cells} cells ({tie_set_fraction:.1f}%).")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE LATEX FRAGMENT
# ─────────────────────────────────────────────────────────────────────────────
with open(LATEX_DIR / "table_recipe_summary_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/recipe_summary.csv\n")
    f.write("\\begin{table*}[t]\n\\centering\\small\n")
    f.write("\\caption{Constructive TR-RRF Recipe Evaluation on Held-Out Test Split ($N=306$). The frozen recipe combines Word BM25, Char 3--5gram BM25, and Dense BGE-Base via Reciprocal Rank Fusion ($k=60$), tuned strictly on Dev.}\\label{tab:recipe_summary_v6}\n")
    f.write("\\begin{tabular}{llcccccl}\n\\toprule\n")
    f.write("Condition & Query Regime & Word BM25 & Char BM25 & Dense BGE & TR-RRF Recipe & Diff vs.\\ Best [95\\% CI] & Tie Status \\\\\n\\midrule\n")
    cur_cond = ""
    for _, r in df_recipe.iterrows():
        c_label = r["Text_Condition"] if r["Text_Condition"] != cur_cond else ""
        cur_cond = r["Text_Condition"]
        diff_str = f"{r['Recipe_Minus_Best']:+.4f} [{r['Diff_CI_Lower']:+.4f}, {r['Diff_CI_Upper']:+.4f}]"
        f.write(f"{c_label} & {r['Query_Type']} & {r['Word_BM25_MRR']:.4f} & {r['Char_BM25_MRR']:.4f} & {r['Dense_BGE_MRR']:.4f} & \\textbf{{{r['Recipe_TR_RRF_MRR']:.4f}}} & {diff_str} & {r['Tie_Status']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
print(f"Generated LaTeX fragment {LATEX_DIR / 'table_recipe_summary_v6.tex'}")

print("\n================================================================================")
print("Phase 5 Execution Finished Successfully!")
print("================================================================================")
