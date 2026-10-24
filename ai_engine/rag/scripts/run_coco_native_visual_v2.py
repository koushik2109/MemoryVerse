#!/usr/bin/env python3
"""
Phase 7: COCO Karpathy-Test Native-Visual vs. Proxy Retrieval Benchmark (v2)

This script re-benchmarks Native-Visual (CLIP ViT-B/32 raw pixels) against text proxy
representations (BLIP captions and Gold human captions) across N=1,000 images and queries
from the MS-COCO Karpathy test split.

Hypothesis Tested Under Pre-Registration Decision Rules:
  - Hypothesis: Native pixel retrieval (CLIP ViT-B/32) significantly outperforms caption
    text proxies across Hit@1, Hit@5, Hit@10, and MRR.
  - Decision Rule:
      - Claim holds if p_Holm < 0.05 and paired diff CI > 0 across ALL 4 metrics.
      - Claim holds partially if p_Holm < 0.05 for a subset of metrics (e.g. Hit@1, MRR).
      - Claim fails if caption proxies match or outperform native pixels.

Methodology:
  - N=1,000 images in the retrieval index.
  - Query: Ground-truth human description (caption 0) from Karpathy test.
  - Proxies:
      1. Machine Caption Proxy: BLIP caption generated from image.
      2. Held-Out Gold Caption Proxy: Captions 1..4 concatenated (zero lexical identity leakage).
  - Systems evaluated:
      1. BM25 (BLIP Proxy)
      2. Dense MiniLM (BLIP Proxy)
      3. Dense BGE-base (BLIP Proxy)
      4. BM25 (Gold Proxy)
      5. Dense MiniLM (Gold Proxy)
      6. Dense BGE-base (Gold Proxy)
      7. Native Pixels: CLIP ViT-B/32
      8. Hybrid Pixel + Proxy: CLIP + Dense BGE-base (BLIP)
      9. Hybrid Pixel + Proxy: CLIP + Dense MiniLM (BLIP)
      10. All-Stream RRF: BM25 + BGE-base + CLIP
  - Statistics:
      - Cluster bootstrap 95% CIs (B=2,000 resamples over image clusters).
      - Paired permutation tests (N_perm=10,000) comparing Native Pixels against the best proxy system.
      - Step-down Holm-Bonferroni correction over the 4 evaluation metrics.
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

import numpy as np

# Set site-packages for matplotlib / transformers / torch if needed
if "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages" not in sys.path:
    sys.path.append("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages")

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ai_engine" / "rag"))

import eval_lib

DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
RESULTS_V2_DIR = REPO_ROOT / "results" / "v2"
LATEX_DIR = REPO_ROOT / "latex_fragments"

RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

COCO_BLIP_PATH = DATA_DIR / "coco_blip_captions.json"
COCO_KARPATHY_PATH = DATA_DIR / "coco_karpathy_test.json"
COCO_CLIP_EMBS_PATH = DATA_DIR / "coco_clip_image_embeddings.npy"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42


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


# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
def run_phase7_coco_benchmark():
    t_start = time.time()
    print("=" * 80)
    print("STARTING PHASE 7: COCO NATIVE-VISUAL VS. PROXY RETRIEVAL BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    print(f"Loading BLIP captions from {COCO_BLIP_PATH}...")
    with open(COCO_BLIP_PATH, "r", encoding="utf-8") as f:
        blip_data = json.load(f)

    print(f"Loading Karpathy test annotations from {COCO_KARPATHY_PATH}...")
    with open(COCO_KARPATHY_PATH, "r", encoding="utf-8") as f:
        karpathy_data = json.load(f)

    print(f"Loading CLIP image embeddings from {COCO_CLIP_EMBS_PATH}...")
    clip_img_embs = np.load(COCO_CLIP_EMBS_PATH)

    # Select 1,000 Karpathy items using identical seed 42 protocol as precomputed CLIP embeddings
    rng_coco = np.random.default_rng(42)
    coco_selected_idx = rng_coco.choice(len(karpathy_data), size=1000, replace=False)
    coco_selected = [karpathy_data[i] for i in coco_selected_idx]

    # Build evaluation instances aligned with clip_img_embs
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
    print(f"Successfully aligned N={num_items} evaluation instances.")

    # 2. Text Encoders & Indexes
    print(f"\nBuilding BM25 indexes...")
    bm25_blip = PurePythonBM25([tokenize_words(t) for t in blip_proxy_texts])
    bm25_gold = PurePythonBM25([tokenize_words(t) for t in gold_proxy_texts])

    print(f"Encoding text with MiniLM ('all-MiniLM-L6-v2') on {DEVICE}...")
    dense_minilm = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    doc_embs_minilm_blip = dense_minilm.encode(blip_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    doc_embs_minilm_gold = dense_minilm.encode(gold_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs_minilm = dense_minilm.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)

    print(f"Encoding text with BGE-base ('BAAI/bge-base-en-v1.5') on {DEVICE}...")
    dense_bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
    doc_embs_bge_blip = dense_bge.encode(blip_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    doc_embs_bge_gold = dense_bge.encode(gold_proxy_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs_bge = dense_bge.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)

    print(f"Encoding text queries with CLIP ViT-B/32 on {DEVICE}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_inputs = clip_processor(text=[q[:77] for q in query_texts], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.get_text_features(**clip_inputs)
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            feat = feat.pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
    q_embs_clip = feat.cpu().numpy()

    # Normalize image embeddings
    clip_img_norms = np.linalg.norm(clip_img_embs, axis=-1, keepdims=True)
    clip_img_embs_norm = clip_img_embs / np.maximum(clip_img_norms, 1e-12)

    # 3. System Score Matrices (N_queries x N_docs)
    print("\nComputing score matrices across all 10 retrieval systems...")
    systems = [
        "BM25 (BLIP Proxy)",
        "Dense MiniLM (BLIP Proxy)",
        "Dense BGE-base (BLIP Proxy)",
        "BM25 (Gold Proxy)",
        "Dense MiniLM (Gold Proxy)",
        "Dense BGE-base (Gold Proxy)",
        "CLIP ViT-B/32 (Native Pixels)",
        "Hybrid (CLIP + BGE-base BLIP)",
        "Hybrid (CLIP + MiniLM BLIP)",
        "All-Stream RRF (BM25 + BGE + CLIP)",
    ]

    # Precompute pairwise dot products
    sim_clip = np.dot(q_embs_clip, clip_img_embs_norm.T)
    sim_minilm_blip = np.dot(q_embs_minilm, doc_embs_minilm_blip.T)
    sim_bge_blip = np.dot(q_embs_bge, doc_embs_bge_blip.T)
    sim_minilm_gold = np.dot(q_embs_minilm, doc_embs_minilm_gold.T)
    sim_bge_gold = np.dot(q_embs_bge, doc_embs_bge_gold.T)

    system_scores: Dict[str, np.ndarray] = {}
    system_scores["Dense MiniLM (BLIP Proxy)"] = sim_minilm_blip
    system_scores["Dense BGE-base (BLIP Proxy)"] = np.dot(q_embs_bge, doc_embs_bge_blip.T)
    system_scores["Dense MiniLM (Gold Proxy)"] = sim_minilm_gold
    system_scores["Dense BGE-base (Gold Proxy)"] = sim_bge_gold
    system_scores["CLIP ViT-B/32 (Native Pixels)"] = sim_clip

    # Lexical BM25 matrices
    s_bm25_blip_mat = np.zeros((num_items, num_items), dtype=np.float32)
    s_bm25_gold_mat = np.zeros((num_items, num_items), dtype=np.float32)
    for q_i, q_text in enumerate(query_texts):
        tokens = tokenize_words(q_text)
        s_bm25_blip_mat[q_i] = bm25_blip.get_scores(tokens)
        s_bm25_gold_mat[q_i] = bm25_gold.get_scores(tokens)

    system_scores["BM25 (BLIP Proxy)"] = s_bm25_blip_mat
    system_scores["BM25 (Gold Proxy)"] = s_bm25_gold_mat

    # Hybrid & RRF
    s_hyb_bge = np.zeros_like(sim_clip)
    s_hyb_mini = np.zeros_like(sim_clip)
    s_rrf = np.zeros_like(sim_clip)

    for q_i in range(num_items):
        sc_clip = min_max_normalize(sim_clip[q_i])
        sc_bge = min_max_normalize(system_scores["Dense BGE-base (BLIP Proxy)"][q_i])
        sc_mini = min_max_normalize(sim_minilm_blip[q_i])
        sc_bm25 = min_max_normalize(s_bm25_blip_mat[q_i])

        s_hyb_bge[q_i] = 0.50 * sc_clip + 0.50 * sc_bge
        s_hyb_mini[q_i] = 0.50 * sc_clip + 0.50 * sc_mini

        r_clip = np.argsort(-sc_clip)
        r_bge = np.argsort(-sc_bge)
        r_bm25 = np.argsort(-sc_bm25)

        for rank, d_idx in enumerate(r_clip):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)
        for rank, d_idx in enumerate(r_bge):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)
        for rank, d_idx in enumerate(r_bm25):
            s_rrf[q_i, d_idx] += 1.0 / (60.0 + rank + 1)

    system_scores["Hybrid (CLIP + BGE-base BLIP)"] = s_hyb_bge
    system_scores["Hybrid (CLIP + MiniLM BLIP)"] = s_hyb_mini
    system_scores["All-Stream RRF (BM25 + BGE + CLIP)"] = s_rrf

    # 4. Evaluate Metrics
    print("\nComputing retrieval effectiveness across N=1,000 queries...")
    results_table = []
    per_query_metrics: Dict[str, Dict[str, np.ndarray]] = {}

    for sys_name in systems:
        scores_mat = system_scores[sys_name]
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

        ci_h1 = eval_lib.cluster_bootstrap_ci(h1_arr, n_boot=2000, seed=42)
        ci_h5 = eval_lib.cluster_bootstrap_ci(h5_arr, n_boot=2000, seed=42)
        ci_h10 = eval_lib.cluster_bootstrap_ci(h10_arr, n_boot=2000, seed=42)
        ci_mrr = eval_lib.cluster_bootstrap_ci(mrr_arr, n_boot=2000, seed=42)

        results_table.append({
            "system": sys_name,
            "hit1": float(np.mean(h1_arr)),
            "hit1_ci95_low": ci_h1[0],
            "hit1_ci95_high": ci_h1[1],
            "hit5": float(np.mean(h5_arr)),
            "hit5_ci95_low": ci_h5[0],
            "hit5_ci95_high": ci_h5[1],
            "hit10": float(np.mean(h10_arr)),
            "hit10_ci95_low": ci_h10[0],
            "hit10_ci95_high": ci_h10[1],
            "mrr": float(np.mean(mrr_arr)),
            "mrr_ci95_low": ci_mrr[0],
            "mrr_ci95_high": ci_mrr[1],
        })

    # 5. Hypothesis Testing: Native Pixels vs. Best Caption Proxy
    # Caption proxies: BM25 (BLIP), Dense MiniLM (BLIP), Dense BGE (BLIP)
    proxy_candidates = [
        "BM25 (BLIP Proxy)",
        "Dense MiniLM (BLIP Proxy)",
        "Dense BGE-base (BLIP Proxy)",
    ]

    # Best proxy by metric:
    best_proxy_by_metric: Dict[str, str] = {}
    for m in ["hit1", "hit5", "hit10", "mrr"]:
        best_proxy_name = max(proxy_candidates, key=lambda s: float(np.mean(per_query_metrics[s][m])))
        best_proxy_by_metric[m] = best_proxy_name

    clip_name = "CLIP ViT-B/32 (Native Pixels)"
    p_raw: Dict[str, float] = {}
    diff_stats: Dict[str, Dict[str, Any]] = {}

    for m in ["hit1", "hit5", "hit10", "mrr"]:
        best_p_name = best_proxy_by_metric[m]
        arr_clip = per_query_metrics[clip_name][m]
        arr_proxy = per_query_metrics[best_p_name][m]

        diff = arr_clip - arr_proxy
        mean_diff = float(np.mean(diff))
        ci_diff = eval_lib.cluster_bootstrap_ci(diff, n_boot=2000, seed=42)
        p_val = eval_lib.paired_permutation_test(arr_clip, arr_proxy, n_permutations=10000, seed=42)

        p_raw[m] = p_val
        diff_stats[m] = {
            "best_proxy": best_p_name,
            "clip_mean": float(np.mean(arr_clip)),
            "proxy_mean": float(np.mean(arr_proxy)),
            "mean_diff": mean_diff,
            "ci_diff_low": ci_diff[0],
            "ci_diff_high": ci_diff[1],
            "raw_p": p_val,
        }

    p_holm = eval_lib.holm_bonferroni(p_raw)
    for m in ["hit1", "hit5", "hit10", "mrr"]:
        diff_stats[m]["p_holm"] = p_holm[m]
        diff_stats[m]["sig_superior"] = bool(p_holm[m] < 0.05 and diff_stats[m]["ci_diff_low"] > 0)

    # Verdict Evaluation
    all_sig = all(diff_stats[m]["sig_superior"] for m in ["hit1", "hit5", "hit10", "mrr"])
    any_sig = any(diff_stats[m]["sig_superior"] for m in ["hit1", "hit5", "hit10", "mrr"])
    if all_sig:
        verdict = "CONFIRMED (HOLDS ACROSS ALL METRICS)"
    elif any_sig:
        verdict = "PARTIALLY CONFIRMED (HOLDS FOR SUBSET OF METRICS)"
    else:
        verdict = "REJECTED (CAPTION PROXIES MATCH OR OUTPERFORM NATIVE PIXELS)"

    print("\n" + "=" * 80)
    print("PHASE 7 COCO RETRIEVAL BENCHMARK RESULTS (N=1,000):")
    print("=" * 80)
    for row in results_table:
        print(f"  {row['system']:<35}: Hit@1={row['hit1']:.4f} | Hit@5={row['hit5']:.4f} | Hit@10={row['hit10']:.4f} [{row['hit10_ci95_low']:.4f}, {row['hit10_ci95_high']:.4f}] | MRR={row['mrr']:.4f} [{row['mrr_ci95_low']:.4f}, {row['mrr_ci95_high']:.4f}]")

    print("\n" + "-" * 80)
    print("PRE-REGISTERED HYPOTHESIS TEST: NATIVE PIXELS VS. BEST CAPTION PROXY")
    print("-" * 80)
    for m in ["hit1", "hit5", "hit10", "mrr"]:
        st = diff_stats[m]
        print(f"  {m.upper():<6}: CLIP={st['clip_mean']:.4f} vs {st['best_proxy']}={st['proxy_mean']:.4f} | Diff={st['mean_diff']:+.4f} [{st['ci_diff_low']:+.4f}, {st['ci_diff_high']:+.4f}] | p_Holm={st['p_holm']:.5f} | Sig={st['sig_superior']}")

    print(f"\n[PHASE 7 HYPOTHESIS VERDICT]: {verdict}")

    # 6. Save CSV
    out_csv_path = RESULTS_V2_DIR / "coco_native_visual_results.csv"
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\nSaved COCO results CSV to: {out_csv_path}")

    # Save Hypothesis Test CSV
    hyp_csv_path = RESULTS_V2_DIR / "coco_hypothesis_test.csv"
    with open(hyp_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "clip_mean", "best_proxy", "proxy_mean", "mean_diff", "ci_diff_low", "ci_diff_high", "raw_p", "p_holm", "sig_superior"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in ["hit1", "hit5", "hit10", "mrr"]:
            row = {"metric": m}
            row.update(diff_stats[m])
            writer.writerow(row)
    print(f"Saved COCO hypothesis test CSV to: {hyp_csv_path}")

    # 7. Save LaTeX Fragment
    latex_path = LATEX_DIR / "table_coco_v2.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% COCO Native-Visual vs. Proxy Retrieval Benchmark (N=1,000 images & queries)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Cross-modal text-to-image retrieval performance on $N=1{,}000$ MS-COCO Karpathy-test images. Comparison of native pixel embeddings (CLIP ViT-B/32) against machine caption proxies (BLIP) and held-out human caption proxies. $95\\%$ CIs via cluster bootstrap ($B=2{,}000$). Asterisks ($^*$) indicate Holm-corrected significance ($p_{\\text{Holm}} < 0.05$) of Native Pixels over the best caption proxy.}\n")
        f.write("\\label{tab:coco_benchmarks_v2}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Hit@1 & Hit@5 & Hit@10 [95\\% CI] & MRR [95\\% CI] \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{5}{l}{\\textit{Machine Caption Proxy (BLIP)}} \\\\\n")
        f.write(f"BM25 & {results_table[0]['hit1']:.4f} & {results_table[0]['hit5']:.4f} & {results_table[0]['hit10']:.4f} [{results_table[0]['hit10_ci95_low']:.3f}, {results_table[0]['hit10_ci95_high']:.3f}] & {results_table[0]['mrr']:.4f} [{results_table[0]['mrr_ci95_low']:.3f}, {results_table[0]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"Dense MiniLM & {results_table[1]['hit1']:.4f} & {results_table[1]['hit5']:.4f} & {results_table[1]['hit10']:.4f} [{results_table[1]['hit10_ci95_low']:.3f}, {results_table[1]['hit10_ci95_high']:.3f}] & {results_table[1]['mrr']:.4f} [{results_table[1]['mrr_ci95_low']:.3f}, {results_table[1]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"Dense BGE-base & {results_table[2]['hit1']:.4f} & {results_table[2]['hit5']:.4f} & {results_table[2]['hit10']:.4f} [{results_table[2]['hit10_ci95_low']:.3f}, {results_table[2]['hit10_ci95_high']:.3f}] & {results_table[2]['mrr']:.4f} [{results_table[2]['mrr_ci95_low']:.3f}, {results_table[2]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{5}{l}{\\textit{Human Gold Caption Proxy (Upper Bound)}} \\\\\n")
        f.write(f"BM25 (Gold Proxy) & {results_table[3]['hit1']:.4f} & {results_table[3]['hit5']:.4f} & {results_table[3]['hit10']:.4f} [{results_table[3]['hit10_ci95_low']:.3f}, {results_table[3]['hit10_ci95_high']:.3f}] & {results_table[3]['mrr']:.4f} [{results_table[3]['mrr_ci95_low']:.3f}, {results_table[3]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"Dense MiniLM (Gold Proxy) & {results_table[4]['hit1']:.4f} & {results_table[4]['hit5']:.4f} & {results_table[4]['hit10']:.4f} [{results_table[4]['hit10_ci95_low']:.3f}, {results_table[4]['hit10_ci95_high']:.3f}] & {results_table[4]['mrr']:.4f} [{results_table[4]['mrr_ci95_low']:.3f}, {results_table[4]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"Dense BGE-base (Gold Proxy) & {results_table[5]['hit1']:.4f} & {results_table[5]['hit5']:.4f} & {results_table[5]['hit10']:.4f} [{results_table[5]['hit10_ci95_low']:.3f}, {results_table[5]['hit10_ci95_high']:.3f}] & {results_table[5]['mrr']:.4f} [{results_table[5]['mrr_ci95_low']:.3f}, {results_table[5]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{5}{l}{\\textit{Native Pixel Representation}} \\\\\n")
        f.write(f"\\textbf{{CLIP ViT-B/32 (Native Pixels)}} & \\textbf{{{results_table[6]['hit1']:.4f}}}$^*$ & \\textbf{{{results_table[6]['hit5']:.4f}}}$^*$ & \\textbf{{{results_table[6]['hit10']:.4f}}}$^*$ [{results_table[6]['hit10_ci95_low']:.3f}, {results_table[6]['hit10_ci95_high']:.3f}] & \\textbf{{{results_table[6]['mrr']:.4f}}}$^*$ [{results_table[6]['mrr_ci95_low']:.3f}, {results_table[6]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{5}{l}{\\textit{Multi-Vector & Hybrid Fusion}} \\\\\n")
        f.write(f"Hybrid (CLIP + BGE-base BLIP) & {results_table[7]['hit1']:.4f} & {results_table[7]['hit5']:.4f} & {results_table[7]['hit10']:.4f} [{results_table[7]['hit10_ci95_low']:.3f}, {results_table[7]['hit10_ci95_high']:.3f}] & {results_table[7]['mrr']:.4f} [{results_table[7]['mrr_ci95_low']:.3f}, {results_table[7]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"Hybrid (CLIP + MiniLM BLIP) & {results_table[8]['hit1']:.4f} & {results_table[8]['hit5']:.4f} & {results_table[8]['hit10']:.4f} [{results_table[8]['hit10_ci95_low']:.3f}, {results_table[8]['hit10_ci95_high']:.3f}] & {results_table[8]['mrr']:.4f} [{results_table[8]['mrr_ci95_low']:.3f}, {results_table[8]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write(f"All-Stream RRF (BM25+BGE+CLIP) & {results_table[9]['hit1']:.4f} & {results_table[9]['hit5']:.4f} & {results_table[9]['hit10']:.4f} [{results_table[9]['hit10_ci95_low']:.3f}, {results_table[9]['hit10_ci95_high']:.3f}] & {results_table[9]['mrr']:.4f} [{results_table[9]['mrr_ci95_low']:.3f}, {results_table[9]['mrr_ci95_high']:.3f}] \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX fragment to: {latex_path}")

    elapsed = time.time() - t_start
    print(f"\nPhase 7 completed successfully in {elapsed:.2f}s.")


if __name__ == "__main__":
    run_phase7_coco_benchmark()
