#!/usr/bin/env python3
"""
Phase 4: Component Ablation & RAMP-RAG Architecture Re-Evaluation (v2)

This script re-evaluates each architectural component of RAMP-RAG on the held-out
public benchmark test split (N=306 probes, 813 test items pooled with COCO distractors):
  1. Router (predicting query-adaptive alpha_q)
  2. Reliability weighting (rho_ocr heuristic)
  3. Margin-triggered Cascade Ranker (MiniLM cross-encoder vs BGE-reranker-base)
  4. Clean Stream (CLIP visual pixel stream)
  5. Full RAMP-RAG combinations

Hypotheses Tested Under Pre-Registration Decision Rules:
  - H1 (Reliability Weighting): Reject if BM25 alone or Tuned Hybrid achieves higher MRR
    or if the paired difference CI includes zero.
  - H2 (Query Routing): Reject if unweighted All-Stream RRF achieves higher or equal MRR.
  - Reliability AUROC: Measured on balanced clean vs. corrupted evaluation sets across noise levels.

Zero Test Contamination Guarantee:
  - Router trained exclusively on Dev split (N=306).
  - Cascade margin tau tuned exclusively on Dev split (N=306).
  - Hybrid alpha* and CLIP gamma* loaded from Phase 2 Dev tuning.
  - Test split (N=306) evaluated strictly out-of-sample with multi-seeds (42, 43, 44),
    cluster bootstrap CIs (B=2,000), paired permutation tests (N_perm=10,000), and Holm correction.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ai_engine" / "rag"))

import eval_lib

# Set site-packages for matplotlib / transformers if needed
if "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages" not in sys.path:
    sys.path.append("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages")

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DIR = DATA_DIR / "v2"
RESULTS_V2_DIR = REPO_ROOT / "results" / "v2"
LATEX_DIR = REPO_ROOT / "latex_fragments"

RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PATH = V2_DIR / "public_benchmark_v2.json"
FROZEN_PROBES_PATH = V2_DIR / "frozen_probes_with_paraphrases.json"
DEGRADED_OCR_PATH = V2_DIR / "sroie_degraded_ocr.json"
DEV_PARAMS_CACHE = V2_DIR / "dev_tuned_params_cache.json"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── TOKENIZERS & LEXICAL IMPLEMENTATIONS ──────────────────────────────────────
def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


class PurePythonBM25:
    """High-performance Inverted Index Okapi BM25 implementation."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0

        # Build Inverted Index: term -> list of (doc_idx, tf)
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
            chars[idx] = str(rng.choice(list("0123456789abcdefghijklmnopqrstuvwxyz_")))
        elif action == "del":
            chars[idx] = ""
        elif action == "ins":
            chars[idx] = chars[idx] + str(rng.choice(list("0123456789abcdefghijklmnopqrstuvwxyz_")))
    return "".join(chars)


# ─── RELIABILITY HEURISTIC ─────────────────────────────────────────────────────
def compute_observable_reliability(text: str, vocab_set: Set[str]) -> float:
    """Estimates stream reliability rho_ocr(d) from observable surface features."""
    tokens = tokenize_words(text)
    if not tokens:
        return 0.15
    valid_count = sum(1 for t in tokens if t in vocab_set)
    rho_vocab = valid_count / max(1, len(tokens))
    avg_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    len_penalty = min(1.0, max(0.4, avg_len / 3.0))
    return float(np.clip(rho_vocab * len_penalty, 0.10, 1.0))


# ─── LEARNED QUERY ROUTER ──────────────────────────────────────────────────────
class LearnedQueryRouter:
    """Supervised logistic classifier predicting dense vs. lexical routing preference."""

    def __init__(self, bm25_idf: Dict[str, float]):
        self.bm25_idf = bm25_idf
        self.clf = LogisticRegression(C=1.0, random_state=42)
        self.is_fitted = False

    def extract_features(self, query: str) -> np.ndarray:
        tokens = tokenize_words(query)
        if not tokens:
            return np.zeros(6, dtype=np.float32)

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
        cjk_ratio = float(cjk_count / max(len(query), 1))

        digits = sum(1 for c in query if c.isdigit())
        digit_ratio = float(min(1.0, digits / max(len(query), 1)))

        has_upper = float(any(len(w) >= 3 and w.isupper() for w in query.split()))
        has_cue = float(bool(re.search(r"total|tax|gst|cash|rm|invoice|date|receipt|bill|tel|no\.|card|amount|balance", query, re.I)))
        length_feat = float(min(1.0, len(tokens) / 15.0))

        idfs = [self.bm25_idf.get(t, 2.0) for t in tokens]
        avg_idf = float(np.mean(idfs)) / 10.0 if idfs else 0.5

        return np.array([cjk_ratio, digit_ratio, has_upper, has_cue, length_feat, avg_idf], dtype=np.float32)

    def fit(self, queries: List[str], labels: List[int]) -> Dict[str, float]:
        X = np.array([self.extract_features(q) for q in queries], dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        self.clf.fit(X, y)
        self.is_fitted = True
        preds = self.clf.predict(X)
        return {
            "train_accuracy": float(accuracy_score(y, preds)),
            "train_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        }

    def evaluate(self, queries: List[str], labels: List[int]) -> Dict[str, float]:
        if not self.is_fitted:
            return {"accuracy": 0.5, "f1": 0.5}
        X = np.array([self.extract_features(q) for q in queries], dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        preds = self.clf.predict(X)
        return {
            "test_accuracy": float(accuracy_score(y, preds)),
            "test_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        }

    def predict_alpha(self, query: str) -> float:
        if not self.is_fitted:
            has_cue = bool(re.search(r"total|tax|gst|cash|rm|invoice|date|receipt|bill|tel", query, re.I))
            return 0.20 if has_cue else 0.70
        feat = self.extract_features(query).reshape(1, -1)
        prob_dense = float(self.clf.predict_proba(feat)[0, 1])
        return 0.85 if prob_dense >= 0.5 else 0.20


# ─── CASCADE RANKER ────────────────────────────────────────────────────────────
class CascadeRanker:
    """Confidence-Triggered Cascade Ranker: escalates top-k to Cross-Encoder when margin < tau."""

    def __init__(self, cross_encoder: CrossEncoder, tau_margin: float = 0.05, top_k: int = 5):
        self.cross_encoder = cross_encoder
        self.tau_margin = tau_margin
        self.top_k = top_k
        self.escalations = 0
        self.total = 0

    def rank(
        self,
        query: str,
        candidate_indices: np.ndarray,
        candidate_texts: List[str],
        stage1_scores: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        self.total += 1
        sorted_indices = np.argsort(-stage1_scores)
        top_k_indices = sorted_indices[: self.top_k]

        s1 = float(stage1_scores[top_k_indices[0]]) if len(top_k_indices) > 0 else 0.0
        s2 = float(stage1_scores[top_k_indices[1]]) if len(top_k_indices) > 1 else 0.0
        margin = s1 - s2

        if margin >= self.tau_margin or len(top_k_indices) <= 1:
            return sorted_indices, False

        self.escalations += 1
        pairs = [(query, candidate_texts[idx][:300]) for idx in top_k_indices]
        ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        reordered_top_k = top_k_indices[np.argsort(-ce_scores)]
        full_ranking = np.concatenate([reordered_top_k, sorted_indices[self.top_k :]])
        return full_ranking, True


# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
def run_phase4_ablation():
    t_start = time.time()
    print("=" * 80)
    print("STARTING PHASE 4: COMPONENT ABLATION & ARCHITECTURAL RE-EVALUATION")
    print("=" * 80)

    # 1. Load Datasets
    print(f"Loading public benchmark from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    print(f"Loading frozen paraphrased probes from {FROZEN_PROBES_PATH}...")
    with open(FROZEN_PROBES_PATH, "r", encoding="utf-8") as f:
        probe_data = json.load(f)

    print(f"Loading degraded OCR data from {DEGRADED_OCR_PATH}...")
    with open(DEGRADED_OCR_PATH, "r", encoding="utf-8") as f:
        degraded_ocr = json.load(f)

    records = bench_data["records"]
    dev_records = [r for r in records if r["split"] == "dev"]
    test_records = [r for r in records if r["split"] == "test"]
    dev_probes = probe_data["dev_probes"]
    test_probes = probe_data["test_probes"]

    dev_id_to_idx = {r["id"]: idx for idx, r in enumerate(dev_records)}
    test_id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}

    print(f"Loaded: Dev Items = {len(dev_records)}, Test Items = {len(test_records)}")
    print(f"Loaded: Dev Probes = {len(dev_probes)}, Test Probes = {len(test_probes)}")

    # Load Dev Parameters Cache from Phase 2
    print(f"Loading Dev tuned parameters from {DEV_PARAMS_CACHE}...")
    with open(DEV_PARAMS_CACHE, "r", encoding="utf-8") as f:
        raw_cache = json.load(f)
    dev_tuned_params = {tuple(k.split(":::")): v for k, v in raw_cache.items()}

    # Initialize Sentence Transformers & Cross Encoders
    print(f"Initializing dense bi-encoder on {DEVICE}...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print(f"Initializing CrossEncoder: ms-marco-MiniLM-L-6-v2 on {DEVICE}...")
    ce_minilm = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

    print(f"Initializing CrossEncoder: BAAI/bge-reranker-base on {DEVICE}...")
    ce_bge = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

    print(f"Loading precomputed CLIP image embeddings...")
    dev_clip_img_embs = np.load(V2_DIR / "clip_image_embeddings_dev.npy")
    test_clip_img_embs = np.load(V2_DIR / "clip_image_embeddings_test.npy")

    # Load CLIP Model for Text Encoding
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_clip_texts(texts: List[str]) -> np.ndarray:
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()

    # Pre-encode queries
    dev_q_texts_verb = [p["verbatim"] for p in dev_probes]
    test_q_texts_verb = [p["verbatim"] for p in test_probes]
    test_q_texts_light = [p["light_paraphrase"] for p in test_probes]

    dev_dense_q_verb = dense_model.encode(dev_q_texts_verb, convert_to_numpy=True, normalize_embeddings=True)
    test_dense_q_verb = dense_model.encode(test_q_texts_verb, convert_to_numpy=True, normalize_embeddings=True)
    test_dense_q_light = dense_model.encode(test_q_texts_light, convert_to_numpy=True, normalize_embeddings=True)

    test_clip_q_verb = encode_clip_texts(test_q_texts_verb)
    test_clip_q_light = encode_clip_texts(test_q_texts_light)

    # ─── RELIABILITY AUROC EVALUATION ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUB-EXPERIMENT 4.3A: OBSERVABLE RELIABILITY DISCRIMINATION (AUROC)")
    print("=" * 80)

    clean_dev_vocab = set()
    for r in dev_records:
        clean_dev_vocab.update(tokenize_words(r.get("ground_truth_text", "")))

    test_clean_texts = [r.get("ground_truth_text", "") for r in test_records]
    auroc_rows = []

    for noise_pct in [10, 20, 40]:
        rng = np.random.default_rng(42 + noise_pct)
        half_n = len(test_clean_texts) // 2
        labels = [0] * half_n + [1] * (len(test_clean_texts) - half_n)
        mixed_texts = [test_clean_texts[i] for i in range(half_n)] + [
            corrupt_text_random(test_clean_texts[i], noise_pct / 100.0, rng)
            for i in range(half_n, len(test_clean_texts))
        ]
        r_ocr = [compute_observable_reliability(t, clean_dev_vocab) for t in mixed_texts]
        # Low reliability -> high probability of corruption
        auroc_val = float(roc_auc_score(labels, 1.0 - np.array(r_ocr)))
        print(f"Synthetic Noise {noise_pct}%: Observable rho_ocr AUROC = {auroc_val:.4f}")
        auroc_rows.append({
            "Noise_Condition": f"Synthetic {noise_pct}%",
            "AUROC": round(auroc_val, 4),
            "Discriminative_Quality": "Good" if auroc_val >= 0.70 else "Poor (Heuristic Failure)",
        })

    # Real OCR degradation AUROC
    deg_items = degraded_ocr["items"]
    real_ocr_clean = [r.get("real_ocr_text", "") or r.get("ground_truth_text", "") for r in test_records]
    real_ocr_deg = [deg_items.get(r["id"], {}).get("degraded_ocr_text", "") or r.get("real_ocr_text", "") or r.get("ground_truth_text", "") for r in test_records]

    half_n = len(real_ocr_clean) // 2
    labels_real = [0] * half_n + [1] * (len(real_ocr_clean) - half_n)
    mixed_real = [real_ocr_clean[i] for i in range(half_n)] + [real_ocr_deg[i] for i in range(half_n, len(real_ocr_clean))]
    r_ocr_real = [compute_observable_reliability(t, clean_dev_vocab) for t in mixed_real]
    auroc_real = float(roc_auc_score(labels_real, 1.0 - np.array(r_ocr_real)))
    print(f"Real Degraded OCR vs Clean OCR: Observable rho_ocr AUROC = {auroc_real:.4f}")
    auroc_rows.append({
        "Noise_Condition": "Real Degraded OCR vs Clean OCR",
        "AUROC": round(auroc_real, 4),
        "Discriminative_Quality": "Good" if auroc_real >= 0.70 else "Poor (Heuristic Failure)",
    })

    auroc_csv_path = RESULTS_V2_DIR / "reliability_auroc.csv"
    with open(auroc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(auroc_rows)
    print(f"Saved AUROC results to {auroc_csv_path}.")

    # ─── ROUTER TRAINING ON DEV SPLIT ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 1A: TRAINING LEARNED QUERY ROUTER ON DEV SPLIT (N=306)")
    print("=" * 80)

    dev_texts_clean = [r.get("ground_truth_text", "") for r in dev_records]
    dev_bm25_clean = PurePythonBM25([tokenize_words(t) for t in dev_texts_clean])
    dev_dense_mat_clean = dense_model.encode(dev_texts_clean, convert_to_numpy=True, normalize_embeddings=True)

    # Determine ground-truth preference on Dev: y=1 if Dense MRR > BM25 MRR, else 0
    dev_router_labels = []
    for q_idx, p in enumerate(dev_probes):
        tgt_id = p["target_id"]
        tgt_idx = dev_id_to_idx[tgt_id]
        s_dense = np.dot(dev_dense_mat_clean, dev_dense_q_verb[q_idx])
        s_bm25 = dev_bm25_clean.get_scores(tokenize_words(dev_q_texts_verb[q_idx]))
        r_dense = int(np.where(np.argsort(-s_dense) == tgt_idx)[0][0]) + 1
        r_bm25 = int(np.where(np.argsort(-s_bm25) == tgt_idx)[0][0]) + 1
        dev_router_labels.append(1 if (1.0 / r_dense) > (1.0 / r_bm25) else 0)

    router = LearnedQueryRouter(dev_bm25_clean.idf)
    router_train_stats = router.fit(dev_q_texts_verb, dev_router_labels)
    print(f"Router Dev Train Accuracy = {router_train_stats['train_accuracy']:.4f}, Macro-F1 = {router_train_stats['train_f1']:.4f}")

    # ─── CASCADE MARGIN TAU CALIBRATION ON DEV SPLIT ───────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 1B: CALIBRATING CASCADE MARGIN TAU ON DEV SPLIT")
    print("=" * 80)

    # Dev sweep over tau for MiniLM and BGE rerankers
    alpha_dev_clean = dev_tuned_params[("verbatim", "clean")]["alpha"]

    best_tau_minilm = 0.05
    best_tau_bge = 0.05
    best_mrr_minilm = -1.0
    best_mrr_bge = -1.0

    dev_pairs_valid = [(q_idx, dev_id_to_idx[p["target_id"]]) for q_idx, p in enumerate(dev_probes) if p["target_id"] in dev_id_to_idx]

    # Precompute stage 1 hybrid scores on Dev clean
    dev_s1_hybrid_list = []
    for q_idx, _ in dev_pairs_valid:
        s_d = min_max_normalize(np.dot(dev_dense_mat_clean, dev_dense_q_verb[q_idx]))
        s_b = min_max_normalize(dev_bm25_clean.get_scores(tokenize_words(dev_q_texts_verb[q_idx])))
        dev_s1_hybrid_list.append(alpha_dev_clean * s_d + (1.0 - alpha_dev_clean) * s_b)

    for tau in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
        casc_mini = CascadeRanker(ce_minilm, tau_margin=tau, top_k=5)
        mrrs_mini = []
        for pair_idx, (q_idx, tgt_idx) in enumerate(dev_pairs_valid[:60]):  # Fast calibration subset
            ranking, _ = casc_mini.rank(dev_q_texts_verb[q_idx], np.arange(len(dev_records)), dev_texts_clean, dev_s1_hybrid_list[pair_idx])
            rank = int(np.where(ranking == tgt_idx)[0][0]) + 1
            mrrs_mini.append(1.0 / rank if rank <= 100 else 0.0)
        mean_m = float(np.mean(mrrs_mini))
        if mean_m > best_mrr_minilm:
            best_mrr_minilm = mean_m
            best_tau_minilm = tau

    for tau in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
        casc_b = CascadeRanker(ce_bge, tau_margin=tau, top_k=5)
        mrrs_b = []
        for pair_idx, (q_idx, tgt_idx) in enumerate(dev_pairs_valid[:60]):
            ranking, _ = casc_b.rank(dev_q_texts_verb[q_idx], np.arange(len(dev_records)), dev_texts_clean, dev_s1_hybrid_list[pair_idx])
            rank = int(np.where(ranking == tgt_idx)[0][0]) + 1
            mrrs_b.append(1.0 / rank if rank <= 100 else 0.0)
        mean_b = float(np.mean(mrrs_b))
        if mean_b > best_mrr_bge:
            best_mrr_bge = mean_b
            best_tau_bge = tau

    print(f"Optimal Dev Tau: MiniLM CE tau* = {best_tau_minilm:.2f}, BGE Reranker tau* = {best_tau_bge:.2f}")

    # ─── STAGE 2: TEST SPLIT ABLATION EVALUATION ───────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 2: EVALUATING ABLATION MATRIX ON HELD-OUT TEST SPLIT (N=306)")
    print("=" * 80)

    ablation_systems = [
        "Hybrid (Tuned)",
        "+ Clean Stream",
        "BM25 + Clean Stream",
        "Dense + Clean Stream",
        "+ Router",
        "+ Reliability",
        "+ Cascade (MiniLM)",
        "+ Cascade (BGE-reranker)",
        "Full RAMP-RAG (MiniLM)",
        "Full RAMP-RAG (BGE-reranker)",
        "BM25 Alone",
        "All-Stream RRF",
    ]

    conditions = ["clean", "syn_20", "real_ocr", "real_ocr_deg"]
    cond_display_map = {
        "clean": "Clean (Ground Truth)",
        "syn_20": "Syn 20% Noise (Seeds 42-44)",
        "real_ocr": "Real Apple Vision OCR",
        "real_ocr_deg": "Real Degraded OCR",
    }

    test_targets = [p["target_id"] for p in test_probes]
    test_cluster_ids = test_targets

    ablation_results = []
    ablation_raw_scores = []

    # Text corpus helper for test split
    def get_test_texts(cond: str, seed: int = 42) -> List[str]:
        if cond == "clean":
            return [r.get("ground_truth_text", "") for r in test_records]
        elif cond == "syn_20":
            rng = np.random.Generator(np.random.PCG64(seed))
            return [corrupt_text_random(r.get("ground_truth_text", ""), 0.20, rng) for r in test_records]
        elif cond == "real_ocr":
            return [r.get("real_ocr_text", "") or r.get("ground_truth_text", "") for r in test_records]
        elif cond == "real_ocr_deg":
            deg_items = degraded_ocr["items"]
            return [deg_items.get(r["id"], {}).get("degraded_ocr_text", "") or r.get("real_ocr_text", "") or r.get("ground_truth_text", "") for r in test_records]
        raise ValueError(f"Unknown condition: {cond}")

    for cond in conditions:
        print(f"\n--- Evaluating Condition: {cond_display_map[cond]} ---")
        tuned_param = dev_tuned_params[("verbatim", cond)]
        alpha_star = tuned_param["alpha"]
        gamma_hyb_star = tuned_param["gamma_hyb"]
        gamma_bm25_star = tuned_param["gamma_bm25"]

        seeds_to_run = SEEDS if cond == "syn_20" else [42]

        sys_mrrs: Dict[str, List[float]] = {s: [] for s in ablation_systems}
        sys_hits: Dict[str, List[float]] = {s: [] for s in ablation_systems}

        for seed in seeds_to_run:
            test_texts = get_test_texts(cond, seed=seed)
            test_tokens = [tokenize_words(t) for t in test_texts]
            bm25_test = PurePythonBM25(test_tokens)
            dense_mat_test = dense_model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            # Precompute document observable reliabilities
            doc_reliabilities = np.array([compute_observable_reliability(t, clean_dev_vocab) for t in test_texts], dtype=np.float32)

            casc_mini_eval = CascadeRanker(ce_minilm, tau_margin=best_tau_minilm, top_k=5)
            casc_bge_eval = CascadeRanker(ce_bge, tau_margin=best_tau_bge, top_k=5)

            for q_idx, target_id in enumerate(test_targets):
                if target_id not in test_id_to_idx:
                    continue
                tgt_idx = test_id_to_idx[target_id]
                q_text = test_q_texts_verb[q_idx]

                s_d = min_max_normalize(np.dot(dense_mat_test, test_dense_q_verb[q_idx]))
                s_b = min_max_normalize(bm25_test.get_scores(tokenize_words(q_text)))
                s_c = min_max_normalize(np.dot(test_clip_img_embs, test_clip_q_verb[q_idx]))

                # 1. Baseline: Hybrid (Tuned)
                s_hybrid = alpha_star * s_d + (1.0 - alpha_star) * s_b
                # 2. + Clean Stream
                s_hyb_clean = min_max_normalize(s_hybrid) + gamma_hyb_star * s_c
                # 3. BM25 + Clean Stream
                s_bm25_clean = s_b + gamma_bm25_star * s_c
                # 4. Dense + Clean Stream
                s_dense_clean = s_d + 0.10 * s_c
                # 5. + Router
                alpha_q = router.predict_alpha(q_text)
                s_router = alpha_q * s_d + (1.0 - alpha_q) * s_b
                # 6. + Reliability
                s_reliab = alpha_star * s_d + (1.0 - alpha_star) * (doc_reliabilities * s_b)
                # 7. BM25 Alone
                s_bm25_alone = s_b
                # 8. All-Stream RRF
                r_b = np.argsort(-s_b)
                r_d = np.argsort(-s_d)
                r_c = np.argsort(-s_c)
                s_rrf = np.zeros(len(test_records), dtype=np.float32)
                for r, idx in enumerate(r_b):
                    s_rrf[idx] += 1.0 / (60.0 + r + 1)
                for r, idx in enumerate(r_d):
                    s_rrf[idx] += 1.0 / (60.0 + r + 1)
                for r, idx in enumerate(r_c):
                    s_rrf[idx] += 1.0 / (60.0 + r + 1)

                scores_stage1 = {
                    "Hybrid (Tuned)": s_hybrid,
                    "+ Clean Stream": s_hyb_clean,
                    "BM25 + Clean Stream": s_bm25_clean,
                    "Dense + Clean Stream": s_dense_clean,
                    "+ Router": s_router,
                    "+ Reliability": s_reliab,
                    "BM25 Alone": s_bm25_alone,
                    "All-Stream RRF": s_rrf,
                }

                # Evaluate non-cascade systems
                for s_name, sc in scores_stage1.items():
                    ranking = np.argsort(-sc)
                    rank = int(np.where(ranking == tgt_idx)[0][0]) + 1
                    mrr = 1.0 / rank if rank <= 100 else 0.0
                    hit = 1.0 if rank <= 10 else 0.0
                    sys_mrrs[s_name].append(mrr)
                    sys_hits[s_name].append(hit)
                    ablation_raw_scores.append({
                        "condition": cond,
                        "seed": seed,
                        "probe_id": test_probes[q_idx]["probe_id"],
                        "target_id": target_id,
                        "system": s_name,
                        "rank": rank,
                        "mrr": mrr,
                        "hit10": hit,
                    })

                # 7. + Cascade (MiniLM)
                r_casc_mini, _ = casc_mini_eval.rank(q_text, np.arange(len(test_records)), test_texts, s_hybrid)
                rank = int(np.where(r_casc_mini == tgt_idx)[0][0]) + 1
                mrr = 1.0 / rank if rank <= 100 else 0.0
                hit = 1.0 if rank <= 10 else 0.0
                sys_mrrs["+ Cascade (MiniLM)"].append(mrr)
                sys_hits["+ Cascade (MiniLM)"].append(hit)

                # 8. + Cascade (BGE-reranker)
                r_casc_bge, _ = casc_bge_eval.rank(q_text, np.arange(len(test_records)), test_texts, s_hybrid)
                rank = int(np.where(r_casc_bge == tgt_idx)[0][0]) + 1
                mrr = 1.0 / rank if rank <= 100 else 0.0
                hit = 1.0 if rank <= 10 else 0.0
                sys_mrrs["+ Cascade (BGE-reranker)"].append(mrr)
                sys_hits["+ Cascade (BGE-reranker)"].append(hit)

                # 9. Full RAMP-RAG (MiniLM)
                s_ramp_stage1 = alpha_q * s_d + (1.0 - alpha_q) * (doc_reliabilities * s_b) + 0.15 * (1.0 - doc_reliabilities) * s_c
                r_ramp_mini, _ = casc_mini_eval.rank(q_text, np.arange(len(test_records)), test_texts, s_ramp_stage1)
                rank = int(np.where(r_ramp_mini == tgt_idx)[0][0]) + 1
                mrr = 1.0 / rank if rank <= 100 else 0.0
                hit = 1.0 if rank <= 10 else 0.0
                sys_mrrs["Full RAMP-RAG (MiniLM)"].append(mrr)
                sys_hits["Full RAMP-RAG (MiniLM)"].append(hit)

                # 10. Full RAMP-RAG (BGE-reranker)
                r_ramp_bge, _ = casc_bge_eval.rank(q_text, np.arange(len(test_records)), test_texts, s_ramp_stage1)
                rank = int(np.where(r_ramp_bge == tgt_idx)[0][0]) + 1
                mrr = 1.0 / rank if rank <= 100 else 0.0
                hit = 1.0 if rank <= 10 else 0.0
                sys_mrrs["Full RAMP-RAG (BGE-reranker)"].append(mrr)
                sys_hits["Full RAMP-RAG (BGE-reranker)"].append(hit)

        # Statistical analysis for this condition
        base_mrrs = sys_mrrs["Hybrid (Tuned)"]
        comp_systems = [s for s in ablation_systems if s != "Hybrid (Tuned)"]

        # Permutation tests vs Baseline
        p_val_map = {}
        for s in comp_systems:
            p_val = eval_lib.paired_permutation_test(
                sys_mrrs[s], base_mrrs, test_cluster_ids * len(seeds_to_run), n_permutations=10000, seed=42
            )
            p_val_map[s] = p_val

        adj_p_map = eval_lib.holm_bonferroni(p_val_map)

        for s in ablation_systems:
            m_mrrs = sys_mrrs[s]
            m_hits = sys_hits[s]
            mean_m = float(np.mean(m_mrrs))
            ci_m = eval_lib.cluster_bootstrap_ci(m_mrrs, test_cluster_ids * len(seeds_to_run), n_boot=2000, ci=0.95, seed=42)
            mean_h = float(np.mean(m_hits))
            ci_h = eval_lib.cluster_bootstrap_ci(m_hits, test_cluster_ids * len(seeds_to_run), n_boot=2000, ci=0.95, seed=42)

            if s != "Hybrid (Tuned)":
                diff_ci = eval_lib.cluster_bootstrap_diff_ci(m_mrrs, base_mrrs, test_cluster_ids * len(seeds_to_run), n_boot=2000, ci=0.95, seed=42)
                p_raw = p_val_map[s]
                p_holm = adj_p_map.get(s, 1.0)
            else:
                diff_ci = (0.0, 0.0)
                p_raw = 1.0
                p_holm = 1.0

            ablation_results.append({
                "Condition": cond,
                "Configuration": s,
                "MRR": round(mean_m, 4),
                "MRR_CI_Lower": round(ci_m[0], 4),
                "MRR_CI_Upper": round(ci_m[1], 4),
                "Hit10": round(mean_h, 4),
                "Hit10_CI_Lower": round(ci_h[0], 4),
                "Hit10_CI_Upper": round(ci_h[1], 4),
                "Diff_MRR_vs_Base": round(mean_m - float(np.mean(base_mrrs)), 4),
                "Diff_CI_Lower": round(diff_ci[0], 4),
                "Diff_CI_Upper": round(diff_ci[1], 4),
                "Perm_P_Raw": round(p_raw, 5),
                "Holm_P_Adj": round(p_holm, 5),
                "Is_Significant": p_holm < 0.05,
            })
            print(f"  {s:<30}: MRR={mean_m:.4f} [{ci_m[0]:.3f}, {ci_m[1]:.3f}] | Diff={mean_m - float(np.mean(base_mrrs)):+.4f} | p_holm={p_holm:.4f}")

    # Save Ablation Results CSV
    abl_csv_path = RESULTS_V2_DIR / "ablation_results.csv"
    with open(abl_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)
    print(f"\nSaved ablation aggregate results to {abl_csv_path}.")

    # ─── DECISION RULES EVALUATION (H1 and H2) ─────────────────────────────────
    print("\n" + "=" * 80)
    print("HYPOTHESIS DECISION RULES VERDICT (PRE-REGISTRATION ADDENDUM)")
    print("=" * 80)

    # H1: Reliability-Weighted Fusion
    # Pre-registration rule: Reject H1 if BM25 alone or Tuned Hybrid achieves higher MRR
    # or if the difference CI includes zero.
    h1_records = [r for r in ablation_results if r["Configuration"] == "+ Reliability" and r["Condition"] == "syn_20"][0]
    hybrid_records = [r for r in ablation_results if r["Configuration"] == "Hybrid (Tuned)" and r["Condition"] == "syn_20"][0]
    bm25_records = [r for r in ablation_results if r["Configuration"] == "BM25 Alone" and r["Condition"] == "syn_20"][0]

    h1_mrr = h1_records["MRR"]
    hybrid_mrr = hybrid_records["MRR"]
    bm25_mrr = bm25_records["MRR"]
    diff_ci_h1 = (h1_records["Diff_CI_Lower"], h1_records["Diff_CI_Upper"])

    h1_pass = (h1_mrr > hybrid_mrr) and (h1_mrr > bm25_mrr) and (diff_ci_h1[0] > 0.0)
    h1_verdict = "PASS" if h1_pass else "FAIL (REJECTED)"

    print(f"H1 (Reliability Weighting under 20% Noise):")
    print(f"  + Reliability MRR = {h1_mrr:.4f}")
    print(f"  Hybrid (Tuned) MRR = {hybrid_mrr:.4f}")
    print(f"  BM25 Alone MRR     = {bm25_mrr:.4f}")
    print(f"  Difference CI vs Hybrid = [{diff_ci_h1[0]:+.4f}, {diff_ci_h1[1]:+.4f}], p_holm = {h1_records['Holm_P_Adj']:.4f}")
    print(f"  Decision Rule Verdict: >>> {h1_verdict} <<<")

    # H2: Learned Query Routing
    # Pre-registration rule: Reject H2 if All-Stream RRF >= best router MRR.
    h2_records = [r for r in ablation_results if r["Configuration"] == "+ Router" and r["Condition"] == "syn_20"][0]
    rrf_records = [r for r in ablation_results if r["Configuration"] == "All-Stream RRF" and r["Condition"] == "syn_20"][0]

    router_mrr = h2_records["MRR"]
    rrf_mrr = rrf_records["MRR"]
    h2_pass = router_mrr > rrf_mrr
    h2_verdict = "PASS" if h2_pass else "FAIL (REJECTED)"

    print(f"\nH2 (Learned Query Router vs All-Stream RRF under 20% Noise):")
    print(f"  + Router MRR       = {router_mrr:.4f}")
    print(f"  All-Stream RRF MRR = {rrf_mrr:.4f}")
    print(f"  Decision Rule Verdict: >>> {h2_verdict} <<<")

    # Save LaTeX table fragment
    generate_ablation_latex_fragment(ablation_results)

    t_end = time.time()
    print(f"\nPhase 4 completed end-to-end in {t_end - t_start:.2f}s.")


def generate_ablation_latex_fragment(ablation_results: List[Dict[str, Any]]):
    latex_path = LATEX_DIR / "table_ablation_v2.tex"
    lines = []
    lines.append("% Auto-generated by run_ablation_v2.py (DO NOT HAND EDIT)")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Architectural Component Ablation on Public Held-Out Test Split ($N=306$ Probes, 813 Distractor Pool). 95\\% cluster bootstrap CIs in brackets. $p_{\\text{Holm}}$ computed via paired sign-flip permutation tests ($N_{\\text{perm}}=10,000$) against Tuned Hybrid baseline.}")
    lines.append("\\label{tab:ablation_v2}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Configuration & Clean MRR & Syn 20\\% MRR [95\\% CI] & Real-OCR MRR & Real-OCR Deg MRR \\\\")
    lines.append("\\midrule")

    display_configs = [
        "Hybrid (Tuned)",
        "+ Clean Stream",
        "BM25 + Clean Stream",
        "Dense + Clean Stream",
        "+ Router",
        "+ Reliability",
        "+ Cascade (MiniLM)",
        "+ Cascade (BGE-reranker)",
        "Full RAMP-RAG (MiniLM)",
        "Full RAMP-RAG (BGE-reranker)",
        "BM25 Alone",
        "All-Stream RRF",
    ]

    for cfg in display_configs:
        c_clean = [r for r in ablation_results if r["Configuration"] == cfg and r["Condition"] == "clean"][0]
        c_syn20 = [r for r in ablation_results if r["Configuration"] == cfg and r["Condition"] == "syn_20"][0]
        c_real = [r for r in ablation_results if r["Configuration"] == cfg and r["Condition"] == "real_ocr"][0]
        c_deg = [r for r in ablation_results if r["Configuration"] == cfg and r["Condition"] == "real_ocr_deg"][0]

        star = "$^*$" if c_syn20["Is_Significant"] else ""
        ci_str = f"[{c_syn20['MRR_CI_Lower']:.3f}, {c_syn20['MRR_CI_Upper']:.3f}]"
        bold_prefix = "\\textbf{" if cfg.startswith("Full RAMP-RAG") else ""
        bold_suffix = "}" if cfg.startswith("Full RAMP-RAG") else ""

        line_str = f"{bold_prefix}{cfg}{bold_suffix} & {c_clean['MRR']:.3f} & {c_syn20['MRR']:.3f}{star} {ci_str} & {c_real['MRR']:.3f} & {c_deg['MRR']:.3f} \\\\"
        lines.append(line_str)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved ready-to-paste LaTeX table fragment to {latex_path}.")


if __name__ == "__main__":
    run_phase4_ablation()
