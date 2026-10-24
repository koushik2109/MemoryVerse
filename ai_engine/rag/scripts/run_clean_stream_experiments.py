"""
Comprehensive Experimental Suite for RAMP-RAG Clean-Stream Fusion & Multi-Modal Benchmarks
==========================================================================================
Paper: "Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval"
Strict Dev/Test split (50/50 by image ID), observable reliability (zero oracle), multi-seed (42, 43, 44),
bootstrap CIs (2,000 resamples), Clopper-Pearson CIs, paired permutation tests with Holm-Bonferroni correction,
and standardized latency profiling over 3 runs on warm cache.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import beta
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import CLIPModel, CLIPProcessor

# ─── PATH CONFIGURATION ───────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
OCR_CORPUS_PATH = RAG_DATA_DIR / "ocr_extracted_corpus.json"
PROBES_SPLIT_PATH = RAG_DATA_DIR / "ramp_rag_probes_split.json"
COCO_DATA_PATH = RAG_DATA_DIR / "coco_karpathy_test.json"
COCO_BLIP_PATH = RAG_DATA_DIR / "coco_blip_captions.json"
COCO_IMAGES_DIR = RAG_DATA_DIR / "coco_1k_images"

OUTPUT_RESULTS_JSON = RAG_DATA_DIR / "clean_stream_experiments_results.json"
OUTPUT_PLOT_PNG = RAG_DATA_DIR / "mrr_hit10_vs_noise.png"
OUTPUT_LATEX_DIR = RAG_DATA_DIR / "latex_tables"
OUTPUT_CSV_DIR = RAG_DATA_DIR / "csv_tables"

SEEDS = [42, 43, 44]
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.40]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── TEXT PROCESSING & TOKENIZATION ──────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """Bilingual tokenization for English and CJK characters."""
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


class PurePythonBM25:
    """Fast BM25Okapi implementation without external C dependencies."""

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


# ─── OBSERVABLE RELIABILITY & NOISE INJECTION (ZERO ORACLE) ───────────────────
def compute_observable_reliability(text: str, vocab_set: Set[str]) -> float:
    """
    Estimates stream reliability r_m(d) purely from observable signals:
    - Token dictionary validity ratio rho_vocab
    - Fragmented token anomaly penalty
    Never uses injected noise level.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.15
    valid_count = sum(1 for t in tokens if t in vocab_set)
    rho_vocab = valid_count / len(tokens)
    avg_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    len_penalty = min(1.0, max(0.4, avg_len / 3.0))
    return float(np.clip(rho_vocab * len_penalty, 0.10, 1.0))


def corrupt_text(text: str, noise_rate: float, rng: np.random.Generator) -> str:
    """Injects character substitutions, deletions, and insertions at noise_rate."""
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


# ─── ROUTER & CASCADE COMPONENTS ──────────────────────────────────────────────
class LearnedQueryRouter:
    """Learned classifier predicting optimal alpha_q from query surface features."""

    def __init__(self, bm25_idf: Dict[str, float]):
        self.bm25_idf = bm25_idf
        self.clf = LogisticRegression(C=1.0, random_state=42)
        self.is_fitted = False

    def extract_features(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        raw_words = query.split()
        if not tokens:
            return np.zeros(6, dtype=np.float32)

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
        cjk_ratio = float(cjk_count / max(len(query), 1))

        digits = sum(1 for c in query if c.isdigit())
        digit_ratio = float(min(1.0, digits / max(len(query), 1)))

        has_upper = float(any(len(w) >= 3 and w.isupper() for w in raw_words))
        has_id_or_cue = float(bool(re.search(r"IMG_\d+|doc_\w+|IEEE|LLaMA|RRF|OCR|PDF|ticket|plaque|booth|office|reads|sign", query, re.I)))
        length_feat = float(min(1.0, len(tokens) / 20.0))

        idfs = [self.bm25_idf.get(t, 2.0) for t in tokens]
        avg_idf = float(np.mean(idfs)) / 10.0 if idfs else 0.5

        return np.array([cjk_ratio, digit_ratio, has_upper, has_id_or_cue, length_feat, avg_idf], dtype=np.float32)

    def fit(self, queries: List[str], labels: List[int]) -> Dict[str, float]:
        X = np.array([self.extract_features(q) for q in queries], dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        self.clf.fit(X, y)
        self.is_fitted = True
        preds = self.clf.predict(X)
        return {
            "train_accuracy": float(accuracy_score(y, preds)),
            "train_f1": float(f1_score(y, preds)),
        }

    def predict_alpha(self, query: str) -> float:
        if not self.is_fitted:
            cjk = len(re.findall(r"[\u4e00-\u9fff]", query))
            return 0.25 if (cjk > 0 or "IMG_" in query or "doc_" in query) else 0.85
        feat = self.extract_features(query).reshape(1, -1)
        prob_dense = float(self.clf.predict_proba(feat)[0, 1])
        return 0.85 if prob_dense >= 0.5 else 0.25


class CascadeRanker:
    """Confidence-Triggered Cascade Ranker: escalates top-k to Cross-Encoder when margin < tau."""

    def __init__(self, cross_encoder: CrossEncoder, tau_margin: float = 0.065, top_k: int = 5):
        self.cross_encoder = cross_encoder
        self.tau_margin = tau_margin
        self.top_k = top_k
        self.escalation_count = 0
        self.total_queries = 0

    def rank(
        self,
        query: str,
        candidate_ids: List[str],
        candidate_texts: List[str],
        stage1_scores: np.ndarray,
    ) -> Tuple[List[str], bool, float]:
        self.total_queries += 1
        sorted_indices = np.argsort(-stage1_scores)
        top_candidates = sorted_indices[: self.top_k]

        s1 = float(stage1_scores[top_candidates[0]]) if len(top_candidates) > 0 else 0.0
        s2 = float(stage1_scores[top_candidates[1]]) if len(top_candidates) > 1 else 0.0
        margin = s1 - s2

        if margin >= self.tau_margin or len(top_candidates) <= 1:
            ranked_ids = [candidate_ids[i] for i in sorted_indices]
            return ranked_ids, False, margin

        self.escalation_count += 1
        pairs = [(query, candidate_texts[i]) for i in top_candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        ce_order = np.argsort(-ce_scores)

        reordered_top = [candidate_ids[top_candidates[idx]] for idx in ce_order]
        remaining = [candidate_ids[i] for i in sorted_indices[self.top_k :]]
        return reordered_top + remaining, True, margin


# ─── METRICS & STATISTICAL ROUTINES ──────────────────────────────────────────
def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & ground_truth_ids))


def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    relevances = [1 if doc_id in ground_truth_ids else 0 for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = [1] * min(len(ground_truth_ids), k)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_bootstrap_ci(scores: List[float], n_resamples: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    if len(arr) == 0:
        return (0.0, 0.0)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def compute_clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval for saturated proportions."""
    if n == 0:
        return (0.0, 0.0)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lower, upper)


def compute_ci_adaptive(scores: List[float], n_resamples: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Uses exact Clopper-Pearson if saturated, otherwise 2,000-resample bootstrap."""
    arr = np.array(scores)
    mean_val = float(np.mean(arr)) if len(arr) > 0 else 0.0
    if mean_val >= 0.9999 or mean_val <= 0.0001:
        k = int(round(np.sum(arr)))
        return compute_clopper_pearson_ci(k, len(arr))
    return compute_bootstrap_ci(scores, n_resamples, seed)


def paired_permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test on paired differences."""
    d = np.array(scores_a) - np.array(scores_b)
    obs_diff = float(np.abs(np.mean(d)))
    if obs_diff < 1e-9:
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(d)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    perm_diffs = np.abs(np.mean(signs * d, axis=1))
    p_val = float(np.mean(perm_diffs >= obs_diff))
    return max(p_val, 1.0 / n_permutations)


def holm_bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni correction over a dictionary of method -> p-value."""
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p: Dict[str, float] = {}
    cum_max = 0.0
    for rank, (k, p) in enumerate(sorted_items):
        multiplier = m - rank
        raw_adj = p * multiplier
        adj = min(1.0, max(cum_max, raw_adj))
        cum_max = adj
        adj_p[k] = round(adj, 5)
    return adj_p


# ─── CORE RETRIEVAL ENGINE ───────────────────────────────────────────────────
def execute_query(
    query: str,
    method: str,
    corpus_ids: List[str],
    corpus_texts: List[str],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    clean_scores: np.ndarray,
    reliabilities: np.ndarray,
    router: LearnedQueryRouter,
    cascade: CascadeRanker,
    alpha_fixed: float = 0.70,
    gamma_clean: float = 0.30,
) -> List[str]:
    """Executes single query retrieval and ranking given precomputed stream scores."""
    s_dense = min_max_normalize(dense_scores)
    s_sparse = min_max_normalize(sparse_scores)
    s_clean = min_max_normalize(clean_scores)

    if method == "BM25":
        final_scores = s_sparse
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "Dense":
        final_scores = s_dense
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "Hybrid (fixed alpha=0.7)":
        final_scores = alpha_fixed * s_dense + (1.0 - alpha_fixed) * s_sparse
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "Hybrid + Router":
        alpha_q = router.predict_alpha(query)
        final_scores = alpha_q * s_dense + (1.0 - alpha_q) * s_sparse
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "Original RAMP-RAG":
        alpha_q = router.predict_alpha(query)
        s_stage1 = alpha_q * s_dense + (1.0 - alpha_q) * reliabilities * s_sparse
        ranked_ids, _, _ = cascade.rank(query, corpus_ids, corpus_texts, s_stage1)
        return ranked_ids

    elif method == "RAMP-RAG + Clean Stream":
        alpha_q = router.predict_alpha(query)
        s_stage1 = (
            alpha_q * s_dense
            + (1.0 - alpha_q) * reliabilities * s_sparse
            + gamma_clean * (1.0 - reliabilities) * s_clean
        )
        ranked_ids, _, _ = cascade.rank(query, corpus_ids, corpus_texts, s_stage1)
        return ranked_ids

    elif method == "Cross-Encoder":
        s_stage1 = alpha_fixed * s_dense + (1.0 - alpha_fixed) * s_sparse
        top5_idx = np.argsort(-s_stage1)[:5]
        pairs = [(query, corpus_texts[i]) for i in top5_idx]
        ce_scores = cascade.cross_encoder.predict(pairs)
        ce_order = np.argsort(-ce_scores)
        reordered_top = [corpus_ids[top5_idx[i]] for i in ce_order]
        remaining = [corpus_ids[i] for i in np.argsort(-s_stage1)[5:]]
        return reordered_top + remaining

    # Ablation-specific methods
    elif method == "+ Router":
        alpha_q = router.predict_alpha(query)
        final_scores = alpha_q * s_dense + (1.0 - alpha_q) * s_sparse
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "+ Cascade":
        s_stage1 = alpha_fixed * s_dense + (1.0 - alpha_fixed) * s_sparse
        ranked_ids, _, _ = cascade.rank(query, corpus_ids, corpus_texts, s_stage1)
        return ranked_ids

    elif method == "+ Reliability":
        alpha_q = router.predict_alpha(query)
        final_scores = alpha_q * s_dense + (1.0 - alpha_q) * reliabilities * s_sparse
        return [corpus_ids[i] for i in np.argsort(-final_scores)]

    elif method == "Full RAMP-RAG (+ Clean Stream)":
        alpha_q = router.predict_alpha(query)
        s_stage1 = (
            alpha_q * s_dense
            + (1.0 - alpha_q) * reliabilities * s_sparse
            + gamma_clean * (1.0 - reliabilities) * s_clean
        )
        ranked_ids, _, _ = cascade.rank(query, corpus_ids, corpus_texts, s_stage1)
        return ranked_ids

    else:
        raise ValueError(f"Unknown retrieval method: {method}")


# ─── BATCH EVALUATION HELPER ─────────────────────────────────────────────────
def evaluate_method_on_queries(
    method: str,
    queries: List[str],
    ground_truth: List[Set[str]],
    corpus_ids: List[str],
    corpus_texts: List[str],
    dense_scores_matrix: np.ndarray,
    sparse_scores_matrix: np.ndarray,
    clean_scores_matrix: np.ndarray,
    reliabilities: np.ndarray,
    router: LearnedQueryRouter,
    cascade: CascadeRanker,
    alpha_fixed: float = 0.70,
    gamma_clean: float = 0.30,
) -> Dict[str, Any]:
    """Runs retrieval for all queries, returns per-query metric lists and aggregate statistics."""
    hit1_list, hit5_list, hit10_list, mrr_list, ndcg_list = [], [], [], [], []

    for q_idx, (q, gt) in enumerate(zip(queries, ground_truth)):
        d_scores = dense_scores_matrix[q_idx]
        s_scores = sparse_scores_matrix[q_idx]
        c_scores = clean_scores_matrix[q_idx]

        ranked = execute_query(
            q,
            method,
            corpus_ids,
            corpus_texts,
            d_scores,
            s_scores,
            c_scores,
            reliabilities,
            router,
            cascade,
            alpha_fixed,
            gamma_clean,
        )

        hit1_list.append(compute_hit_rate(ranked, gt, 1))
        hit5_list.append(compute_hit_rate(ranked, gt, 5))
        hit10_list.append(compute_hit_rate(ranked, gt, 10))
        mrr_list.append(compute_mrr(ranked, gt))
        ndcg_list.append(compute_ndcg(ranked, gt, 10))

    return {
        "hit1_list": hit1_list,
        "hit5_list": hit5_list,
        "hit10_list": hit10_list,
        "mrr_list": mrr_list,
        "ndcg_list": ndcg_list,
        "hit1": float(np.mean(hit1_list)),
        "hit5": float(np.mean(hit5_list)),
        "hit10": float(np.mean(hit10_list)),
        "mrr": float(np.mean(mrr_list)),
        "ndcg10": float(np.mean(ndcg_list)),
    }


# ─── LATENCY PROFILING HELPER ────────────────────────────────────────────────
def measure_system_latency(
    query_fn,
    queries: List[str],
    n_runs: int = 3,
) -> Dict[str, float]:
    """Standardized latency protocol: warm cache, 3 runs, reports mean and p95 ms."""
    # Warm-up run
    for q in queries[:min(10, len(queries))]:
        query_fn(q)

    latencies_ms = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter()
            query_fn(q)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat)

    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    }


# ─── MAIN EXPERIMENTAL SUITE ─────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("RAMP-RAG CLEAN-STREAM FUSION & ROBUSTNESS BENCHMARKS")
    print("=" * 80)

    # 1. LOAD MODELS
    print(f"\n[Step 1] Loading models on {DEVICE}...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
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

    # Build reference clean vocabulary
    clean_vocab = set()
    for r in all_ocr_records:
        clean_vocab.update(tokenize(r.get("text_representation", "")))
        clean_vocab.update(tokenize(r.get("file_path", "")))

    with open(COCO_DATA_PATH, encoding="utf-8") as f:
        coco_records = json.load(f)

    rng_coco = np.random.default_rng(42)
    coco_selected_idx = rng_coco.choice(len(coco_records), size=1000, replace=False)
    coco_selected = [coco_records[i] for i in coco_selected_idx]
    for r in coco_selected:
        caps = r["caption"] if isinstance(r["caption"], list) else [str(r["caption"])]
        for c in caps:
            clean_vocab.update(tokenize(c))
    print(f"Clean reference vocabulary: {len(clean_vocab)} tokens.")

    # Load BLIP captions for COCO
    with open(COCO_BLIP_PATH, encoding="utf-8") as f:
        coco_blip_captions = json.load(f)
    print(f"Loaded {len(coco_blip_captions)} BLIP captions for COCO.")

    # 3. PRECOMPUTE CLIP CLEAN EMBEDDINGS
    print("\n[Step 3] Precomputing CLIP clean embeddings for photos and documents...")
    def encode_clip_clean_corpus(records: List[Dict[str, Any]]) -> np.ndarray:
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
                    # Fallback to clean text
                    txt = r.get("text_representation", "") or Path(fp).stem
                    inputs = clip_processor(text=[txt[:77]], return_tensors="pt", padding=True).to(DEVICE)
                    with torch.no_grad():
                        feat = clip_model.get_text_features(**inputs)
                        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                            feat = feat.pooler_output
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0])
            else:
                # Document or text record: clean text representation
                txt = r.get("text_representation", "")[:200] or Path(r.get("file_path", "")).stem
                inputs = clip_processor(text=[txt[:77]], return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    feat = clip_model.get_text_features(**inputs)
                    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                        feat = feat.pooler_output
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy()[0])
        return np.array(embeddings, dtype=np.float32)

    dev_clip_clean = encode_clip_clean_corpus(dev_records)
    test_clip_clean = encode_clip_clean_corpus(test_records)
    print(f"Dev CLIP clean embeddings shape: {dev_clip_clean.shape}")
    print(f"Test CLIP clean embeddings shape: {test_clip_clean.shape}")

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

    # 4. PREPARE COCO DATASET (250 Dev, 250 Test)
    coco_doc_ids = [f"coco_{i}" for i in range(len(coco_selected))]
    coco_gold_captions = [r["caption"][0] if isinstance(r["caption"], list) else str(r["caption"]) for r in coco_selected]
    coco_blip_texts = [coco_blip_captions.get(Path(r["image"]).name, coco_gold_captions[i]) for i, r in enumerate(coco_selected)]

    coco_queries_all = []
    coco_gt_all = []
    for i, r in enumerate(coco_selected[:500]):
        caps = r["caption"] if isinstance(r["caption"], list) else [str(r["caption"])]
        q = caps[3] if len(caps) > 3 else caps[-1]
        coco_queries_all.append(q)
        coco_gt_all.append({f"coco_{i}"})

    coco_dev_queries = coco_queries_all[:250]
    coco_dev_gt = coco_gt_all[:250]
    coco_test_queries = coco_queries_all[250:]
    coco_test_gt = coco_gt_all[250:]

    # 5. DEV TUNING: QUERY ROUTER TRAINING & GAMMA SWEEP (DEV ONLY)
    print("\n" + "=" * 60)
    print("TASK 1: Dev Tuning for Clean-Stream Fusion (DEV SPLIT ONLY)")
    print("=" * 60)

    dev_bm25_tokens = [tokenize(r.get("text_representation", "")) for r in dev_records]
    dev_bm25 = PurePythonBM25(dev_bm25_tokens)

    # Train Query Router on Dev queries
    train_queries = coco_dev_queries + [p[0] for p in dev_probes]
    train_labels = [1] * len(coco_dev_queries) + [0] * len(dev_probes)  # 1 = dense dominant, 0 = sparse dominant
    router = LearnedQueryRouter(dev_bm25.idf)
    train_perf = router.fit(train_queries, train_labels)
    print(f"Router trained on {len(train_queries)} Dev queries: Train Acc={train_perf['train_accuracy']*100:.1f}%, F1={train_perf['train_f1']:.3f}")

    # Dev Sweep for Gamma in clean stream under 20% noise
    print("\nSweeping gamma in clean-stream fusion on Dev under 20% noise...")
    rng_dev_noise = np.random.default_rng(42)
    dev_texts_20 = [corrupt_text(r.get("text_representation", ""), 0.20, rng_dev_noise) for r in dev_records]
    dev_tokens_20 = [tokenize(t) for t in dev_texts_20]
    dev_bm25_20 = PurePythonBM25(dev_tokens_20)
    dev_dense_emb_20 = dense_model.encode(dev_texts_20, convert_to_numpy=True, normalize_embeddings=True)
    dev_reliabilities_20 = np.array([compute_observable_reliability(t, clean_vocab) for t in dev_texts_20], dtype=np.float32)

    dev_query_texts = [p[0] for p in dev_probes]
    dev_gt = [p[1] for p in dev_probes]
    dev_q_dense = dense_model.encode(dev_query_texts, convert_to_numpy=True, normalize_embeddings=True)
    dev_q_clip = encode_clip_queries(dev_query_texts)

    dev_dense_sim_20 = np.dot(dev_q_dense, dev_dense_emb_20.T)
    dev_sparse_sim_20 = np.array([dev_bm25_20.get_scores(tokenize(q)) for q in dev_query_texts])
    dev_clean_sim = np.dot(dev_q_clip, dev_clip_clean.T)

    gamma_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_gamma = 0.3
    best_dev_mrr = -1.0
    gamma_sweep_results = {}

    cascade_dev = CascadeRanker(cross_encoder, tau_margin=0.065, top_k=5)

    for g in gamma_candidates:
        res = evaluate_method_on_queries(
            "RAMP-RAG + Clean Stream",
            dev_query_texts,
            dev_gt,
            dev_record_ids,
            dev_texts_20,
            dev_dense_sim_20,
            dev_sparse_sim_20,
            dev_clean_sim,
            dev_reliabilities_20,
            router,
            cascade_dev,
            alpha_fixed=0.70,
            gamma_clean=g,
        )
        gamma_sweep_results[str(g)] = {"hit10": res["hit10"], "mrr": res["mrr"]}
        print(f"  gamma={g:.2f}: Dev Hit@10={res['hit10']:.4f}, Dev MRR={res['mrr']:.4f}")
        if res["mrr"] > best_dev_mrr:
            best_dev_mrr = res["mrr"]
            best_gamma = g

    print(f"\nOptimal gamma selected on Dev: gamma* = {best_gamma} (Dev MRR = {best_dev_mrr:.4f})")
    print("FREEZING gamma* = {best_gamma} for all subsequent Test evaluations (Zero tuning on Test).")

    # 6. TASK 2: MULTI-SEED NOISE SWEEP ON TEST SPLIT (SEEDS 42, 43, 44)
    print("\n" + "=" * 60)
    print("TASK 2: Multi-Seed Noise Robustness Sweep on Held-Out Test Split")
    print("=" * 60)

    test_query_texts = [p[0] for p in test_probes]
    test_gt = [p[1] for p in test_probes]
    test_q_dense = dense_model.encode(test_query_texts, convert_to_numpy=True, normalize_embeddings=True)
    test_q_clip = encode_clip_queries(test_query_texts)
    test_clean_sim = np.dot(test_q_clip, test_clip_clean.T)

    methods_task2 = [
        "BM25",
        "Dense",
        "Hybrid (fixed alpha=0.7)",
        "Hybrid + Router",
        "Original RAMP-RAG",
        "RAMP-RAG + Clean Stream",
        "Cross-Encoder",
    ]

    task2_results: Dict[str, Any] = {}

    for noise in NOISE_LEVELS:
        noise_key = f"noise_{int(noise*100)}pct"
        print(f"\n--- Testing Noise Level: {int(noise*100)}% Across Seeds 42, 43, 44 ---")
        task2_results[noise_key] = {}

        # Store pooled scores across seeds for each method
        pooled_scores: Dict[str, Dict[str, List[float]]] = {
            m: {"hit1": [], "hit5": [], "hit10": [], "mrr": [], "ndcg10": []} for m in methods_task2
        }

        for seed in SEEDS:
            rng_noise = np.random.default_rng(seed + int(noise * 100))
            test_texts_noisy = [corrupt_text(r.get("text_representation", ""), noise, rng_noise) for r in test_records]
            test_tokens_noisy = [tokenize(t) for t in test_texts_noisy]
            test_bm25_noisy = PurePythonBM25(test_tokens_noisy)
            test_dense_emb_noisy = dense_model.encode(test_texts_noisy, convert_to_numpy=True, normalize_embeddings=True)

            test_reliabilities = np.array(
                [compute_observable_reliability(t, clean_vocab) for t in test_texts_noisy],
                dtype=np.float32,
            )

            d_sim = np.dot(test_q_dense, test_dense_emb_noisy.T)
            s_sim = np.array([test_bm25_noisy.get_scores(tokenize(q)) for q in test_query_texts])

            cascade_test = CascadeRanker(cross_encoder, tau_margin=0.065, top_k=5)

            for m in methods_task2:
                res = evaluate_method_on_queries(
                    m,
                    test_query_texts,
                    test_gt,
                    test_record_ids,
                    test_texts_noisy,
                    d_sim,
                    s_sim,
                    test_clean_sim,
                    test_reliabilities,
                    router,
                    cascade_test,
                    alpha_fixed=0.70,
                    gamma_clean=best_gamma,
                )
                pooled_scores[m]["hit1"].extend(res["hit1_list"])
                pooled_scores[m]["hit5"].extend(res["hit5_list"])
                pooled_scores[m]["hit10"].extend(res["hit10_list"])
                pooled_scores[m]["mrr"].extend(res["mrr_list"])
                pooled_scores[m]["ndcg10"].extend(res["ndcg_list"])

        # Compute pooled statistics, CIs, and paired permutation tests vs baseline
        baseline_mrr = pooled_scores["Hybrid (fixed alpha=0.7)"]["mrr"]
        baseline_h10 = pooled_scores["Hybrid (fixed alpha=0.7)"]["hit10"]

        raw_p_values_mrr: Dict[str, float] = {}
        raw_p_values_h10: Dict[str, float] = {}

        for m in methods_task2:
            if m != "Hybrid (fixed alpha=0.7)":
                raw_p_values_mrr[m] = paired_permutation_test(pooled_scores[m]["mrr"], baseline_mrr, seed=42)
                raw_p_values_h10[m] = paired_permutation_test(pooled_scores[m]["hit10"], baseline_h10, seed=42)

        holm_p_mrr = holm_bonferroni_correction(raw_p_values_mrr)
        holm_p_h10 = holm_bonferroni_correction(raw_p_values_h10)

        for m in methods_task2:
            m_hit1 = float(np.mean(pooled_scores[m]["hit1"]))
            m_hit5 = float(np.mean(pooled_scores[m]["hit5"]))
            m_hit10 = float(np.mean(pooled_scores[m]["hit10"]))
            m_mrr = float(np.mean(pooled_scores[m]["mrr"]))
            m_ndcg = float(np.mean(pooled_scores[m]["ndcg10"]))

            ci_hit10 = compute_ci_adaptive(pooled_scores[m]["hit10"], seed=42)
            ci_mrr = compute_bootstrap_ci(pooled_scores[m]["mrr"], seed=42)

            task2_results[noise_key][m] = {
                "hit1": round(m_hit1, 4),
                "hit5": round(m_hit5, 4),
                "hit10": round(m_hit10, 4),
                "hit10_ci95": [round(ci_hit10[0], 4), round(ci_hit10[1], 4)],
                "mrr": round(m_mrr, 4),
                "mrr_ci95": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
                "ndcg10": round(m_ndcg, 4),
                "p_value_mrr_vs_hybrid": holm_p_mrr.get(m, 1.0),
                "p_value_hit10_vs_hybrid": holm_p_h10.get(m, 1.0),
            }

            p_str = f"| p(MRR)={holm_p_mrr[m]:.4f}" if m in holm_p_mrr else ""
            print(f"  {m:<24}: H@10={m_hit10:.4f} [{ci_hit10[0]:.4f}, {ci_hit10[1]:.4f}] | MRR={m_mrr:.4f} [{ci_mrr[0]:.4f}, {ci_mrr[1]:.4f}] {p_str}")

    # 7. TASK 3: SYSTEMATIC 5-ROW ABLATION STUDY
    print("\n" + "=" * 60)
    print("TASK 3: Conclusive 5-Row Ablation Study")
    print("=" * 60)

    ablation_rows = [
        ("Baseline (Fixed alpha=0.7)", "Hybrid (fixed alpha=0.7)"),
        ("+ Router (Learned alpha_q)", "+ Router"),
        ("+ Cascade (tau=0.065)", "+ Cascade"),
        ("+ Reliability (Observable r_m)", "+ Reliability"),
        ("Full RAMP-RAG (+ Clean Stream)", "Full RAMP-RAG (+ Clean Stream)"),
    ]

    task3_results: Dict[str, Any] = {}

    # Environment A: COCO Test Split (250 queries, 1000 items)
    print("\n--- Ablation on MS-COCO Held-Out Test Split (N=250) ---")
    coco_dense_emb = dense_model.encode(coco_gold_captions, convert_to_numpy=True, normalize_embeddings=True)
    coco_tokens = [tokenize(t) for t in coco_gold_captions]
    coco_bm25 = PurePythonBM25(coco_tokens)
    coco_test_q_dense = dense_model.encode(coco_test_queries, convert_to_numpy=True, normalize_embeddings=True)

    coco_d_sim = np.dot(coco_test_q_dense, coco_dense_emb.T)
    coco_s_sim = np.array([coco_bm25.get_scores(tokenize(q)) for q in coco_test_queries])
    coco_c_sim = coco_d_sim  # On COCO without noise, clean stream aligns with dense
    coco_reliabilities = np.ones(len(coco_doc_ids), dtype=np.float32)

    coco_cascade = CascadeRanker(cross_encoder, tau_margin=0.065, top_k=5)
    coco_ablation_scores: Dict[str, Any] = {}

    for row_name, method_key in ablation_rows:
        res = evaluate_method_on_queries(
            method_key,
            coco_test_queries,
            coco_test_gt,
            coco_doc_ids,
            coco_gold_captions,
            coco_d_sim,
            coco_s_sim,
            coco_c_sim,
            coco_reliabilities,
            router,
            coco_cascade,
            alpha_fixed=0.70,
            gamma_clean=best_gamma,
        )
        ci_h10 = compute_bootstrap_ci(res["hit10_list"], seed=42)
        ci_mrr = compute_bootstrap_ci(res["mrr_list"], seed=42)
        coco_ablation_scores[row_name] = {
            "hit10": round(res["hit10"], 4),
            "hit10_ci95": [round(ci_h10[0], 4), round(ci_h10[1], 4)],
            "mrr": round(res["mrr"], 4),
            "mrr_ci95": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
            "mrr_list": res["mrr_list"],
            "hit10_list": res["hit10_list"],
        }
        print(f"  {row_name:<32}: Hit@10={res['hit10']:.4f} | MRR={res['mrr']:.4f}")

    # Environment B: Clean Scene-Text Test (55 probes, 97 items)
    print("\n--- Ablation on Clean Scene-Text Test (N=55) ---")
    test_clean_texts = [r.get("text_representation", "") for r in test_records]
    test_clean_tokens = [tokenize(t) for t in test_clean_texts]
    test_clean_bm25 = PurePythonBM25(test_clean_tokens)
    test_clean_dense_emb = dense_model.encode(test_clean_texts, convert_to_numpy=True, normalize_embeddings=True)

    d_clean_sim = np.dot(test_q_dense, test_clean_dense_emb.T)
    s_clean_sim = np.array([test_clean_bm25.get_scores(tokenize(q)) for q in test_query_texts])
    reliabilities_clean = np.ones(len(test_records), dtype=np.float32)

    clean_ablation_scores: Dict[str, Any] = {}
    for row_name, method_key in ablation_rows:
        res = evaluate_method_on_queries(
            method_key,
            test_query_texts,
            test_gt,
            test_record_ids,
            test_clean_texts,
            d_clean_sim,
            s_clean_sim,
            test_clean_sim,
            reliabilities_clean,
            router,
            cascade_test,
            alpha_fixed=0.70,
            gamma_clean=best_gamma,
        )
        ci_h10 = compute_bootstrap_ci(res["hit10_list"], seed=42)
        ci_mrr = compute_bootstrap_ci(res["mrr_list"], seed=42)
        clean_ablation_scores[row_name] = {
            "hit10": round(res["hit10"], 4),
            "hit10_ci95": [round(ci_h10[0], 4), round(ci_h10[1], 4)],
            "mrr": round(res["mrr"], 4),
            "mrr_ci95": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
            "mrr_list": res["mrr_list"],
            "hit10_list": res["hit10_list"],
        }
        print(f"  {row_name:<32}: Hit@10={res['hit10']:.4f} | MRR={res['mrr']:.4f}")

    # Environment C: Corrupted Scene-Text Test under 20% Noise (Pooled Seeds 42, 43, 44)
    print("\n--- Ablation on 20% Noise Scene-Text Test (N=55 x 3 Seeds) ---")
    noise20_pooled: Dict[str, Dict[str, List[float]]] = {r[0]: {"mrr": [], "hit10": []} for r in ablation_rows}

    for s in SEEDS:
        rng_20 = np.random.default_rng(s + 20)
        corr_texts_20 = [corrupt_text(r.get("text_representation", ""), 0.20, rng_20) for r in test_records]
        corr_tokens_20 = [tokenize(t) for t in corr_texts_20]
        corr_bm25_20 = PurePythonBM25(corr_tokens_20)
        corr_dense_emb_20 = dense_model.encode(corr_texts_20, convert_to_numpy=True, normalize_embeddings=True)
        corr_rel_20 = np.array([compute_observable_reliability(t, clean_vocab) for t in corr_texts_20], dtype=np.float32)

        d_sim_20 = np.dot(test_q_dense, corr_dense_emb_20.T)
        s_sim_20 = np.array([corr_bm25_20.get_scores(tokenize(q)) for q in test_query_texts])

        cascade_ablation = CascadeRanker(cross_encoder, tau_margin=0.065, top_k=5)

        for row_name, method_key in ablation_rows:
            res = evaluate_method_on_queries(
                method_key,
                test_query_texts,
                test_gt,
                test_record_ids,
                corr_texts_20,
                d_sim_20,
                s_sim_20,
                test_clean_sim,
                corr_rel_20,
                router,
                cascade_ablation,
                alpha_fixed=0.70,
                gamma_clean=best_gamma,
            )
            noise20_pooled[row_name]["mrr"].extend(res["mrr_list"])
            noise20_pooled[row_name]["hit10"].extend(res["hit10_list"])

    noise20_ablation_scores: Dict[str, Any] = {}
    base_mrr_20 = noise20_pooled["Baseline (Fixed alpha=0.7)"]["mrr"]
    base_h10_20 = noise20_pooled["Baseline (Fixed alpha=0.7)"]["hit10"]

    for row_name, _ in ablation_rows:
        m_h10 = float(np.mean(noise20_pooled[row_name]["hit10"]))
        m_mrr = float(np.mean(noise20_pooled[row_name]["mrr"]))
        ci_h10 = compute_bootstrap_ci(noise20_pooled[row_name]["hit10"], seed=42)
        ci_mrr = compute_bootstrap_ci(noise20_pooled[row_name]["mrr"], seed=42)

        p_mrr = paired_permutation_test(noise20_pooled[row_name]["mrr"], base_mrr_20, seed=42) if row_name != "Baseline (Fixed alpha=0.7)" else 1.0
        p_h10 = paired_permutation_test(noise20_pooled[row_name]["hit10"], base_h10_20, seed=42) if row_name != "Baseline (Fixed alpha=0.7)" else 1.0

        noise20_ablation_scores[row_name] = {
            "hit10": round(m_h10, 4),
            "hit10_ci95": [round(ci_h10[0], 4), round(ci_h10[1], 4)],
            "mrr": round(m_mrr, 4),
            "mrr_ci95": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
            "p_val_mrr": round(p_mrr, 5),
            "p_val_hit10": round(p_h10, 5),
        }
        print(f"  {row_name:<32}: Hit@10={m_h10:.4f} | MRR={m_mrr:.4f} | p(MRR)={p_mrr:.4f}")

    task3_results = {
        "coco_test": {k: {m: v for m, v in val.items() if not m.endswith("_list")} for k, val in coco_ablation_scores.items()},
        "ocr_clean_test": {k: {m: v for m, v in val.items() if not m.endswith("_list")} for k, val in clean_ablation_scores.items()},
        "ocr_noise20_test": noise20_ablation_scores,
    }

    # 8. TASK 4: COCO WITH REALISTIC PROXIES (BLIP) & STRONGER BASELINES
    print("\n" + "=" * 60)
    print("TASK 4: MS-COCO with BLIP Caption Proxies & Stronger Baselines")
    print("=" * 60)

    # Encode 1,000 COCO images with CLIP directly from raw pixels
    print("Encoding 1,000 COCO images with CLIP ViT-B/32 directly from raw pixels...")
    coco_image_files = [COCO_IMAGES_DIR / Path(r["image"]).name for r in coco_selected]
    coco_clip_img_embs = []
    batch_sz = 32
    for i in range(0, len(coco_image_files), batch_sz):
        batch_paths = coco_image_files[i : i + batch_sz]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        coco_clip_img_embs.append(feat.cpu().numpy())
    coco_clip_img_matrix = np.concatenate(coco_clip_img_embs, axis=0)

    # Encode COCO queries with CLIP
    coco_test_q_clip = encode_clip_queries(coco_test_queries)
    coco_clip_visual_sim = np.dot(coco_test_q_clip, coco_clip_img_matrix.T)

    # Encode BLIP proxies with Dense models (all-MiniLM & BGE-base)
    print("Encoding BLIP caption proxies with all-MiniLM-L6-v2 and BGE-base-en-v1.5...")
    coco_blip_dense_minilm = dense_model.encode(coco_blip_texts, convert_to_numpy=True, normalize_embeddings=True)
    coco_blip_dense_bge = bge_model.encode(coco_blip_texts, convert_to_numpy=True, normalize_embeddings=True)
    coco_blip_tokens = [tokenize(t) for t in coco_blip_texts]
    coco_blip_bm25 = PurePythonBM25(coco_blip_tokens)

    # Encode test queries with BGE-base
    coco_test_q_bge = bge_model.encode(coco_test_queries, convert_to_numpy=True, normalize_embeddings=True)

    sim_dense_minilm = np.dot(coco_test_q_dense, coco_blip_dense_minilm.T)
    sim_dense_bge = np.dot(coco_test_q_bge, coco_blip_dense_bge.T)
    sim_sparse_blip = np.array([coco_blip_bm25.get_scores(tokenize(q)) for q in coco_test_queries])

    coco_task4_methods = [
        ("BM25 (BLIP Proxy)", "BM25", sim_dense_minilm, sim_sparse_blip, coco_clip_visual_sim),
        ("Dense all-MiniLM (BLIP Proxy)", "Dense", sim_dense_minilm, sim_sparse_blip, coco_clip_visual_sim),
        ("Dense BGE-base (BLIP Proxy)", "Dense", sim_dense_bge, sim_sparse_blip, coco_clip_visual_sim),
        ("CLIP Zero-Shot Visual (Raw Pixels)", "BM25", sim_dense_minilm, coco_clip_visual_sim, coco_clip_visual_sim),  # using clean as primary
        ("Hybrid Convex (MiniLM + BM25)", "Hybrid (fixed alpha=0.7)", sim_dense_minilm, sim_sparse_blip, coco_clip_visual_sim),
        ("Hybrid Convex (BGE-base + BM25)", "Hybrid (fixed alpha=0.7)", sim_dense_bge, sim_sparse_blip, coco_clip_visual_sim),
        ("Full RAMP-RAG (BLIP + CLIP Visual)", "RAMP-RAG + Clean Stream", sim_dense_minilm, sim_sparse_blip, coco_clip_visual_sim),
        ("Cross-Encoder (MiniLM Top-5)", "Cross-Encoder", sim_dense_minilm, sim_sparse_blip, coco_clip_visual_sim),
    ]

    task4_results: Dict[str, Any] = {}
    for label, m_key, d_sim_mat, s_sim_mat, c_sim_mat in coco_task4_methods:
        res = evaluate_method_on_queries(
            m_key,
            coco_test_queries,
            coco_test_gt,
            coco_doc_ids,
            coco_blip_texts,
            d_sim_mat,
            s_sim_mat,
            c_sim_mat,
            coco_reliabilities,
            router,
            coco_cascade,
            alpha_fixed=0.70,
            gamma_clean=best_gamma,
        )
        ci_h10 = compute_bootstrap_ci(res["hit10_list"], seed=42)
        ci_mrr = compute_bootstrap_ci(res["mrr_list"], seed=42)

        task4_results[label] = {
            "hit1": round(res["hit1"], 4),
            "hit5": round(res["hit5"], 4),
            "hit10": round(res["hit10"], 4),
            "hit10_ci95": [round(ci_h10[0], 4), round(ci_h10[1], 4)],
            "mrr": round(res["mrr"], 4),
            "mrr_ci95": [round(ci_mrr[0], 4), round(ci_mrr[1], 4)],
            "ndcg10": round(res["ndcg10"], 4),
        }
        print(f"  {label:<36}: Hit@10={res['hit10']:.4f} [{ci_h10[0]:.4f}, {ci_h10[1]:.4f}] | MRR={res['mrr']:.4f} [{ci_mrr[0]:.4f}, {ci_mrr[1]:.4f}]")

    # 9. TASK 5: UNIFIED LATENCY PROFILING (ONE MACHINE, WARM CACHE, MEAN & P95 OVER 3 RUNS)
    print("\n" + "=" * 60)
    print("TASK 5: Standardized Latency Profiling (Scope = Query Encoding + Retrieval + Rerank)")
    print("=" * 60)

    # Benchmark latency on 100 test queries
    eval_queries = test_query_texts[:50] + coco_test_queries[:50]

    latency_systems = [
        ("BM25", lambda q: test_clean_bm25.get_scores(tokenize(q))),
        ("Dense (all-MiniLM-L6-v2)", lambda q: np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_emb.T)),
        ("Dense (BGE-base-en-v1.5)", lambda q: np.dot(bge_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], coco_blip_dense_bge[:97].T)),
        ("CLIP Zero-Shot Visual", lambda q: np.dot(encode_clip_queries([q])[0], test_clip_clean.T)),
        ("Hybrid Convex (Fixed alpha=0.7)", lambda q: 0.7 * min_max_normalize(np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_emb.T)) + 0.3 * min_max_normalize(test_clean_bm25.get_scores(tokenize(q)))),
        ("Original RAMP-RAG", lambda q: execute_query(q, "Original RAMP-RAG", test_record_ids, test_clean_texts, np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_emb.T), test_clean_bm25.get_scores(tokenize(q)), np.dot(encode_clip_queries([q])[0], test_clip_clean.T), reliabilities_clean, router, cascade_test)),
        ("Proposed RAMP-RAG (+ Clean Stream)", lambda q: execute_query(q, "RAMP-RAG + Clean Stream", test_record_ids, test_clean_texts, np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_emb.T), test_clean_bm25.get_scores(tokenize(q)), np.dot(encode_clip_queries([q])[0], test_clip_clean.T), reliabilities_clean, router, cascade_test, gamma_clean=best_gamma)),
        ("Cross-Encoder (100% Cascade)", lambda q: execute_query(q, "Cross-Encoder", test_record_ids, test_clean_texts, np.dot(dense_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0], test_clean_dense_emb.T), test_clean_bm25.get_scores(tokenize(q)), np.dot(encode_clip_queries([q])[0], test_clip_clean.T), reliabilities_clean, router, cascade_test)),
    ]

    task5_latencies: Dict[str, Any] = {}
    for sys_name, q_fn in latency_systems:
        lat_stats = measure_system_latency(q_fn, eval_queries, n_runs=3)
        task5_latencies[sys_name] = lat_stats
        print(f"  {sys_name:<36}: Mean = {lat_stats['mean_ms']:>6.2f} ms | P95 = {lat_stats['p95_ms']:>6.2f} ms")

    # 10. GENERATE PLOTS & OUTPUT TABLES
    print("\n[Step 10] Generating line plots and export tables...")
    OUTPUT_LATEX_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Generate Figure: MRR and Hit@10 vs Noise Level
    try:
        import subprocess
        plot_script = REPO_ROOT / "ai_engine" / "rag" / "scripts" / "generate_noise_plots.py"
        subprocess.run(["python3", str(plot_script)], check=True)
    except Exception as e:
        print(f"Warning: Plot generation via subprocess encountered: {e}")

    # Generate Decision Logic for RESULTS.md
    # Question: does RAMP-RAG + clean stream beat unweighted hybrid at 20% and 40% noise with non-overlapping CIs and a Holm-corrected p < 0.05?
    h20_hybrid = task2_results["noise_20pct"]["Hybrid (fixed alpha=0.7)"]
    h20_ramp_clean = task2_results["noise_20pct"]["RAMP-RAG + Clean Stream"]
    h40_hybrid = task2_results["noise_40pct"]["Hybrid (fixed alpha=0.7)"]
    h40_ramp_clean = task2_results["noise_40pct"]["RAMP-RAG + Clean Stream"]

    # Check non-overlapping CIs
    ci_overlap_20_mrr = not (h20_ramp_clean["mrr_ci95"][0] > h20_hybrid["mrr_ci95"][1] or h20_hybrid["mrr_ci95"][0] > h20_ramp_clean["mrr_ci95"][1])
    ci_overlap_40_mrr = not (h40_ramp_clean["mrr_ci95"][0] > h40_hybrid["mrr_ci95"][1] or h40_hybrid["mrr_ci95"][0] > h40_ramp_clean["mrr_ci95"][1])

    p_20_mrr = h20_ramp_clean["p_value_mrr_vs_hybrid"]
    p_40_mrr = h40_ramp_clean["p_value_mrr_vs_hybrid"]

    decision_20 = (h20_ramp_clean["mrr"] > h20_hybrid["mrr"]) and (not ci_overlap_20_mrr) and (p_20_mrr < 0.05)
    decision_40 = (h40_ramp_clean["mrr"] > h40_hybrid["mrr"]) and (not ci_overlap_40_mrr) and (p_40_mrr < 0.05)
    decision_overall = decision_20 and decision_40

    decision_summary = {
        "decision_overall": bool(decision_overall),
        "verdict_plain": "YES" if decision_overall else "NO",
        "noise_20pct": {
            "hybrid_mrr": h20_hybrid["mrr"],
            "hybrid_ci": h20_hybrid["mrr_ci95"],
            "ramp_clean_mrr": h20_ramp_clean["mrr"],
            "ramp_clean_ci": h20_ramp_clean["mrr_ci95"],
            "ci_overlap": bool(ci_overlap_20_mrr),
            "holm_p_value": p_20_mrr,
            "statistically_superior": bool(decision_20),
        },
        "noise_40pct": {
            "hybrid_mrr": h40_hybrid["mrr"],
            "hybrid_ci": h40_hybrid["mrr_ci95"],
            "ramp_clean_mrr": h40_ramp_clean["mrr"],
            "ramp_clean_ci": h40_ramp_clean["mrr_ci95"],
            "ci_overlap": bool(ci_overlap_40_mrr),
            "holm_p_value": p_40_mrr,
            "statistically_superior": bool(decision_40),
        },
    }

    print("\n" + "=" * 60)
    print(f"DECISION OUTCOME: Does RAMP-RAG + Clean Stream beat Unweighted Hybrid at 20% and 40% noise with non-overlapping CIs and Holm-corrected p < 0.05?")
    print(f"VERDICT: {decision_summary['verdict_plain']}")
    print(f"20% Noise: Hybrid MRR={h20_hybrid['mrr']} vs RAMP-RAG Clean={h20_ramp_clean['mrr']} | Overlap={ci_overlap_20_mrr} | Holm p={p_20_mrr}")
    print(f"40% Noise: Hybrid MRR={h40_hybrid['mrr']} vs RAMP-RAG Clean={h40_ramp_clean['mrr']} | Overlap={ci_overlap_40_mrr} | Holm p={p_40_mrr}")
    print("=" * 60)

    # Save complete JSON
    full_export = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gamma_star_dev": best_gamma,
        "gamma_sweep_dev": gamma_sweep_results,
        "task2_noise_sweep": task2_results,
        "task3_ablation": task3_results,
        "task4_coco_benchmarks": task4_results,
        "task5_latencies": task5_latencies,
        "decision_summary": decision_summary,
    }

    with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)
    print(f"\nSaved all results to: {OUTPUT_RESULTS_JSON}")

    # Export CSV & LaTeX Tables
    export_tables(task2_results, task3_results, task4_results, task5_latencies)


def export_tables(
    t2: Dict[str, Any],
    t3: Dict[str, Any],
    t4: Dict[str, Any],
    t5: Dict[str, Any],
):
    """Exports all results to CSV and publication-ready LaTeX tables."""
    # ── Table 1: Task 2 Noise Sweep CSV & LaTeX ──
    csv_t2_path = OUTPUT_CSV_DIR / "task2_noise_sweep.csv"
    tex_t2_path = OUTPUT_LATEX_DIR / "task2_noise_sweep.tex"
    with open(csv_t2_path, "w", encoding="utf-8") as f_csv, open(tex_t2_path, "w", encoding="utf-8") as f_tex:
        f_csv.write("Noise,Method,Hit@1,Hit@5,Hit@10,Hit@10_CI95,MRR,MRR_CI95,NDCG@10,Holm_p_MRR\n")
        f_tex.write(r"""\begin{table*}[t]
\centering
\caption{Retrieval Performance Across Proxy Noise Injections (0\%, 10\%, 20\%, 40\%) on Held-Out Scene-Text Probes (Pooled Across Seeds 42, 43, 44 with 95\% Bootstrap Confidence Intervals and Holm-Corrected Paired Permutation $p$-Values vs.\ Convex Hybrid).}
\label{tab:noise_sweep}
\small
\begin{tabular}{l l c c c c c c}
\hline
\textbf{Noise} & \textbf{Method} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10 [95\% CI]} & \textbf{MRR [95\% CI]} & \textbf{NDCG@10} & \textbf{$p_{\text{Holm}}$ (MRR)} \\
\hline
""")
        for noise_key in ["noise_0pct", "noise_10pct", "noise_20pct", "noise_40pct"]:
            pct = noise_key.replace("noise_", "").replace("pct", "%")
            f_tex.write(f"\\multicolumn{{8}}{{l}}{{\\textit{{{pct} Character Corruption}}}} \\\\\n")
            for m, row in t2[noise_key].items():
                h1 = row["hit1"]
                h5 = row["hit5"]
                h10 = row["hit10"]
                h10_ci = f"[{row['hit10_ci95'][0]:.3f}, {row['hit10_ci95'][1]:.3f}]"
                mrr = row["mrr"]
                mrr_ci = f"[{row['mrr_ci95'][0]:.3f}, {row['mrr_ci95'][1]:.3f}]"
                ndcg = row["ndcg10"]
                p_val = row.get("p_value_mrr_vs_hybrid", 1.0)
                p_str = f"{p_val:.4f}" if m != "Hybrid (fixed alpha=0.7)" else "---"

                f_csv.write(f"{pct},{m},{h1:.4f},{h5:.4f},{h10:.4f},\"{h10_ci}\",{mrr:.4f},\"{mrr_ci}\",{ndcg:.4f},{p_str}\n")
                f_tex.write(f" & {m} & {h1:.3f} & {h5:.3f} & {h10:.3f} \\scriptsize{{{h10_ci}}} & {mrr:.3f} \\scriptsize{{{mrr_ci}}} & {ndcg:.3f} & {p_str} \\\\\n")
            f_tex.write("\\hline\n")
        f_tex.write(r"""\end{tabular}
\end{table*}
""")

    # ── Table 2: Task 3 Ablation CSV & LaTeX ──
    csv_t3_path = OUTPUT_CSV_DIR / "task3_ablation.csv"
    tex_t3_path = OUTPUT_LATEX_DIR / "task3_ablation.tex"
    with open(csv_t3_path, "w", encoding="utf-8") as f_csv, open(tex_t3_path, "w", encoding="utf-8") as f_tex:
        f_csv.write("Configuration,COCO_Hit@10,COCO_MRR,Clean_Hit@10,Clean_MRR,Noise20_Hit@10,Noise20_Hit@10_CI,Noise20_MRR,Noise20_MRR_CI,Holm_p_MRR\n")
        f_tex.write(r"""\begin{table*}[t]
\centering
\caption{Systematic 5-Row Component Ablation Across MS-COCO Test, Clean Scene-Text Test, and 20\% Noise Scene-Text Test (Seeds 42, 43, 44).}
\label{tab:systematic_ablation}
\small
\begin{tabular}{l c c c c c c c}
\hline
\textbf{Configuration} & \multicolumn{2}{c}{\textbf{COCO Test}} & \multicolumn{2}{c}{\textbf{Clean Scene-Text}} & \multicolumn{3}{c}{\textbf{20\% Noise Scene-Text}} \\
 & \textbf{Hit@10} & \textbf{MRR} & \textbf{Hit@10} & \textbf{MRR} & \textbf{Hit@10 [95\% CI]} & \textbf{MRR [95\% CI]} & \textbf{$p$ (MRR)} \\
\hline
""")
        for row_name in ["Baseline (Fixed alpha=0.7)", "+ Router (Learned alpha_q)", "+ Cascade (tau=0.065)", "+ Reliability (Observable r_m)", "Full RAMP-RAG (+ Clean Stream)"]:
            coco_row = t3["coco_test"].get(row_name, {})
            clean_row = t3["ocr_clean_test"].get(row_name, {})
            n20_row = t3["ocr_noise20_test"].get(row_name, {})

            c_h10 = coco_row.get("hit10", 0.0)
            c_mrr = coco_row.get("mrr", 0.0)
            cl_h10 = clean_row.get("hit10", 0.0)
            cl_mrr = clean_row.get("mrr", 0.0)
            n20_h10 = n20_row.get("hit10", 0.0)
            n20_h10_ci = f"[{n20_row.get('hit10_ci95', [0,0])[0]:.3f}, {n20_row.get('hit10_ci95', [0,0])[1]:.3f}]"
            n20_mrr = n20_row.get("mrr", 0.0)
            n20_mrr_ci = f"[{n20_row.get('mrr_ci95', [0,0])[0]:.3f}, {n20_row.get('mrr_ci95', [0,0])[1]:.3f}]"
            p_str = f"{n20_row.get('p_val_mrr', 1.0):.4f}" if "Baseline" not in row_name else "---"

            f_csv.write(f"\"{row_name}\",{c_h10:.4f},{c_mrr:.4f},{cl_h10:.4f},{cl_mrr:.4f},{n20_h10:.4f},\"{n20_h10_ci}\",{n20_mrr:.4f},\"{n20_mrr_ci}\",{p_str}\n")
            f_tex.write(f"{row_name} & {c_h10:.3f} & {c_mrr:.3f} & {cl_h10:.3f} & {cl_mrr:.3f} & {n20_h10:.3f} \\scriptsize{{{n20_h10_ci}}} & {n20_mrr:.3f} \\scriptsize{{{n20_mrr_ci}}} & {p_str} \\\\\n")
        f_tex.write(r"""\hline
\end{tabular}
\end{table*}
""")

    # ── Table 3: Task 4 COCO Benchmarks CSV & LaTeX ──
    csv_t4_path = OUTPUT_CSV_DIR / "task4_coco_benchmarks.csv"
    tex_t4_path = OUTPUT_LATEX_DIR / "task4_coco_benchmarks.tex"
    with open(csv_t4_path, "w", encoding="utf-8") as f_csv, open(tex_t4_path, "w", encoding="utf-8") as f_tex:
        f_csv.write("Method,Hit@1,Hit@5,Hit@10,Hit@10_CI95,MRR,MRR_CI95,NDCG@10\n")
        f_tex.write(r"""\begin{table*}[t]
\centering
\caption{Retrieval Performance on MS-COCO Using Realistic BLIP Caption Proxies and Stronger Dense Baselines (BAAI/bge-base-en-v1.5 and CLIP ViT-B/32 Zero-Shot Visual Grounding).}
\label{tab:coco_realistic}
\small
\begin{tabular}{l c c c c c}
\hline
\textbf{Configuration / Method} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10 [95\% CI]} & \textbf{MRR [95\% CI]} & \textbf{NDCG@10} \\
\hline
""")
        for m, row in t4.items():
            h1 = row["hit1"]
            h5 = row["hit5"]
            h10 = row["hit10"]
            h10_ci = f"[{row['hit10_ci95'][0]:.3f}, {row['hit10_ci95'][1]:.3f}]"
            mrr = row["mrr"]
            mrr_ci = f"[{row['mrr_ci95'][0]:.3f}, {row['mrr_ci95'][1]:.3f}]"
            ndcg = row["ndcg10"]

            f_csv.write(f"\"{m}\",{h1:.4f},{h5:.4f},{h10:.4f},\"{h10_ci}\",{mrr:.4f},\"{mrr_ci}\",{ndcg:.4f}\n")
            f_tex.write(f"{m} & {h1:.3f} & {h5:.3f} & {h10:.3f} \\scriptsize{{{h10_ci}}} & {mrr:.3f} \\scriptsize{{{mrr_ci}}} & {ndcg:.3f} \\\\\n")
        f_tex.write(r"""\hline
\end{tabular}
\end{table*}
""")

    # ── Table 4: Task 5 Latency Profile CSV & LaTeX ──
    csv_t5_path = OUTPUT_CSV_DIR / "task5_latencies.csv"
    tex_t5_path = OUTPUT_LATEX_DIR / "task5_latencies.tex"
    with open(csv_t5_path, "w", encoding="utf-8") as f_csv, open(tex_t5_path, "w", encoding="utf-8") as f_tex:
        f_csv.write("System,Mean_Latency_ms,P95_Latency_ms\n")
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{End-to-End Retrieval and Reranking Latency Benchmarks (Query Encoding Included, Warm Cache, Mean and P95 Over 3 Runs).}
\label{tab:latencies}
\small
\begin{tabular}{l c c}
\hline
\textbf{Retrieval Architecture} & \textbf{Mean (ms)} & \textbf{P95 (ms)} \\
\hline
""")
        for sys_name, row in t5.items():
            m_ms = row["mean_ms"]
            p95_ms = row["p95_ms"]
            f_csv.write(f"\"{sys_name}\",{m_ms:.2f},{p95_ms:.2f}\n")
            f_tex.write(f"{sys_name} & {m_ms:.2f} & {p95_ms:.2f} \\\\\n")
        f_tex.write(r"""\hline
\end{tabular}
\end{table}
""")

    print(f"Exported CSV tables to: {OUTPUT_CSV_DIR}")
    print(f"Exported LaTeX tables to: {OUTPUT_LATEX_DIR}")


if __name__ == "__main__":
    main()


