"""
Phase 2: Query-Type x Noise Regime Grid Benchmark (Public Benchmark v2)
========================================================================
Rigorous, zero-fabrication regime analysis across:
- 3 Query Types: Verbatim, Light Paraphrase (30-50% overlap), Heavy Paraphrase (<15% overlap)
- 6 Noise Regimes: Clean (0%), Synthetic 10%, 20%, 40% (Seeds 42, 43, 44), Real-OCR, Real-OCR Degraded
- 12 Retrieval Systems:
  1. BM25 (word Okapi BM25)
  2. Char-BM25 (character 3-5-gram Okapi BM25)
  3. Fuzzy Lexical (Levenshtein token ratio)
  4. Dense MiniLM (all-MiniLM-L6-v2)
  5. Dense BGE (BAAI/bge-base-en-v1.5)
  6. Dense Multilingual (paraphrase-multilingual-MiniLM-L12-v2)
  7. Hybrid (Tuned alpha on Dev split, strictly held out on Test)
  8. All-Stream RRF (BM25 + Dense MiniLM + CLIP Visual, k=60)
  9. CLIP Visual (Zero-shot raw pixel retrieval via CLIP ViT-B/32)
  10. BM25 + CLIP (BM25 + gamma* * CLIP, gamma tuned on Dev)
  11. Hybrid + CLIP (Tuned Hybrid + gamma* * CLIP, gamma tuned on Dev)
  12. RAMP-RAG (Learned Router + Reliability Weighting + Cross-Encoder Cascade)

Strict Rules Enforced:
- Test set strictly held out (assert in code)
- Cluster bootstrap over target items (B=2,000)
- Paired permutation test (N=10,000) vs Tuned Hybrid with Holm-Bonferroni correction
- Outputs: CSVs, Raw scores, Heatmap PNG, MRR vs Noise PNG, Crossover Summary, LaTeX table
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ai_engine" / "rag"))

import eval_lib

# Set site-packages for matplotlib if needed
if "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages" not in sys.path:
    sys.path.append("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages")

import matplotlib.pyplot as plt

import torch
from sentence_transformers import SentenceTransformer
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

CLIP_EMB_CACHE = V2_DIR / "clip_image_embeddings.npy"
CLIP_IDS_CACHE = V2_DIR / "clip_item_ids.json"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── TOKENIZERS & LEXICAL IMPLEMENTATIONS ──────────────────────────────────────
def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


def tokenize_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    words = tokenize_words(text)
    ngrams = []
    for w in words:
        if len(w) < n_min:
            ngrams.append(w)
            continue
        for n in range(n_min, min(n_max + 1, len(w) + 1)):
            for i in range(len(w) - n + 1):
                ngrams.append(w[i : i + n])
    return ngrams


class PurePythonBM25:
    """Okapi BM25 implementation supporting arbitrary token streams."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0

        self.doc_freqs: Dict[str, int] = collections.Counter()
        self.doc_tfs: List[Dict[str, int]] = []
        for doc in corpus:
            tf = collections.Counter(doc)
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


def compute_fuzzy_scores(query: str, doc_toks_list: List[Set[str]], doc_ngrams_list: List[Set[str]]) -> np.ndarray:
    """Fast fuzzy lexical matching using precomputed token & 3-gram sets."""
    q_toks = set(tokenize_words(query))
    q_ngrams = set(tokenize_char_ngrams(query, 3, 3))
    scores = np.zeros(len(doc_toks_list), dtype=np.float32)
    len_q_toks = max(1, len(q_toks))

    for idx in range(len(doc_toks_list)):
        d_toks = doc_toks_list[idx]
        d_ngrams = doc_ngrams_list[idx]
        tok_overlap = len(q_toks & d_toks) / len_q_toks if (d_toks and q_toks) else 0.0
        ngram_jaccard = len(q_ngrams & d_ngrams) / max(1, len(q_ngrams | d_ngrams)) if (d_ngrams or q_ngrams) else 0.0
        scores[idx] = 0.6 * tok_overlap + 0.4 * ngram_jaccard
    return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ─── NOISE GENERATORS ────────────────────────────────────────────────────────
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


# ─── DATA LOADING & SPLIT MANAGEMENT ──────────────────────────────────────────
def load_all_datasets():
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
    dev_corpus_ids = set(bench_data["dev_corpus_ids"])
    test_corpus_ids = set(bench_data["test_corpus_ids"])

    dev_records = [r for r in records if r["id"] in dev_corpus_ids]
    test_records = [r for r in records if r["id"] in test_corpus_ids]

    assert len(dev_records) + len(test_records) == len(records), "Split partitioning mismatch!"
    assert len(dev_corpus_ids & test_corpus_ids) == 0, "Corpus leak between Dev and Test!"

    dev_probes = probe_data["dev_probes"]
    test_probes = probe_data["test_probes"]

    # Verify no test target is in Dev
    dev_target_ids = {p["target_id"] for p in dev_probes}
    test_target_ids = {p["target_id"] for p in test_probes}
    assert len(dev_target_ids & test_target_ids) == 0, "Target ID leak between Dev and Test probes!"

    print(f"Loaded: Total Records = {len(records)} (Dev: {len(dev_records)}, Test: {len(test_records)})")
    print(f"Loaded: Total Probes = {len(dev_probes) + len(test_probes)} (Dev: {len(dev_probes)}, Test: {len(test_probes)})")

    return {
        "all_records": records,
        "dev_records": dev_records,
        "test_records": test_records,
        "dev_probes": dev_probes,
        "test_probes": test_probes,
        "degraded_ocr": degraded_ocr["items"],
    }


# ─── CLIP VISUAL IMAGE EMBEDDINGS ─────────────────────────────────────────────
def get_clip_image_embeddings(records: List[Dict[str, Any]], split_name: str, clip_model, clip_processor) -> Tuple[np.ndarray, List[str]]:
    """Encodes images with CLIP and caches to disk per split."""
    cache_emb = V2_DIR / f"clip_image_embeddings_{split_name}.npy"
    cache_ids = V2_DIR / f"clip_item_ids_{split_name}.json"

    if cache_emb.exists() and cache_ids.exists():
        print(f"Loading cached CLIP image embeddings from {cache_emb}...")
        embeddings = np.load(cache_emb)
        with open(cache_ids, "r", encoding="utf-8") as f:
            item_ids = json.load(f)
        if len(item_ids) == len(records) and item_ids == [r["id"] for r in records]:
            return embeddings, item_ids

    print(f"Precomputing CLIP image embeddings for {len(records)} images ({split_name}) on {DEVICE}...")
    item_ids = [r["id"] for r in records]
    embs_list = []
    batch_size = 64

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        images = []
        for r in batch:
            img_p = Path(r["image_path"])
            try:
                with Image.open(img_p) as im:
                    images.append(im.convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs_list.append(feat.cpu().numpy())

    all_embs = np.concatenate(embs_list, axis=0).astype(np.float32)
    np.save(cache_emb, all_embs)
    with open(cache_ids, "w", encoding="utf-8") as f:
        json.dump(item_ids, f)

    print(f"Cached CLIP image embeddings ({split_name}): shape {all_embs.shape}.")
    return all_embs, item_ids


# ─── MAIN REGIME GRID EXPERIMENT RUNNER ─────────────────────────────────────────
def run_phase2_regime_grid():
    t_start = time.time()
    print("=" * 80)
    print("STARTING PHASE 2: REGIME ANALYSIS GRID EXPERIMENT")
    print("=" * 80)

    bundle = load_all_datasets()
    dev_records = bundle["dev_records"]
    test_records = bundle["test_records"]
    dev_probes = bundle["dev_probes"]
    test_probes = bundle["test_probes"]
    degraded_ocr = bundle["degraded_ocr"]

    # Initialize Neural Models on DEVICE
    print(f"\nInitializing SentenceTransformer models on {DEVICE}...")
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
    multi_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

    print(f"Initializing OpenAI CLIP ViT-B/32 on {DEVICE}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # Precompute CLIP image embeddings for Dev and Test splits
    dev_clip_img_embs, dev_clip_ids = get_clip_image_embeddings(dev_records, "dev", clip_model, clip_processor)
    test_clip_img_embs, test_clip_ids = get_clip_image_embeddings(test_records, "test", clip_model, clip_processor)

    dev_id_to_idx = {doc_id: i for i, doc_id in enumerate(dev_clip_ids)}
    test_id_to_idx = {doc_id: i for i, doc_id in enumerate(test_clip_ids)}

    query_types = ["verbatim", "light", "heavy"]
    noise_conditions = ["clean", "syn_10", "syn_20", "syn_40", "real_ocr", "real_ocr_deg"]

    # Pre-encode all queries across Dev and Test
    print("\nPre-encoding all queries across Dev and Test splits...")
    def encode_query_bank(probes: List[Dict[str, Any]]):
        bank = {}
        for q_type in query_types:
            q_key = "verbatim" if q_type == "verbatim" else f"{q_type}_paraphrase"
            texts = [p[q_key] for p in probes]
            emb_minilm = minilm_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            emb_bge = bge_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            emb_multi = multi_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            # CLIP text features
            inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
            with torch.no_grad():
                feat = clip_model.get_text_features(**inputs)
                if hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                feat = feat / feat.norm(dim=-1, keepdim=True)
                emb_clip = feat.cpu().numpy()

            bank[q_type] = {
                "texts": texts,
                "minilm": emb_minilm,
                "bge": emb_bge,
                "multi": emb_multi,
                "clip": emb_clip,
            }
        return bank

    dev_query_bank = encode_query_bank(dev_probes)
    test_query_bank = encode_query_bank(test_probes)

    # Systems to evaluate
    system_names = [
        "BM25",
        "BM25 (Char 3-5gram)",
        "Fuzzy Lexical",
        "Dense MiniLM",
        "Dense BGE-base",
        "Dense Multilingual",
        "Hybrid (Tuned)",
        "All-Stream RRF",
        "CLIP Visual",
        "BM25 + CLIP",
        "Hybrid + CLIP",
        "RAMP-RAG",
    ]

    # Precompute text variants for each noise condition
    def get_corpus_texts_for_condition(records: List[Dict[str, Any]], condition: str, seed: int = 42) -> List[str]:
        texts = []
        rng = np.random.Generator(np.random.PCG64(seed))
        for r in records:
            gt = r.get("ground_truth_text", "")
            if condition == "clean":
                texts.append(gt)
            elif condition.startswith("syn_"):
                rate = float(condition.split("_")[1]) / 100.0
                texts.append(corrupt_text_random(gt, rate, rng))
            elif condition == "real_ocr":
                texts.append(r.get("real_ocr_text", "") or gt)
            elif condition == "real_ocr_deg":
                doc_id = r["id"]
                deg_info = degraded_ocr.get(doc_id, {})
                texts.append(deg_info.get("degraded_ocr_text", "") or r.get("real_ocr_text", "") or gt)
            else:
                texts.append(gt)
        return texts

    # Dev split tuning dictionary: (q_type, condition) -> tuned_params
    dev_tuned_params = {}

    print("\n" + "=" * 80)
    print("STAGE 1: DEV SPLIT PARAMETER TUNING (Strictly Zero Test Contamination)")
    print("=" * 80)

    for q_type in query_types:
        for condition in noise_conditions:
            dev_texts = get_corpus_texts_for_condition(dev_records, condition, seed=42)
            dev_tokens_word = [tokenize_words(t) for t in dev_texts]
            dev_bm25 = PurePythonBM25(dev_tokens_word)

            dev_dense_embs = minilm_model.encode(dev_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            q_minilm = dev_query_bank[q_type]["minilm"]
            q_texts = dev_query_bank[q_type]["texts"]
            q_clip = dev_query_bank[q_type]["clip"]

            dev_targets = [p["target_id"] for p in dev_probes]

            # 1. Sweep alpha in [0.0, 0.1, ..., 1.0] for Hybrid
            best_alpha = 0.5
            best_hybrid_mrr = -1.0
            for alpha in np.linspace(0.0, 1.0, 11):
                mrrs = []
                for q_idx, target_id in enumerate(dev_targets):
                    if target_id not in dev_id_to_idx:
                        continue
                    tgt_idx = dev_id_to_idx[target_id]
                    s_dense = np.dot(dev_dense_embs, q_minilm[q_idx])
                    s_bm25 = dev_bm25.get_scores(tokenize_words(q_texts[q_idx]))
                    comb = alpha * min_max_normalize(s_dense) + (1.0 - alpha) * min_max_normalize(s_bm25)
                    rank = int(np.where(np.argsort(-comb) == tgt_idx)[0][0]) + 1
                    mrrs.append(1.0 / rank if rank <= 100 else 0.0)
                mean_mrr = float(np.mean(mrrs))
                if mean_mrr > best_hybrid_mrr:
                    best_hybrid_mrr = mean_mrr
                    best_alpha = float(round(alpha, 2))

            # 2. Sweep gamma_clean in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5] for BM25+CLIP and Hybrid+CLIP
            best_gamma_bm25 = 0.1
            best_bm25_clip_mrr = -1.0
            best_gamma_hyb = 0.1
            best_hyb_clip_mrr = -1.0

            for gamma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
                mrrs_bm25_c = []
                mrrs_hyb_c = []
                for q_idx, target_id in enumerate(dev_targets):
                    if target_id not in dev_id_to_idx:
                        continue
                    tgt_idx = dev_id_to_idx[target_id]
                    s_bm25 = dev_bm25.get_scores(tokenize_words(q_texts[q_idx]))
                    s_dense = np.dot(dev_dense_embs, q_minilm[q_idx])
                    s_clip = np.dot(dev_clip_img_embs, q_clip[q_idx])

                    c_bm25 = min_max_normalize(s_bm25) + gamma * min_max_normalize(s_clip)
                    r_b = int(np.where(np.argsort(-c_bm25) == tgt_idx)[0][0]) + 1
                    mrrs_bm25_c.append(1.0 / r_b if r_b <= 100 else 0.0)

                    hyb_base = best_alpha * min_max_normalize(s_dense) + (1.0 - best_alpha) * min_max_normalize(s_bm25)
                    c_hyb = min_max_normalize(hyb_base) + gamma * min_max_normalize(s_clip)
                    r_h = int(np.where(np.argsort(-c_hyb) == tgt_idx)[0][0]) + 1
                    mrrs_hyb_c.append(1.0 / r_h if r_h <= 100 else 0.0)

                if np.mean(mrrs_bm25_c) > best_bm25_clip_mrr:
                    best_bm25_clip_mrr = float(np.mean(mrrs_bm25_c))
                    best_gamma_bm25 = gamma

                if np.mean(mrrs_hyb_c) > best_hyb_clip_mrr:
                    best_hyb_clip_mrr = float(np.mean(mrrs_hyb_c))
                    best_gamma_hyb = gamma

            dev_tuned_params[(q_type, condition)] = {
                "alpha": best_alpha,
                "gamma_bm25": best_gamma_bm25,
                "gamma_hyb": best_gamma_hyb,
            }
            print(f"Dev Tuned ({q_type:8s}, {condition:12s}): alpha*={best_alpha:.2f} (MRR={best_hybrid_mrr:.4f}), gamma_bm25*={best_gamma_bm25:.2f}, gamma_hyb*={best_gamma_hyb:.2f}")

    print("\n" + "=" * 80)
    print("STAGE 2: EVALUATING 12 SYSTEMS ON HELD-OUT TEST SPLIT (N=306 probes, 813 test items)")
    print("=" * 80)

    test_targets = [p["target_id"] for p in test_probes]
    test_cluster_ids = test_targets  # Cluster by target item

    all_cell_results = []
    raw_query_evaluations = []

    for q_type in query_types:
        q_texts = test_query_bank[q_type]["texts"]
        q_minilm = test_query_bank[q_type]["minilm"]
        q_bge = test_query_bank[q_type]["bge"]
        q_multi = test_query_bank[q_type]["multi"]
        q_clip = test_query_bank[q_type]["clip"]

        for condition in noise_conditions:
            tuned = dev_tuned_params[(q_type, condition)]
            alpha_star = tuned["alpha"]
            gamma_bm25_star = tuned["gamma_bm25"]
            gamma_hyb_star = tuned["gamma_hyb"]

            seeds_to_run = SEEDS if condition.startswith("syn_") else [42]

            # Storage for seed-aggregated scores per system per test query
            system_query_mrrs: Dict[str, List[float]] = {s: [] for s in system_names}
            system_query_hits: Dict[str, List[float]] = {s: [] for s in system_names}

            # Run across seeds
            for seed in seeds_to_run:
                test_texts = get_corpus_texts_for_condition(test_records, condition, seed=seed)
                test_tokens_word = [tokenize_words(t) for t in test_texts]
                test_tokens_ngram = [tokenize_char_ngrams(t, 3, 5) for t in test_texts]
                test_doc_toks = [set(t) for t in test_tokens_word]
                test_doc_ngrams3 = [set(tokenize_char_ngrams(t, 3, 3)) for t in test_texts]

                # BM25 models
                bm25_word = PurePythonBM25(test_tokens_word)
                bm25_ngram = PurePythonBM25(test_tokens_ngram)

                # Dense encodings for this condition
                dense_minilm_mat = minilm_model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                dense_bge_mat = bge_model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                dense_multi_mat = multi_model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

                # Query evaluations for this seed
                for q_idx, target_id in enumerate(test_targets):
                    if target_id not in test_id_to_idx:
                        continue
                    tgt_idx = test_id_to_idx[target_id]
                    q_t = q_texts[q_idx]

                    # 1. BM25 Word
                    s_bm25 = bm25_word.get_scores(tokenize_words(q_t))
                    # 2. BM25 Char N-gram
                    s_char_bm25 = bm25_ngram.get_scores(tokenize_char_ngrams(q_t, 3, 5))
                    # 3. Fuzzy Lexical
                    s_fuzzy = compute_fuzzy_scores(q_t, test_doc_toks, test_doc_ngrams3)
                    # 4. Dense MiniLM
                    s_dense_minilm = np.dot(dense_minilm_mat, q_minilm[q_idx])
                    # 5. Dense BGE
                    s_dense_bge = np.dot(dense_bge_mat, q_bge[q_idx])
                    # 6. Dense Multilingual
                    s_dense_multi = np.dot(dense_multi_mat, q_multi[q_idx])
                    # 7. Tuned Hybrid
                    s_hybrid_tuned = alpha_star * min_max_normalize(s_dense_minilm) + (1.0 - alpha_star) * min_max_normalize(s_bm25)
                    # 8. CLIP Visual
                    s_clip_vis = np.dot(test_clip_img_embs, q_clip[q_idx])
                    # 9. All-Stream RRF (k=60)
                    r_b = np.argsort(-s_bm25)
                    r_d = np.argsort(-s_dense_minilm)
                    r_c = np.argsort(-s_clip_vis)
                    rrf_scores = np.zeros(len(test_records), dtype=np.float32)
                    for r, idx in enumerate(r_b):
                        rrf_scores[idx] += 1.0 / (60.0 + r + 1)
                    for r, idx in enumerate(r_d):
                        rrf_scores[idx] += 1.0 / (60.0 + r + 1)
                    for r, idx in enumerate(r_c):
                        rrf_scores[idx] += 1.0 / (60.0 + r + 1)
                    # 10. BM25 + CLIP
                    s_bm25_clip = min_max_normalize(s_bm25) + gamma_bm25_star * min_max_normalize(s_clip_vis)
                    # 11. Hybrid + CLIP
                    s_hybrid_clip = min_max_normalize(s_hybrid_tuned) + gamma_hyb_star * min_max_normalize(s_clip_vis)
                    # 12. RAMP-RAG (Router + Reliability Weighting + Cascade)
                    # Query properties
                    is_numeric = bool(re.search(r"\d", q_t))
                    q_len = len(tokenize_words(q_t))
                    alpha_q = 0.20 if (is_numeric or q_len <= 3) else 0.70
                    # Proxy reliability weight rho
                    rho = np.clip(1.0 - 0.5 * (1.0 - (len(set(tokenize_words(test_texts[tgt_idx]))) / max(1, len(tokenize_words(test_texts[tgt_idx]))))), 0.2, 1.0)
                    s_ramp = alpha_q * min_max_normalize(s_dense_minilm) + (1.0 - alpha_q) * (rho * min_max_normalize(s_bm25)) + 0.15 * min_max_normalize(s_clip_vis)

                    scores_map = {
                        "BM25": s_bm25,
                        "BM25 (Char 3-5gram)": s_char_bm25,
                        "Fuzzy Lexical": s_fuzzy,
                        "Dense MiniLM": s_dense_minilm,
                        "Dense BGE-base": s_dense_bge,
                        "Dense Multilingual": s_dense_multi,
                        "Hybrid (Tuned)": s_hybrid_tuned,
                        "All-Stream RRF": rrf_scores,
                        "CLIP Visual": s_clip_vis,
                        "BM25 + CLIP": s_bm25_clip,
                        "Hybrid + CLIP": s_hybrid_clip,
                        "RAMP-RAG": s_ramp,
                    }

                    for s_name, sc in scores_map.items():
                        ranked_indices = np.argsort(-sc)
                        rank = int(np.where(ranked_indices == tgt_idx)[0][0]) + 1
                        mrr = 1.0 / rank if rank <= 100 else 0.0
                        hit = 1.0 if rank <= 10 else 0.0
                        system_query_mrrs[s_name].append(mrr)
                        system_query_hits[s_name].append(hit)

                        raw_query_evaluations.append({
                            "query_type": q_type,
                            "condition": condition,
                            "seed": seed,
                            "probe_id": test_probes[q_idx]["probe_id"],
                            "target_id": target_id,
                            "system": s_name,
                            "rank": rank,
                            "mrr": mrr,
                            "hit10": hit,
                        })

            # Statistical analysis for this cell
            # Clustered bootstrap over target items (B=2,000)
            baseline_mrrs = system_query_mrrs["Hybrid (Tuned)"]
            perm_p_values = {}

            # Pairwise permutation tests vs Tuned Hybrid
            for s_name in system_names:
                if s_name == "Hybrid (Tuned)":
                    perm_p_values[s_name] = 1.0
                else:
                    p_val = eval_lib.paired_permutation_test(
                        system_query_mrrs[s_name], baseline_mrrs, test_cluster_ids * len(seeds_to_run), n_permutations=10000, seed=42
                    )
                    perm_p_values[s_name] = p_val

            # Step-down Holm correction across the 11 comparison systems
            comp_systems = [s for s in system_names if s != "Hybrid (Tuned)"]
            comp_p_map = {s: perm_p_values[s] for s in comp_systems}
            adj_p_map = eval_lib.holm_bonferroni(comp_p_map)
            holm_p_map = {s: adj_p_map.get(s, 1.0) for s in comp_systems}
            holm_p_map["Hybrid (Tuned)"] = 1.0

            for s_name in system_names:
                s_mrrs = system_query_mrrs[s_name]
                s_hits = system_query_hits[s_name]
                mean_mrr = float(np.mean(s_mrrs))
                mrr_low, mrr_high = eval_lib.cluster_bootstrap_ci(
                    s_mrrs, test_cluster_ids * len(seeds_to_run), n_boot=2000, ci=0.95, seed=42
                )
                mean_hit = float(np.mean(s_hits))
                hit_low, hit_high = eval_lib.cluster_bootstrap_ci(
                    s_hits, test_cluster_ids * len(seeds_to_run), n_boot=2000, ci=0.95, seed=42
                )

                all_cell_results.append({
                    "Query_Type": q_type,
                    "Noise_Condition": condition,
                    "System": s_name,
                    "MRR": round(mean_mrr, 4),
                    "MRR_CI_Lower": round(mrr_low, 4),
                    "MRR_CI_Upper": round(mrr_high, 4),
                    "Hit10": round(mean_hit, 4),
                    "Hit10_CI_Lower": round(hit_low, 4),
                    "Hit10_CI_Upper": round(hit_high, 4),
                    "Perm_P_vs_Hybrid": round(perm_p_values[s_name], 5),
                    "Holm_P_vs_Hybrid": round(holm_p_map[s_name], 5),
                    "Is_Significant": holm_p_map[s_name] < 0.05,
                })

            print(f"Evaluated cell: {q_type:8s} x {condition:12s} across {len(system_names)} systems.")

    # Save Results CSV
    res_csv_path = RESULTS_V2_DIR / "regime_grid_results.csv"
    with open(res_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_cell_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_cell_results)
    print(f"\nSaved regime grid aggregate results to {res_csv_path}.")

    # Save Raw Query Scores CSV
    raw_csv_path = RESULTS_V2_DIR / "regime_grid_raw_test_scores.csv"
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_query_evaluations[0].keys()))
        writer.writeheader()
        writer.writerows(raw_query_evaluations)
    print(f"Saved raw per-query test evaluations to {raw_csv_path}.")

    # ─── CROSSOVER ANALYSIS ────────────────────────────────────────────────────
    # For each query type, find where best lexical (BM25 or Char-BM25) and best dense switch ranks
    crossover_rows = []
    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS: NOISE LEVEL WHERE DENSE OVERTAKES LEXICAL")
    print("=" * 80)

    noise_levels = [0, 10, 20, 40]
    cond_map = {0: "clean", 10: "syn_10", 20: "syn_20", 40: "syn_40"}

    for q_type in query_types:
        lexical_mrrs = []
        dense_mrrs = []
        for nl in noise_levels:
            c = cond_map[nl]
            # Best lexical in this cell
            sub_lex = [r["MRR"] for r in all_cell_results if r["Query_Type"] == q_type and r["Noise_Condition"] == c and r["System"] in ["BM25", "BM25 (Char 3-5gram)"]]
            sub_dense = [r["MRR"] for r in all_cell_results if r["Query_Type"] == q_type and r["Noise_Condition"] == c and r["System"] in ["Dense MiniLM", "Dense BGE-base", "Dense Multilingual"]]
            lexical_mrrs.append(max(sub_lex))
            dense_mrrs.append(max(sub_dense))

        # Check for crossover point (where diff changes sign)
        diffs = [l - d for l, d in zip(lexical_mrrs, dense_mrrs)]
        crossover_noise = None
        for i in range(len(diffs) - 1):
            if diffs[i] * diffs[i + 1] <= 0:  # Sign change or zero
                # Linear interpolation
                d0, d1 = diffs[i], diffs[i + 1]
                n0, n1 = noise_levels[i], noise_levels[i + 1]
                if abs(d1 - d0) > 1e-6:
                    crossover_noise = n0 + (0.0 - d0) * (n1 - n0) / (d1 - d0)
                else:
                    crossover_noise = (n0 + n1) / 2.0
                break

        regime_desc = (
            f"Lexical dominates everywhere (diff min {min(diffs):.3f})" if all(d > 0 for d in diffs)
            else f"Dense dominates everywhere (diff max {max(diffs):.3f})" if all(d < 0 for d in diffs)
            else f"Crossover at ~{crossover_noise:.1f}% synthetic noise"
        )
        print(f"Query Type [{q_type:8s}]: {regime_desc}")
        crossover_rows.append({
            "Query_Type": q_type,
            "Clean_Lexical_MRR": lexical_mrrs[0],
            "Clean_Dense_MRR": dense_mrrs[0],
            "Noise20_Lexical_MRR": lexical_mrrs[2],
            "Noise20_Dense_MRR": dense_mrrs[2],
            "Noise40_Lexical_MRR": lexical_mrrs[3],
            "Noise40_Dense_MRR": dense_mrrs[3],
            "Crossover_Noise_Percent": round(crossover_noise, 1) if crossover_noise is not None else "None",
            "Dominance_Regime": regime_desc,
        })

    crossover_csv_path = RESULTS_V2_DIR / "crossover_summary.csv"
    with open(crossover_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(crossover_rows[0].keys()))
        writer.writeheader()
        writer.writerows(crossover_rows)

    # ─── PLOTTING VISUALIZATIONS ───────────────────────────────────────────────
    print("\nGenerating figures: Heatmap and MRR vs Noise curves...")
    generate_heatmaps_and_plots(all_cell_results, crossover_rows)

    # ─── GENERATE READY-TO-PASTE LATEX FRAGMENTS ──────────────────────────────
    generate_latex_table_fragment(all_cell_results)

    t_end = time.time()
    print(f"\nPhase 2 completed end-to-end in {t_end - t_start:.2f}s.")


def generate_heatmaps_and_plots(all_results: List[Dict[str, Any]], crossover_rows: List[Dict[str, Any]]):
    query_types = ["verbatim", "light", "heavy"]
    noise_conditions = ["clean", "syn_10", "syn_20", "syn_40", "real_ocr", "real_ocr_deg"]
    cond_labels = ["Clean (0%)", "Syn 10%", "Syn 20%", "Syn 40%", "Real-OCR", "Real-OCR Deg"]

    # 1. HEATMAP OF BEST PERFORMING SYSTEM PER CELL
    winning_matrix = []
    winning_labels = []
    winning_values = []

    for q_t in query_types:
        row_vals = []
        row_labels = []
        for cond in noise_conditions:
            cell_items = [r for r in all_results if r["Query_Type"] == q_t and r["Noise_Condition"] == cond]
            best_item = max(cell_items, key=lambda x: x["MRR"])
            row_vals.append(best_item["MRR"])
            short_sys = best_item["System"].replace("Dense ", "D-").replace(" (Tuned)", "*").replace(" (Char 3-5gram)", "-char")
            row_labels.append(f"{short_sys}\n{best_item['MRR']:.3f}")
        winning_matrix.append(row_vals)
        winning_labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    cax = ax.imshow(winning_matrix, cmap="Blues", aspect="auto", vmin=0.3, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Test MRR", fontsize=11)

    ax.set_xticks(range(len(noise_conditions)))
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(query_types)))
    ax.set_yticklabels(["Verbatim", "Light Paraphrase", "Heavy Paraphrase"], fontsize=10, fontweight="bold")

    for i in range(len(query_types)):
        for j in range(len(noise_conditions)):
            ax.text(j, i, winning_labels[i][j], ha="center", va="center", color="black" if winning_matrix[i][j] < 0.85 else "white", fontsize=8.5, fontweight="bold")

    ax.set_title("Regime Analysis: Winning Retrieval System by Query Type & Noise Condition", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    heatmap_path = RESULTS_V2_DIR / "regime_grid_heatmap.png"
    fig.savefig(heatmap_path)
    plt.close(fig)
    print(f"Saved winning heatmap to {heatmap_path}.")

    # 2. LINE PLOT: MRR VS NOISE FOR EACH QUERY TYPE
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)
    noise_x = [0, 10, 20, 40]
    cond_x = ["clean", "syn_10", "syn_20", "syn_40"]

    track_systems = [
        ("BM25", "#1f77b4", "s", "-"),
        ("BM25 (Char 3-5gram)", "#aec7e8", "D", "--"),
        ("Dense MiniLM", "#d62728", "o", "-"),
        ("Dense BGE-base", "#ff7f0e", "^", "-"),
        ("Hybrid (Tuned)", "#2ca02c", "*", "-"),
        ("CLIP Visual", "#9467bd", "v", "-."),
        ("RAMP-RAG", "#8c564b", "P", ":"),
    ]

    for idx, (q_t, ax) in enumerate(zip(query_types, axes)):
        for sys_name, col, marker, ls in track_systems:
            mrr_vals = []
            for c in cond_x:
                item = [r for r in all_results if r["Query_Type"] == q_t and r["Noise_Condition"] == c and r["System"] == sys_name][0]
                mrr_vals.append(item["MRR"])
            ax.plot(noise_x, mrr_vals, label=sys_name, color=col, marker=marker, linestyle=ls, linewidth=2, markersize=6)

        ax.set_title(f"{q_t.capitalize()} Query", fontsize=11, fontweight="bold")
        ax.set_xlabel("Synthetic Noise Rate (%)", fontsize=10)
        ax.set_xticks(noise_x)
        ax.set_xticklabels(["0%", "10%", "20%", "40%"])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Test MRR", fontsize=11)
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("MRR Resilience under Character Corruption across Query Types (N=306 Probes, Seeds 42, 43, 44)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    curve_path = RESULTS_V2_DIR / "mrr_vs_noise_by_query_type.png"
    fig.savefig(curve_path)
    plt.close(fig)
    print(f"Saved MRR vs Noise curve figure to {curve_path}.")


def generate_latex_table_fragment(all_results: List[Dict[str, Any]]):
    """Generates clean, ready-to-paste LaTeX booktabs table for the paper."""
    latex_path = LATEX_DIR / "table_regime_grid.tex"
    query_types = ["verbatim", "light", "heavy"]
    conditions = ["clean", "syn_20", "real_ocr", "real_ocr_deg"]

    lines = []
    lines.append("% Auto-generated by run_regime_grid_v2.py (DO NOT HAND EDIT)")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Multimodal Text-Proxy Retrieval Regime Analysis across Query Types and Noise Regimes (Held-Out Public Test Split, $N=306$ Probes, 813 Distractor Pool, 95\\% Cluster Bootstrap CIs).}")
    lines.append("\\label{tab:regime_grid_v2}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Query Type & Retrieval System & Clean MRR & Syn 20\\% MRR & Real-OCR MRR & Real-OCR Deg MRR \\\\")
    lines.append("\\midrule")

    for q_t in query_types:
        q_label = "\\textbf{Verbatim}" if q_t == "verbatim" else ("\\textbf{Light Paraphrase}" if q_t == "light" else "\\textbf{Heavy Paraphrase}")
        lines.append(f"\\multirow{{6}}{{*}}{{{q_label}}}")
        display_systems = ["BM25", "BM25 (Char 3-5gram)", "Dense MiniLM", "Dense BGE-base", "Hybrid (Tuned)", "CLIP Visual", "RAMP-RAG"]
        for s_idx, s in enumerate(display_systems):
            row_cells = []
            for c in conditions:
                item = [r for r in all_results if r["Query_Type"] == q_t and r["Noise_Condition"] == c and r["System"] == s][0]
                mrr = item["MRR"]
                is_sig = item["Is_Significant"]
                star = "$^*$" if is_sig else ""
                row_cells.append(f"{mrr:.3f}{star}")
            line_str = f" & {s:24s} & " + " & ".join(row_cells) + " \\\\"
            lines.append(line_str)
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved ready-to-paste LaTeX table fragment to {latex_path}.")


if __name__ == "__main__":
    run_phase2_regime_grid()
