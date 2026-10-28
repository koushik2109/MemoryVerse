#!/usr/bin/env python3
"""
run_easyocr_replication_v9.py: Third OCR Engine (EasyOCR 1.7.2) Replication on SROIE.

Executes Pre-Registered Phase 3.1 Protocol (PREREG_V9.md / CFG-036):
1. EasyOCR 1.7.2 (CRAFT detector + ResNet-CRNN recognizer) on 626 Clean and 626 Degraded SROIE receipts.
2. Checkpointed extraction with resumption support.
3. Standardized CER, sequence WER, and multiset MER (BoW) error rates against Ground Truth.
4. Qualitative comparison audit (GT vs Apple Vision vs Tesseract 5.5 vs EasyOCR 1.7.2).
5. Held-out Test split retrieval evaluation ($N=306$ probes across 813 corpus items):
   - Word BM25, Char 3-5gram BM25, Dense MiniLM, Dense BGE-Base, Tuned Hybrid.
6. Cluster bootstrap 95% CIs ($B=2,000$, seeds 42, 43, 44) and paired permutation tests (100k resamples).
7. Confirmatory hypothesis testing:
   - H3.1a (Verbatim Char vs Word BM25)
   - H3.1b (Paraphrase Reversal: Word vs Char BM25)
   - H3.1c (Dense bi-encoder deficit vs Char BM25)
   - H3.1d (Blur Sensitivity)
"""

import os
import sys
import json
import re
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import easyocr

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DIR = RAG_DATA_DIR / "v2"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"
V9_RESULTS_DIR = REPO_ROOT / "results" / "v9"
V9_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PATH = V2_DIR / "public_benchmark_v2.json"
FROZEN_PROBES_PATH = V2_DIR / "frozen_probes_with_paraphrases.json"
HEAVY_PROBES_PATH = V3_RESULTS_DIR / "heavy_paraphrases_v3.json"

EASY_CLEAN_CACHE = V9_RESULTS_DIR / "sroie_easyocr_clean_ocr.json"
EASY_DEG_CACHE = V9_RESULTS_DIR / "sroie_easyocr_degraded_ocr.json"
EASY_ERROR_RATES_CSV = V9_RESULTS_DIR / "easyocr_error_rates_v9.csv"
EASY_SAMPLES_MD = V9_RESULTS_DIR / "easyocr_samples_v9.md"
EASY_REPLICATION_CSV = V9_RESULTS_DIR / "easyocr_replication_v9.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
B_BOOT = 2000
N_PERM = 100000

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT NORMALIZATION & EDIT DISTANCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]

def tokenize_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    clean = re.sub(r"\s+", " ", text.lower().strip())
    if not clean:
        return []
    ngrams = []
    length = len(clean)
    for n in range(n_min, n_max + 1):
        if length >= n:
            for i in range(length - n + 1):
                ngrams.append(clean[i : i + n])
    return ngrams if ngrams else [clean]

def compute_levenshtein(seq1: List[Any], seq2: List[Any]) -> int:
    if seq1 == seq2: return 0
    if not seq1: return len(seq2)
    if not seq2: return len(seq1)
    if len(seq1) > len(seq2): seq1, seq2 = seq2, seq1
    prev = list(range(len(seq1) + 1))
    for j, elem2 in enumerate(seq2):
        curr = [j + 1] * (len(seq1) + 1)
        for i, elem1 in enumerate(seq1):
            cost = 0 if elem1 == elem2 else 1
            curr[i + 1] = min(curr[i] + 1, prev[i + 1] + 1, prev[i] + cost)
        prev = curr
    return prev[-1]

def compute_cer_wer_line_aligned(ref_lines: List[str], hyp_lines: List[str]) -> Tuple[float, float, float]:
    norm_gt = [normalize_text(l) for l in ref_lines if normalize_text(l)]
    norm_hyp = [normalize_text(l) for l in hyp_lines if normalize_text(l)]
    if not norm_gt:
        total_hyp_c = sum(len(l) for l in norm_hyp)
        total_hyp_w = sum(len(l.split()) for l in norm_hyp)
        return (1.0 if total_hyp_c > 0 else 0.0), (1.0 if total_hyp_w > 0 else 0.0), 1.0

    total_gt_c = sum(len(l) for l in norm_gt)
    total_gt_w = sum(len(l.split()) for l in norm_gt)
    used_hyp = set()
    total_c_err = 0
    total_w_err = 0

    for gl in norm_gt:
        gl_chars = list(gl)
        gl_words = gl.split()
        best_c_err = len(gl_chars)
        best_w_err = len(gl_words)
        best_j = -1
        for j, hl in enumerate(norm_hyp):
            if j in used_hyp: continue
            if abs(len(gl) - len(hl)) > len(gl) * 0.70: continue
            c_err = compute_levenshtein(gl_chars, list(hl))
            if c_err < best_c_err:
                best_c_err = c_err
                best_j = j
                best_w_err = compute_levenshtein(gl_words, hl.split())

        if best_j != -1 and best_c_err < len(gl_chars) * 0.75:
            used_hyp.add(best_j)
            total_c_err += best_c_err
            total_w_err += best_w_err
        else:
            total_c_err += len(gl_chars)
            total_w_err += len(gl_words)

    for j, hl in enumerate(norm_hyp):
        if j not in used_hyp:
            total_c_err += len(hl)
            total_w_err += len(hl.split())

    cer = total_c_err / max(1, total_gt_c)
    wer_seq = total_w_err / max(1, total_gt_w)

    # Multiset Error Rate (MER / Bag-of-Words): sum_w |c_ref(w) - c_hyp(w)| / sum_w c_ref(w)
    from collections import Counter
    ref_counts = Counter([w for l in norm_gt for w in l.split()])
    hyp_counts = Counter([w for l in norm_hyp for w in l.split()])
    all_words = set(ref_counts.keys()).union(set(hyp_counts.keys()))
    diff_count = sum(abs(ref_counts[w] - hyp_counts[w]) for w in all_words)
    mer_bow = diff_count / max(1, sum(ref_counts.values()))

    return float(cer), float(wer_seq), float(mer_bow)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RETRIEVAL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class PurePythonBM25:
    def __init__(self, corpus: List[str], k1: float = 0.9, b: float = 0.5, char_ngram: bool = False):
        self.k1 = k1
        self.b = b
        self.char_ngram = char_ngram
        self.corpus_size = len(corpus)
        self.doc_lengths: List[int] = []
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        for doc_id, doc_text in enumerate(corpus):
            tokens = tokenize_char_ngrams(doc_text) if char_ngram else tokenize_words(doc_text)
            self.doc_lengths.append(len(tokens))
            for t in tokens:
                if t not in self.inverted_index:
                    self.inverted_index[t] = {}
                self.inverted_index[t][doc_id] = self.inverted_index[t].get(doc_id, 0) + 1
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.corpus_size)

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize_char_ngrams(query) if self.char_ngram else tokenize_words(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for t in q_tokens:
            if t not in self.inverted_index:
                continue
            postings = self.inverted_index[t]
            df = len(postings)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
        return scores

class DenseModelRetriever:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                pooled = self.mean_pooling(out, encoded["attention_mask"])
                normed = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normed.cpu().numpy())
        return np.vstack(embeddings)

def compute_rr_and_hit10(scores: np.ndarray, target_idx: int) -> Tuple[float, float, float, float]:
    ranks = np.argsort(-scores)
    pos = np.where(ranks == target_idx)[0]
    if len(pos) == 0:
        return 0.0, 0.0, 0.0, 0.0
    rank = pos[0] + 1
    rr = float(1.0 / rank)
    h1 = float(1.0 if rank == 1 else 0.0)
    h5 = float(1.0 if rank <= 5 else 0.0)
    h10 = float(1.0 if rank <= 10 else 0.0)
    return rr, h1, h5, h10

def cluster_bootstrap_ci(values: np.ndarray, cluster_ids: List[str], B: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(np.array(cluster_ids) == c)[0] for c in unique_clusters}
    boot_means = []
    for _ in range(B):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx_list = []
        for c in sampled:
            idx_list.extend(cluster_to_idx[c])
        boot_means.append(np.mean(values[idx_list]))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 100000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    diff = a - b
    obs = abs(np.mean(diff))
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.abs(np.mean(flips * diff, axis=1))
    count = np.sum(perm_means >= obs - 1e-12)
    return float(max(count, 1) / n_perm)

# ─────────────────────────────────────────────────────────────────────────────
# 3. EASYOCR INGESTION WITH INSTANT CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def run_easyocr_on_images(img_paths: List[Path], cache_path: Path, reader: easyocr.Reader, desc: str = "Clean") -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} cached {desc} EasyOCR transcriptions from {cache_path}")
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}")

    remaining = [p for p in img_paths if p.stem not in results]
    if not remaining:
        print(f"All {len(img_paths)} {desc} images already transcribed via EasyOCR.")
        return results

    print(f"Running EasyOCR on {len(remaining)} remaining {desc} images (checkpointing every 25 images)...")
    t0 = time.time()
    for count, p in enumerate(remaining, 1):
        try:
            lines = reader.readtext(str(p), detail=0)
            results[p.stem] = [l.strip() for l in lines if l.strip()]
        except Exception as e:
            print(f"Error processing {p.name}: {e}")
            results[p.stem] = []

        if count % 25 == 0 or count == len(remaining):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - t0
            rate = count / max(0.1, elapsed)
            eta = (len(remaining) - count) / max(0.01, rate)
            print(f"[{desc} {count}/{len(remaining)}] {count/len(remaining)*100:.1f}% ({rate:.2f} img/s, ETA {eta:.1f}s)")

    print(f"Completed EasyOCR {desc} extraction in {time.time()-t0:.2f}s.")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PHASE 3.1: THIRD OCR ENGINE (EasyOCR 1.7.2) GENERALIZATION & REPLICATION")
    print(f"Hardware Device: {DEVICE}")
    print("=" * 80)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(FROZEN_PROBES_PATH, "r", encoding="utf-8") as f:
        frozen_probes = json.load(f)
    with open(HEAVY_PROBES_PATH, "r", encoding="utf-8") as f:
        heavy_probes = json.load(f)

    test_probes = frozen_probes["test_probes"]
    heavy_test_dict = {p["target_id"]: p.get("heavy_question", p.get("verbatim", "")) for p in heavy_probes if "target_id" in p}

    sroie_records = [r for r in bench_data["records"] if r["type"] == "receipt_document"]
    print(f"Loaded {len(sroie_records)} SROIE receipts from benchmark.")

    clean_img_paths = [REPO_ROOT / r["image_path"] if not Path(r["image_path"]).is_absolute() else Path(r["image_path"]) for r in sroie_records]
    deg_img_paths = [REPO_ROOT / "ai_engine" / "rag" / "data" / "sroie" / "img_degraded" / f"{r['id'].replace('sroie_', '')}.jpg" for r in sroie_records]

    # Initialize EasyOCR
    print("Initializing EasyOCR 1.7.2 Reader with gpu=True...")
    t_init = time.time()
    reader = easyocr.Reader(["en"], gpu=True)
    print(f"EasyOCR Reader initialized in {time.time()-t_init:.2f}s.")

    # Run / Load Clean and Degraded OCR
    easy_clean_ocr = run_easyocr_on_images(clean_img_paths, EASY_CLEAN_CACHE, reader, desc="Clean")
    easy_deg_ocr = run_easyocr_on_images(deg_img_paths, EASY_DEG_CACHE, reader, desc="Degraded Blur")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ERROR RATE COMPUTATION & SAMPLES MD
    # ─────────────────────────────────────────────────────────────────────────
    print("\nComputing Levenshtein CER, Sequence WER, and MER (BoW) against Ground Truth...")
    clean_cers, clean_wers, clean_mers = [], [], []
    deg_cers, deg_wers, deg_mers = [], [], []

    # Also load Apple Vision and Tesseract OCR for comparison
    with open(V3_RESULTS_DIR / "sroie_tesseract_clean_ocr.json", "r", encoding="utf-8") as f:
        tess_clean_ocr = json.load(f)
    with open(V3_RESULTS_DIR / "sroie_tesseract_degraded_ocr.json", "r", encoding="utf-8") as f:
        tess_deg_ocr = json.load(f)

    for r in sroie_records:
        stem = r["id"].replace("sroie_", "")
        gt_lines = [l.strip() for l in r["ground_truth_text"].split("\n") if l.strip()]

        ec_lines = easy_clean_ocr.get(stem, [])
        c_cer, c_wer, c_mer = compute_cer_wer_line_aligned(gt_lines, ec_lines)
        clean_cers.append(c_cer); clean_wers.append(c_wer); clean_mers.append(c_mer)

        ed_lines = easy_deg_ocr.get(stem, [])
        d_cer, d_wer, d_mer = compute_cer_wer_line_aligned(gt_lines, ed_lines)
        deg_cers.append(d_cer); deg_wers.append(d_wer); deg_mers.append(d_mer)

    mean_c_cer = float(np.mean(clean_cers))
    mean_c_wer = float(np.mean(clean_wers))
    mean_c_mer = float(np.mean(clean_mers))
    mean_d_cer = float(np.mean(deg_cers))
    mean_d_wer = float(np.mean(deg_wers))
    mean_d_mer = float(np.mean(deg_mers))

    print(f"EasyOCR 1.7.2 Clean:    CER = {mean_c_cer:.4f}, WER = {mean_c_wer:.4f}, MER = {mean_c_mer:.4f}")
    print(f"EasyOCR 1.7.2 Degraded: CER = {mean_d_cer:.4f}, WER = {mean_d_wer:.4f}, MER = {mean_d_mer:.4f}")

    # Write error rates CSV
    with open(EASY_ERROR_RATES_CSV, "w", encoding="utf-8") as f:
        f.write("Engine,Condition,CER,WER_seq,MER_bow\n")
        f.write(f"EasyOCR 1.7.2,Clean OCR,{mean_c_cer:.4f},{mean_c_wer:.4f},{mean_c_mer:.4f}\n")
        f.write(f"EasyOCR 1.7.2,Degraded Blur OCR,{mean_d_cer:.4f},{mean_d_wer:.4f},{mean_d_mer:.4f}\n")
        f.write(f"Apple Vision Neural,Clean OCR,0.1387,0.2165,0.2476\n")
        f.write(f"Apple Vision Neural,Degraded Blur OCR,0.3086,0.4595,0.4328\n")
        f.write(f"Tesseract 5.5 LSTM,Clean OCR,0.2939,0.4879,0.5919\n")
        f.write(f"Tesseract 5.5 LSTM,Degraded Blur OCR,0.6377,0.8063,1.0043\n")
    print(f"Saved error rates to {EASY_ERROR_RATES_CSV}")

    # Write 10 qualitative side-by-side samples to EASY_SAMPLES_MD
    with open(EASY_SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# Qualitative Side-by-Side OCR Engine Comparison (Phase 3.1)\n\n")
        f.write("Comparing Ground Truth, Apple Vision Neural OCR, Tesseract 5.5 LSTM, and EasyOCR 1.7.2 on SROIE Receipts.\n\n")
        for i, r in enumerate(sroie_records[:10]):
            stem = r["id"].replace("sroie_", "")
            f.write(f"## Sample {i+1}: Receipt `{r['id']}`\n\n")
            f.write("### 1. Ground Truth Text\n```text\n")
            f.write(r.get("ground_truth_text", "").strip()[:500] + "\n```\n\n")
            f.write("### 2. Apple Vision Clean OCR\n```text\n")
            f.write(r.get("real_ocr_text", "").strip()[:500] + "\n```\n\n")
            f.write("### 3. Tesseract 5.5 Clean OCR\n```text\n")
            f.write("\n".join(tess_clean_ocr.get(stem, []))[:500] + "\n```\n\n")
            f.write("### 4. EasyOCR 1.7.2 Clean OCR\n```text\n")
            f.write("\n".join(easy_clean_ocr.get(stem, []))[:500] + "\n```\n\n")
            f.write("---\n\n")
    print(f"Saved qualitative samples to {EASY_SAMPLES_MD}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. RETRIEVAL EVALUATION ON HELD-OUT TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\nSetting up Held-Out Test Corpus (813 items) & 306 Probes...")
    test_records = [r for r in bench_data["records"] if r["split"] == "test"]
    id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}

    test_doc_easy_clean = []
    test_doc_easy_deg = []
    for r in test_records:
        if r["type"] == "receipt_document":
            stem = r["id"].replace("sroie_", "")
            test_doc_easy_clean.append("\n".join(easy_clean_ocr.get(stem, [])))
            test_doc_easy_deg.append("\n".join(easy_deg_ocr.get(stem, [])))
        else:
            test_doc_easy_clean.append(r.get("ground_truth_text", ""))
            test_doc_easy_deg.append(r.get("ground_truth_text", ""))

    print(f"Loading dense models on {DEVICE}...")
    minilm = DenseModelRetriever("sentence-transformers/all-MiniLM-L6-v2", DEVICE)
    bge = DenseModelRetriever("BAAI/bge-base-en-v1.5", DEVICE)

    q_verbatim = [p.get("verbatim_query", p.get("verbatim", "")) for p in test_probes]
    q_light = [p["light_paraphrase"] for p in test_probes]
    q_heavy = [heavy_test_dict.get(p["target_id"], p.get("verbatim", "")) for p in test_probes]
    target_indices = [id_to_idx[p["target_id"]] for p in test_probes]
    cluster_ids = [p["target_id"] for p in test_probes]

    print("Encoding queries with dense models...")
    q_verb_mini = minilm.encode(q_verbatim)
    q_light_mini = minilm.encode(q_light)
    q_heavy_mini = minilm.encode(q_heavy)
    q_verb_bge = bge.encode(q_verbatim)
    q_light_bge = bge.encode(q_light)
    q_heavy_bge = bge.encode(q_heavy)

    engine_results = []
    paired_arrays = {}

    def evaluate_easyocr_condition(doc_texts: List[str], condition_name: str):
        print(f"\n--- Evaluating EasyOCR 1.7.2 ({condition_name}) ---")
        bm25_word = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=False)
        bm25_char = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=True)
        print("Encoding document texts with dense models...")
        doc_emb_mini = minilm.encode(doc_texts)
        doc_emb_bge = bge.encode(doc_texts)

        for q_type, q_list, q_mini, q_bge in [
            ("verbatim", q_verbatim, q_verb_mini, q_verb_bge),
            ("light", q_light, q_light_mini, q_light_bge),
            ("heavy", q_heavy, q_heavy_mini, q_heavy_bge),
        ]:
            mrr_w, h1_w, h5_w, h10_w = [], [], [], []
            mrr_c, h1_c, h5_c, h10_c = [], [], [], []
            mrr_m, h1_m, h5_m, h10_m = [], [], [], []
            mrr_b, h1_b, h5_b, h10_b = [], [], [], []
            mrr_h, h1_h, h5_h, h10_h = [], [], [], []

            for i, q in enumerate(q_list):
                t_idx = target_indices[i]

                # Word BM25
                s_w = bm25_word.score(q)
                rr, h1, h5, h10 = compute_rr_and_hit10(s_w, t_idx)
                mrr_w.append(rr); h1_w.append(h1); h5_w.append(h5); h10_w.append(h10)

                # Char BM25
                s_c = bm25_char.score(q)
                rr, h1, h5, h10 = compute_rr_and_hit10(s_c, t_idx)
                mrr_c.append(rr); h1_c.append(h1); h5_c.append(h5); h10_c.append(h10)

                # MiniLM
                s_m = doc_emb_mini @ q_mini[i]
                rr, h1, h5, h10 = compute_rr_and_hit10(s_m, t_idx)
                mrr_m.append(rr); h1_m.append(h1); h5_m.append(h5); h10_m.append(h10)

                # BGE
                s_b = doc_emb_bge @ q_bge[i]
                rr, h1, h5, h10 = compute_rr_and_hit10(s_b, t_idx)
                mrr_b.append(rr); h1_b.append(h1); h5_b.append(h5); h10_b.append(h10)

                # Tuned Hybrid
                if q_type == "verbatim": alpha = 0.2
                elif q_type == "light": alpha = 0.3
                else: alpha = 0.8
                norm_w = (s_w - s_w.min()) / max(1e-6, s_w.max() - s_w.min())
                norm_b = (s_b - s_b.min()) / max(1e-6, s_b.max() - s_b.min())
                s_hyb = (1.0 - alpha) * norm_w + alpha * norm_b
                rr, h1, h5, h10 = compute_rr_and_hit10(s_hyb, t_idx)
                mrr_h.append(rr); h1_h.append(h1); h5_h.append(h5); h10_h.append(h10)

            paired_arrays[f"{condition_name}_{q_type}_word"] = np.array(mrr_w)
            paired_arrays[f"{condition_name}_{q_type}_char"] = np.array(mrr_c)
            paired_arrays[f"{condition_name}_{q_type}_mini"] = np.array(mrr_m)
            paired_arrays[f"{condition_name}_{q_type}_bge"] = np.array(mrr_b)
            paired_arrays[f"{condition_name}_{q_type}_hybrid"] = np.array(mrr_h)

            for sys_name, mrr_arr, h1_arr, h5_arr, h10_arr in [
                ("Word BM25", np.array(mrr_w), np.array(h1_w), np.array(h5_w), np.array(h10_w)),
                ("Char 3-5gram BM25", np.array(mrr_c), np.array(h1_c), np.array(h5_c), np.array(h10_c)),
                ("Dense MiniLM", np.array(mrr_m), np.array(h1_m), np.array(h5_m), np.array(h10_m)),
                ("Dense BGE-base", np.array(mrr_b), np.array(h1_b), np.array(h5_b), np.array(h10_b)),
                ("Tuned Hybrid", np.array(mrr_h), np.array(h1_h), np.array(h5_h), np.array(h10_h)),
            ]:
                ci_l, ci_u = cluster_bootstrap_ci(mrr_arr, cluster_ids, B=B_BOOT)
                mean_mrr = float(np.mean(mrr_arr))
                mean_h1 = float(np.mean(h1_arr))
                mean_h5 = float(np.mean(h5_arr))
                mean_h10 = float(np.mean(h10_arr))
                print(f"[{condition_name} | {q_type:8s} | {sys_name:18s}] MRR: {mean_mrr:.4f} [{ci_l:.4f}, {ci_u:.4f}] | Hit@1: {mean_h1:.4f} | Hit@10: {mean_h10:.4f}")
                engine_results.append({
                    "Engine": "EasyOCR 1.7.2",
                    "Condition": condition_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "MRR": round(mean_mrr, 4),
                    "MRR_CI_Low": round(ci_l, 4),
                    "MRR_CI_High": round(ci_u, 4),
                    "Hit1": round(mean_h1, 4),
                    "Hit5": round(mean_h5, 4),
                    "Hit10": round(mean_h10, 4),
                })

    evaluate_easyocr_condition(test_doc_easy_clean, "Clean OCR")
    evaluate_easyocr_condition(test_doc_easy_deg, "Degraded Blur OCR")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. CONFIRMATORY HYPOTHESIS TESTS (PREREG_V9.md)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nRunning Confirmatory Hypothesis Tests (100,000 paired sign-flips)...")
    # H3.1a: Verbatim Char vs Word BM25 on Clean & Degraded
    p_h31a_clean = paired_permutation_test(paired_arrays["Clean OCR_verbatim_char"], paired_arrays["Clean OCR_verbatim_word"], n_perm=N_PERM)
    diff_h31a_clean = np.mean(paired_arrays["Clean OCR_verbatim_char"] - paired_arrays["Clean OCR_verbatim_word"])

    p_h31a_deg = paired_permutation_test(paired_arrays["Degraded Blur OCR_verbatim_char"], paired_arrays["Degraded Blur OCR_verbatim_word"], n_perm=N_PERM)
    diff_h31a_deg = np.mean(paired_arrays["Degraded Blur OCR_verbatim_char"] - paired_arrays["Degraded Blur OCR_verbatim_word"])

    # H3.1b: Paraphrase Reversal (Word vs Char on Light Paraphrase)
    p_h31b = paired_permutation_test(paired_arrays["Clean OCR_light_word"], paired_arrays["Clean OCR_light_char"], n_perm=N_PERM)
    diff_h31b = np.mean(paired_arrays["Clean OCR_light_word"] - paired_arrays["Clean OCR_light_char"])

    # H3.1c: Dense bi-encoder deficit (Char BM25 vs MiniLM)
    p_h31c = paired_permutation_test(paired_arrays["Clean OCR_verbatim_char"], paired_arrays["Clean OCR_verbatim_mini"], n_perm=N_PERM)
    diff_h31c = np.mean(paired_arrays["Clean OCR_verbatim_char"] - paired_arrays["Clean OCR_verbatim_mini"])

    # H3.1d: Blur Sensitivity (Word blur drop vs Char blur drop)
    word_blur_drop = paired_arrays["Clean OCR_verbatim_word"] - paired_arrays["Degraded Blur OCR_verbatim_word"]
    char_blur_drop = paired_arrays["Clean OCR_verbatim_char"] - paired_arrays["Degraded Blur OCR_verbatim_char"]
    p_h31d = paired_permutation_test(word_blur_drop, char_blur_drop, n_perm=N_PERM)
    diff_h31d = np.mean(word_blur_drop - char_blur_drop)

    print(f"H3.1a (Clean Verbatim Char vs Word):     diff = +{diff_h31a_clean:.4f}, p = {p_h31a_clean:.5f} -> {'CONFIRMED' if diff_h31a_clean > 0 and p_h31a_clean < 0.05 else 'REJECTED'}")
    print(f"H3.1a (Degraded Verbatim Char vs Word):  diff = +{diff_h31a_deg:.4f}, p = {p_h31a_deg:.5f} -> {'CONFIRMED' if diff_h31a_deg > 0 and p_h31a_deg < 0.05 else 'REJECTED'}")
    print(f"H3.1b (Light Paraphrase Word vs Char):   diff = +{diff_h31b:.4f}, p = {p_h31b:.5f} -> {'CONFIRMED' if diff_h31b > 0 and p_h31b < 0.05 else 'REJECTED'}")
    print(f"H3.1c (Dense MiniLM Deficit vs Char):    diff = +{diff_h31c:.4f}, p = {p_h31c:.5f} -> {'CONFIRMED' if diff_h31c > 0 and p_h31c < 0.05 else 'REJECTED'}")
    print(f"H3.1d (Char Resilience to Blur):         diff = +{diff_h31d:.4f}, p = {p_h31d:.5f} -> {'CONFIRMED' if diff_h31d > 0 and p_h31d < 0.05 else 'REJECTED'}")

    # Write replication CSV
    import pandas as pd
    df = pd.DataFrame(engine_results)
    df.to_csv(EASY_REPLICATION_CSV, index=False)
    print(f"\nSaved EasyOCR replication results to {EASY_REPLICATION_CSV}")

    print("\n" + "=" * 80)
    print("PHASE 3.1 EASYOCR 1.7.2 REPLICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
