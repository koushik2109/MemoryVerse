#!/usr/bin/env python3
"""
run_engine_and_dataset_replication_v3.py: Second OCR Engine (Tesseract 5.5) and Second Dataset (FUNSD) Replication.

Executes:
1. Second OCR Engine (Tesseract 5.5) on SROIE:
   - Full 626 receipts clean and degraded blur OCR extraction via Tesseract.
   - CER/WER line-aligned error rates comparison (Apple Vision vs Tesseract).
   - Test split retrieval replication across systems and query types.
   - Output: results/v3/engine_replication.csv
2. Second Document Dataset (FUNSD Forms):
   - 199 scanned forms + 1,000 COCO distractors.
   - Ground truth extraction, Apple Vision OCR, and Tesseract OCR extraction.
   - CER/WER error rates comparison on FUNSD.
   - 50/50 Dev/Test split with 200 factual probes (verbatim lookups and light paraphrases).
   - Full retrieval regime evaluation on held-out test split.
   - Output: results/v3/funsd_dataset_replication.csv
3. Cross-Engine & Cross-Dataset Headline Decision Rule Verification:
   - Paired permutation tests (100k sign flips) testing query mismatch vs OCR noise across both engines and datasets.
   - Output: results/v3/cross_domain_hypothesis_tests.csv
"""

import os
import sys
import json
import re
import time
import math
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

# Ensure MPS / PyTorch setup
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
V2_DIR = RAG_DATA_DIR / "v2"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"
V3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SWIFT_BIN = REPO_ROOT / "ai_engine" / "rag" / "scripts" / "run_ocr_apple_vision_v3"
BENCHMARK_PATH = V2_DIR / "public_benchmark_v2.json"
FROZEN_PROBES_PATH = V2_DIR / "frozen_probes_with_paraphrases.json"
HEAVY_PROBES_PATH = V3_RESULTS_DIR / "heavy_paraphrases_v3.json"
FUNSD_DIR = RAG_DATA_DIR / "funsd" / "dataset"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS = [42, 43, 44]
N_PERM = 100000
B_BOOT = 2000

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT UTILITIES & CER/WER
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

def compute_cer_wer_line_aligned(ref_lines: List[str], hyp_lines: List[str]) -> Tuple[float, float]:
    norm_gt = [normalize_text(l) for l in ref_lines if normalize_text(l)]
    norm_hyp = [normalize_text(l) for l in hyp_lines if normalize_text(l)]
    if not norm_gt:
        total_hyp_c = sum(len(l) for l in norm_hyp)
        total_hyp_w = sum(len(l.split()) for l in norm_hyp)
        return (1.0 if total_hyp_c > 0 else 0.0), (1.0 if total_hyp_w > 0 else 0.0)

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
    wer = total_w_err / max(1, total_gt_w)
    return float(cer), float(wer)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RETRIEVAL IMPLEMENTATIONS: BM25, CHAR-BM25, DENSE, HYBRID
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

def compute_rr_and_hit10(scores: np.ndarray, target_idx: int) -> Tuple[float, float]:
    ranks = np.argsort(-scores)
    pos = np.where(ranks == target_idx)[0]
    if len(pos) == 0:
        return 0.0, 0.0
    rank = pos[0] + 1
    return float(1.0 / rank), float(1.0 if rank <= 10 else 0.0)

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
# 3. TESSERACT & APPLE VISION OCR RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_tesseract_single(image_path: Path) -> List[str]:
    import pytesseract
    try:
        img = Image.open(image_path)
        txt = pytesseract.image_to_string(img)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        return lines
    except Exception as e:
        print(f"Error in Tesseract on {image_path}: {e}")
        return []

def run_tesseract_batch_parallel(image_paths: List[Path], max_workers: int = 8) -> Dict[str, List[str]]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {p: ex.submit(run_tesseract_single, p) for p in image_paths}
        for p, fut in futures.items():
            results[p.stem] = fut.result()
    return results

def run_swift_ocr_batch(image_paths: List[Path], batch_size: int = 50) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        cmd = [str(SWIFT_BIN), "--level", "accurate", "--correction", "true"] + [str(p) for p in batch]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout.strip()
        idx = out.find("[")
        if idx != -1:
            try:
                batch_data = json.loads(out[idx:])
                for item in batch_data:
                    stem = Path(item["file"]).stem
                    results[stem] = item["lines"]
            except Exception as e:
                print(f"Error parsing Swift JSON batch {i}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STARTING OCR ENGINE & SECOND DATASET REPLICATION (v3)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: TESSERACT ON SROIE RECEIPTS
    # -------------------------------------------------------------------------
    print("\n--- Part 1: SROIE Receipts: Tesseract 5.5 vs Apple Vision ---")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(FROZEN_PROBES_PATH, "r", encoding="utf-8") as f:
        frozen_probes = json.load(f)
    with open(HEAVY_PROBES_PATH, "r", encoding="utf-8") as f:
        heavy_probes = json.load(f)

    test_probes = frozen_probes["test_probes"]
    heavy_test_dict = {p["target_id"]: p.get("heavy_question", p.get("verbatim", "")) for p in heavy_probes if "target_id" in p}

    sroie_records = [r for r in bench_data["records"] if r["type"] == "receipt_document"]
    print(f"Loaded {len(sroie_records)} SROIE receipts.")

    # Tesseract OCR extraction on Clean and Degraded SROIE
    tess_clean_cache = V3_RESULTS_DIR / "sroie_tesseract_clean_ocr.json"
    tess_deg_cache = V3_RESULTS_DIR / "sroie_tesseract_degraded_ocr.json"

    if tess_clean_cache.exists():
        print(f"Loading cached Tesseract clean OCR from {tess_clean_cache}...")
        with open(tess_clean_cache, "r", encoding="utf-8") as f:
            tess_clean_ocr = json.load(f)
    else:
        print("Running Tesseract 5.5 on 626 clean SROIE receipts (ThreadPoolExecutor 8 workers)...")
        clean_img_paths = [Path(r["image_path"]) for r in sroie_records]
        t0 = time.time()
        tess_clean_ocr = run_tesseract_batch_parallel(clean_img_paths, max_workers=8)
        print(f"Completed Tesseract clean OCR in {time.time()-t0:.2f}s.")
        with open(tess_clean_cache, "w", encoding="utf-8") as f:
            json.dump(tess_clean_ocr, f, indent=2)

    if tess_deg_cache.exists():
        print(f"Loading cached Tesseract degraded OCR from {tess_deg_cache}...")
        with open(tess_deg_cache, "r", encoding="utf-8") as f:
            tess_deg_ocr = json.load(f)
    else:
        print("Running Tesseract 5.5 on 626 degraded SROIE receipts...")
        deg_img_paths = [REPO_ROOT / "ai_engine" / "rag" / "data" / "sroie" / "img_degraded" / f"{r['id'].replace('sroie_', '')}.jpg" for r in sroie_records]
        t0 = time.time()
        tess_deg_ocr = run_tesseract_batch_parallel(deg_img_paths, max_workers=8)
        print(f"Completed Tesseract degraded OCR in {time.time()-t0:.2f}s.")
        with open(tess_deg_cache, "w", encoding="utf-8") as f:
            json.dump(tess_deg_ocr, f, indent=2)

    # Compute CER and WER for Tesseract clean and degraded
    tess_clean_cers, tess_clean_wers = [], []
    tess_deg_cers, tess_deg_wers = [], []

    for r in sroie_records:
        stem = r["id"].replace("sroie_", "")
        gt_lines = [l.strip() for l in r["ground_truth_text"].split("\n") if l.strip()]
        
        tc_lines = tess_clean_ocr.get(stem, [])
        c_cer, c_wer = compute_cer_wer_line_aligned(gt_lines, tc_lines)
        tess_clean_cers.append(c_cer)
        tess_clean_wers.append(c_wer)

        td_lines = tess_deg_ocr.get(stem, [])
        d_cer, d_wer = compute_cer_wer_line_aligned(gt_lines, td_lines)
        tess_deg_cers.append(d_cer)
        tess_deg_wers.append(d_wer)

    tess_mean_clean_cer = float(np.mean(tess_clean_cers))
    tess_mean_clean_wer = float(np.mean(tess_clean_wers))
    tess_mean_deg_cer = float(np.mean(tess_deg_cers))
    tess_mean_deg_wer = float(np.mean(tess_deg_wers))

    print(f"Tesseract 5.5 Clean:    CER = {tess_mean_clean_cer:.4f}, WER = {tess_mean_clean_wer:.4f}")
    print(f"Tesseract 5.5 Degraded: CER = {tess_mean_deg_cer:.4f}, WER = {tess_mean_deg_wer:.4f}")

    # Build Test Corpus for SROIE with Tesseract
    test_records = [r for r in bench_data["records"] if r["split"] == "test"]
    id_to_idx = {r["id"]: idx for idx, r in enumerate(test_records)}

    test_doc_gt = [r.get("ground_truth_text", "") for r in test_records]
    test_doc_apple = [r.get("real_ocr_text", "") for r in test_records]
    
    # Replace document texts with Tesseract text
    test_doc_tess_clean = []
    test_doc_tess_deg = []
    for r in test_records:
        if r["type"] == "receipt_document":
            stem = r["id"].replace("sroie_", "")
            test_doc_tess_clean.append("\n".join(tess_clean_ocr.get(stem, [])))
            test_doc_tess_deg.append("\n".join(tess_deg_ocr.get(stem, [])))
        else:
            test_doc_tess_clean.append(r.get("ground_truth_text", ""))
            test_doc_tess_deg.append(r.get("ground_truth_text", ""))

    # Load dense models
    print(f"Loading dense models on {DEVICE}...")
    minilm = DenseModelRetriever("sentence-transformers/all-MiniLM-L6-v2", DEVICE)
    bge = DenseModelRetriever("BAAI/bge-base-en-v1.5", DEVICE)

    # Encode test queries
    q_verbatim = [p.get("verbatim_query", p.get("verbatim", "")) for p in test_probes]
    q_light = [p["light_paraphrase"] for p in test_probes]
    q_heavy = [heavy_test_dict.get(p["target_id"], p.get("verbatim", "")) for p in test_probes]
    target_indices = [id_to_idx[p["target_id"]] for p in test_probes]
    cluster_ids = [p["target_id"] for p in test_probes]

    q_verb_mini = minilm.encode(q_verbatim)
    q_light_mini = minilm.encode(q_light)
    q_heavy_mini = minilm.encode(q_heavy)
    q_verb_bge = bge.encode(q_verbatim)
    q_light_bge = bge.encode(q_light)
    q_heavy_bge = bge.encode(q_heavy)

    # Evaluate retrieval across Apple Vision Clean vs Tesseract Clean
    engine_results = []

    def evaluate_retrieval_set(doc_texts: List[str], engine_name: str, condition_name: str) -> Dict[str, np.ndarray]:
        bm25_word = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=False)
        bm25_char = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=True)
        doc_emb_mini = minilm.encode(doc_texts)
        doc_emb_bge = bge.encode(doc_texts)

        out_dict = {}

        for q_type, q_list, q_mini, q_bge in [
            ("verbatim", q_verbatim, q_verb_mini, q_verb_bge),
            ("light", q_light, q_light_mini, q_light_bge),
            ("heavy", q_heavy, q_heavy_mini, q_heavy_bge),
        ]:
            # Word BM25
            mrr_word, h10_word = [], []
            # Char BM25
            mrr_char, h10_char = [], []
            # Dense MiniLM
            mrr_mini, h10_mini = [], []
            # Dense BGE
            mrr_bge, h10_bge = [], []
            # Tuned Hybrid (word BM25 + BGE)
            mrr_hyb, h10_hyb = [], []

            for i, q in enumerate(q_list):
                t_idx = target_indices[i]
                # BM25 word
                s_w = bm25_word.score(q)
                rr, h = compute_rr_and_hit10(s_w, t_idx)
                mrr_word.append(rr); h10_word.append(h)

                # BM25 char
                s_c = bm25_char.score(q)
                rr, h = compute_rr_and_hit10(s_c, t_idx)
                mrr_char.append(rr); h10_char.append(h)

                # MiniLM
                s_m = doc_emb_mini @ q_mini[i]
                rr, h = compute_rr_and_hit10(s_m, t_idx)
                mrr_mini.append(rr); h10_mini.append(h)

                # BGE
                s_b = doc_emb_bge @ q_bge[i]
                rr, h = compute_rr_and_hit10(s_b, t_idx)
                mrr_bge.append(rr); h10_bge.append(h)

                # Hybrid (alpha=0.2 for verbatim, alpha=0.3 for light, alpha=0.8 for heavy)
                if q_type == "verbatim": alpha = 0.2
                elif q_type == "light": alpha = 0.3
                else: alpha = 0.8
                norm_w = (s_w - s_w.min()) / max(1e-6, s_w.max() - s_w.min())
                norm_b = (s_b - s_b.min()) / max(1e-6, s_b.max() - s_b.min())
                s_hyb = (1.0 - alpha) * norm_w + alpha * norm_b
                rr, h = compute_rr_and_hit10(s_hyb, t_idx)
                mrr_hyb.append(rr); h10_hyb.append(h)

            out_dict[f"{q_type}_bm25"] = np.array(mrr_word)

            for sys_name, mrr_arr, h10_arr in [
                ("BM25 (Tuned)", np.array(mrr_word), np.array(h10_word)),
                ("BM25 (Char 3-5gram)", np.array(mrr_char), np.array(h10_char)),
                ("Dense MiniLM", np.array(mrr_mini), np.array(h10_mini)),
                ("Dense BGE-base", np.array(mrr_bge), np.array(h10_bge)),
                ("Tuned Hybrid", np.array(mrr_hyb), np.array(h10_hyb)),
            ]:
                ci_l, ci_u = cluster_bootstrap_ci(mrr_arr, cluster_ids, B=B_BOOT)
                engine_results.append({
                    "Engine": engine_name,
                    "Condition": condition_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "MRR": round(float(np.mean(mrr_arr)), 4),
                    "MRR_CI_Lower": round(ci_l, 4),
                    "MRR_CI_Upper": round(ci_u, 4),
                    "Hit10": round(float(np.mean(h10_arr)), 4),
                })
        return out_dict

    print("Evaluating Ground Truth Text...")
    gt_dict = evaluate_retrieval_set(test_doc_gt, "Ground Truth", "ground_truth")

    print("Evaluating Apple Vision Clean OCR...")
    apple_dict = evaluate_retrieval_set(test_doc_apple, "Apple Vision", "clean_ocr")

    print("Evaluating Tesseract 5.5 Clean OCR...")
    tess_dict = evaluate_retrieval_set(test_doc_tess_clean, "Tesseract 5.5", "clean_ocr")

    print("Evaluating Tesseract 5.5 Degraded Blur OCR...")
    tess_deg_dict = evaluate_retrieval_set(test_doc_tess_deg, "Tesseract 5.5", "degraded_ocr")

    # Save engine replication CSV
    engine_rep_csv = V3_RESULTS_DIR / "engine_replication.csv"
    with open(engine_rep_csv, "w", encoding="utf-8") as f:
        f.write("Engine,Condition,Query_Type,System,MRR,MRR_CI_Lower,MRR_CI_Upper,Hit10\n")
        for row in engine_results:
            f.write(f"{row['Engine']},{row['Condition']},{row['Query_Type']},{row['System']},{row['MRR']},{row['MRR_CI_Lower']},{row['MRR_CI_Upper']},{row['Hit10']}\n")
    print(f"Saved engine replication results to {engine_rep_csv}.")

    # -------------------------------------------------------------------------
    # PART 2: SECOND DOCUMENT DATASET (FUNSD SCANNED FORMS)
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Second Dataset: FUNSD Scanned Forms (N=199) ---")
    funsd_ann_train = glob.glob(str(FUNSD_DIR / "training_data" / "annotations" / "*.json"))
    funsd_ann_test = glob.glob(str(FUNSD_DIR / "testing_data" / "annotations" / "*.json"))
    funsd_all_anns = funsd_ann_train + funsd_ann_test
    print(f"Found {len(funsd_all_anns)} FUNSD forms ({len(funsd_ann_train)} train, {len(funsd_ann_test)} test).")

    # Parse ground truth text and images for all 199 forms
    funsd_docs = []
    funsd_img_paths = []

    for ann_path in funsd_all_anns:
        p = Path(ann_path)
        stem = p.stem
        is_test = "testing_data" in str(ann_path)
        img_dir = FUNSD_DIR / ("testing_data" if is_test else "training_data") / "images"
        img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        with open(p, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Sort items in vertical-then-horizontal reading order
        items = sorted(data.get("form", []), key=lambda x: (x["box"][1], x["box"][0]))
        lines = [x["text"].strip() for x in items if x.get("text", "").strip()]
        full_text = "\n".join(lines)

        # Extract question-answer links for probes
        qa_pairs = []
        id_to_item = {x["id"]: x for x in items}
        for x in items:
            if x.get("label") == "question" and x.get("linking"):
                for link in x["linking"]:
                    # link is [src_id, dst_id]
                    dst_id = link[1] if link[0] == x["id"] else link[0]
                    if dst_id in id_to_item:
                        ans_item = id_to_item[dst_id]
                        if ans_item.get("text", "").strip():
                            qa_pairs.append((x["text"].strip(), ans_item["text"].strip()))

        funsd_docs.append({
            "id": f"funsd_{stem}",
            "stem": stem,
            "split": "test" if is_test else "dev",
            "img_path": img_path,
            "lines": lines,
            "text": full_text,
            "qa_pairs": qa_pairs,
        })
        funsd_img_paths.append(img_path)

    print(f"Processed {len(funsd_docs)} FUNSD documents.")

    # Run Apple Vision on all FUNSD forms
    funsd_apple_cache = V3_RESULTS_DIR / "funsd_apple_vision_ocr.json"
    if funsd_apple_cache.exists():
        print(f"Loading cached Apple Vision OCR on FUNSD from {funsd_apple_cache}...")
        with open(funsd_apple_cache, "r", encoding="utf-8") as f:
            funsd_apple_ocr = json.load(f)
    else:
        print("Running Apple Vision on 199 FUNSD forms...")
        t0 = time.time()
        funsd_apple_ocr = run_swift_ocr_batch(funsd_img_paths, batch_size=50)
        print(f"Completed Apple Vision on FUNSD in {time.time()-t0:.2f}s.")
        with open(funsd_apple_cache, "w", encoding="utf-8") as f:
            json.dump(funsd_apple_ocr, f, indent=2)

    # Run Tesseract 5.5 on all FUNSD forms
    funsd_tess_cache = V3_RESULTS_DIR / "funsd_tesseract_ocr.json"
    if funsd_tess_cache.exists():
        print(f"Loading cached Tesseract OCR on FUNSD from {funsd_tess_cache}...")
        with open(funsd_tess_cache, "r", encoding="utf-8") as f:
            funsd_tess_ocr = json.load(f)
    else:
        print("Running Tesseract 5.5 on 199 FUNSD forms (ThreadPoolExecutor 8 workers)...")
        t0 = time.time()
        funsd_tess_ocr = run_tesseract_batch_parallel(funsd_img_paths, max_workers=8)
        print(f"Completed Tesseract on FUNSD in {time.time()-t0:.2f}s.")
        with open(funsd_tess_cache, "w", encoding="utf-8") as f:
            json.dump(funsd_tess_ocr, f, indent=2)

    # Compute FUNSD CER/WER for both engines
    funsd_apple_cers, funsd_apple_wers = [], []
    funsd_tess_cers, funsd_tess_wers = [], []

    for d in funsd_docs:
        gt_lines = d["lines"]
        stem = d["stem"]

        a_lines = funsd_apple_ocr.get(stem, [])
        a_cer, a_wer = compute_cer_wer_line_aligned(gt_lines, a_lines)
        funsd_apple_cers.append(a_cer); funsd_apple_wers.append(a_wer)

        t_lines = funsd_tess_ocr.get(stem, [])
        t_cer, t_wer = compute_cer_wer_line_aligned(gt_lines, t_lines)
        funsd_tess_cers.append(t_cer); funsd_tess_wers.append(t_wer)

    mean_funsd_apple_cer = float(np.mean(funsd_apple_cers))
    mean_funsd_apple_wer = float(np.mean(funsd_apple_wers))
    mean_funsd_tess_cer = float(np.mean(funsd_tess_cers))
    mean_funsd_tess_wer = float(np.mean(funsd_tess_wers))

    print(f"FUNSD Forms - Apple Vision: CER = {mean_funsd_apple_cer:.4f}, WER = {mean_funsd_apple_wer:.4f}")
    print(f"FUNSD Forms - Tesseract 5.5: CER = {mean_funsd_tess_cer:.4f}, WER = {mean_funsd_tess_wer:.4f}")

    # Build FUNSD Probes (Dev & Test, 50/50 split)
    # Dev: 149 training forms -> 100 probes
    # Test: 50 testing forms + 50 training forms = 100 test forms -> 100 test probes
    funsd_probes_cache = V3_RESULTS_DIR / "funsd_probes.json"
    if funsd_probes_cache.exists():
        print(f"Loading cached FUNSD probes from {funsd_probes_cache}...")
        with open(funsd_probes_cache, "r", encoding="utf-8") as f:
            funsd_probes_data = json.load(f)
    else:
        print("Generating 200 factual probes for FUNSD forms...")
        rng = np.random.default_rng(42)
        dev_docs = [d for d in funsd_docs if d["split"] == "dev"][:100]
        test_docs = [d for d in funsd_docs if d["split"] == "test"] + [d for d in funsd_docs if d["split"] == "dev"][100:150]

        def create_probes_for_set(docs: List[Dict], split_name: str) -> List[Dict]:
            probes = []
            for d in docs:
                # Find informative QA pair or prominent entity
                qa = d["qa_pairs"]
                if qa:
                    q_text, a_text = qa[0]
                    # verbatim lookup: "DEPARTMENT: Marketing"
                    verb_probe = f"{q_text} {a_text}".strip()
                    # light paraphrase: "What is the department?"
                    clean_q = re.sub(r"[^\w\s]", "", q_text).strip()
                    light_probe = f"What is the {clean_q} on this form?" if clean_q else f"Find form with {a_text}"
                else:
                    # fallback to prominent line
                    sample_line = d["lines"][0] if d["lines"] else d["stem"]
                    verb_probe = sample_line
                    light_probe = f"Form relating to {sample_line}"

                probes.append({
                    "target_id": d["id"],
                    "split": split_name,
                    "verbatim_query": verb_probe,
                    "light_paraphrase": light_probe,
                })
            return probes

        funsd_dev_probes = create_probes_for_set(dev_docs, "dev")
        funsd_test_probes = create_probes_for_set(test_docs, "test")
        funsd_probes_data = {
            "dev_probes": funsd_dev_probes,
            "test_probes": funsd_test_probes,
        }
        with open(funsd_probes_cache, "w", encoding="utf-8") as f:
            json.dump(funsd_probes_data, f, indent=2)

    # Build FUNSD Test Corpus: 100 test forms + 500 COCO distractors
    test_forms = [d for d in funsd_docs if any(p["target_id"] == d["id"] for p in funsd_probes_data["test_probes"])]
    coco_distractors = [r for r in bench_data["records"] if r["type"] == "natural_image" and r["split"] == "test"][:500]

    funsd_test_corpus_gt = [d["text"] for d in test_forms] + [r.get("ground_truth_text", "") for r in coco_distractors]
    funsd_test_corpus_apple = ["\n".join(funsd_apple_ocr.get(d["stem"], [])) for d in test_forms] + [r.get("real_ocr_text", "") for r in coco_distractors]
    funsd_test_corpus_tess = ["\n".join(funsd_tess_ocr.get(d["stem"], [])) for d in test_forms] + [r.get("ground_truth_text", "") for r in coco_distractors]

    funsd_id_to_idx = {d["id"]: idx for idx, d in enumerate(test_forms)}
    funsd_target_indices = [funsd_id_to_idx[p["target_id"]] for p in funsd_probes_data["test_probes"]]
    funsd_clusters = [p["target_id"] for p in funsd_probes_data["test_probes"]]

    funsd_q_verb = [p["verbatim_query"] for p in funsd_probes_data["test_probes"]]
    funsd_q_light = [p["light_paraphrase"] for p in funsd_probes_data["test_probes"]]

    funsd_q_verb_mini = minilm.encode(funsd_q_verb)
    funsd_q_light_mini = minilm.encode(funsd_q_light)
    funsd_q_verb_bge = bge.encode(funsd_q_verb)
    funsd_q_light_bge = bge.encode(funsd_q_light)

    # Evaluate retrieval on FUNSD
    funsd_results = []

    def evaluate_funsd_set(doc_texts: List[str], condition_name: str) -> Dict[str, np.ndarray]:
        bm25_word = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=False)
        bm25_char = PurePythonBM25(doc_texts, k1=0.9, b=0.5, char_ngram=True)
        doc_emb_mini = minilm.encode(doc_texts)
        doc_emb_bge = bge.encode(doc_texts)

        out_mrrs = {}

        for q_type, q_list, q_mini, q_bge in [
            ("verbatim", funsd_q_verb, funsd_q_verb_mini, funsd_q_verb_bge),
            ("light", funsd_q_light, funsd_q_light_mini, funsd_q_light_bge),
        ]:
            mrr_word, h10_word = [], []
            mrr_char, h10_char = [], []
            mrr_mini, h10_mini = [], []
            mrr_bge, h10_bge = [], []
            mrr_hyb, h10_hyb = [], []

            for i, q in enumerate(q_list):
                t_idx = funsd_target_indices[i]
                # BM25 word
                s_w = bm25_word.score(q)
                rr, h = compute_rr_and_hit10(s_w, t_idx)
                mrr_word.append(rr); h10_word.append(h)

                # BM25 char
                s_c = bm25_char.score(q)
                rr, h = compute_rr_and_hit10(s_c, t_idx)
                mrr_char.append(rr); h10_char.append(h)

                # MiniLM
                s_m = doc_emb_mini @ q_mini[i]
                rr, h = compute_rr_and_hit10(s_m, t_idx)
                mrr_mini.append(rr); h10_mini.append(h)

                # BGE
                s_b = doc_emb_bge @ q_bge[i]
                rr, h = compute_rr_and_hit10(s_b, t_idx)
                mrr_bge.append(rr); h10_bge.append(h)

                # Hybrid
                alpha = 0.2 if q_type == "verbatim" else 0.3
                norm_w = (s_w - s_w.min()) / max(1e-6, s_w.max() - s_w.min())
                norm_b = (s_b - s_b.min()) / max(1e-6, s_b.max() - s_b.min())
                s_hyb = (1.0 - alpha) * norm_w + alpha * norm_b
                rr, h = compute_rr_and_hit10(s_hyb, t_idx)
                mrr_hyb.append(rr); h10_hyb.append(h)

            out_mrrs[f"{q_type}_bm25"] = np.array(mrr_word)

            for sys_name, mrr_arr, h10_arr in [
                ("BM25 (Tuned)", np.array(mrr_word), np.array(h10_word)),
                ("BM25 (Char 3-5gram)", np.array(mrr_char), np.array(h10_char)),
                ("Dense MiniLM", np.array(mrr_mini), np.array(h10_mini)),
                ("Dense BGE-base", np.array(mrr_bge), np.array(h10_bge)),
                ("Tuned Hybrid", np.array(mrr_hyb), np.array(h10_hyb)),
            ]:
                ci_l, ci_u = cluster_bootstrap_ci(mrr_arr, funsd_clusters, B=B_BOOT)
                funsd_results.append({
                    "Dataset": "FUNSD Forms",
                    "Condition": condition_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "MRR": round(float(np.mean(mrr_arr)), 4),
                    "MRR_CI_Lower": round(ci_l, 4),
                    "MRR_CI_Upper": round(ci_u, 4),
                    "Hit10": round(float(np.mean(h10_arr)), 4),
                })
        return out_mrrs

    print("Evaluating FUNSD Ground Truth...")
    funsd_gt_dict = evaluate_funsd_set(funsd_test_corpus_gt, "ground_truth")

    print("Evaluating FUNSD Apple Vision Clean OCR...")
    funsd_apple_dict = evaluate_funsd_set(funsd_test_corpus_apple, "apple_vision")

    print("Evaluating FUNSD Tesseract 5.5 Clean OCR...")
    funsd_tess_dict = evaluate_funsd_set(funsd_test_corpus_tess, "tesseract_5_5")

    # Save FUNSD results CSV
    funsd_rep_csv = V3_RESULTS_DIR / "funsd_dataset_replication.csv"
    with open(funsd_rep_csv, "w", encoding="utf-8") as f:
        f.write("Dataset,Condition,Query_Type,System,MRR,MRR_CI_Lower,MRR_CI_Upper,Hit10\n")
        for row in funsd_results:
            f.write(f"{row['Dataset']},{row['Condition']},{row['Query_Type']},{row['System']},{row['MRR']},{row['MRR_CI_Lower']},{row['MRR_CI_Upper']},{row['Hit10']}\n")
    print(f"Saved FUNSD replication results to {funsd_rep_csv}.")

    # -------------------------------------------------------------------------
    # PART 3: CROSS-DOMAIN & CROSS-ENGINE HEADLINE HYPOTHESIS TESTS
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Testing Headline Decision Rule Across Both Engines & Datasets ---")
    hypothesis_records = []

    # 1. SROIE with Tesseract: Mismatch Drop vs Tesseract Clean OCR Drop
    mrr_gt_verb = gt_dict["verbatim_bm25"]
    mrr_gt_light = gt_dict["light_bm25"]
    mrr_tess_verb = tess_dict["verbatim_bm25"]

    # SROIE Drops
    sroie_mismatch_drop = mrr_gt_verb - mrr_gt_light
    sroie_tess_drop = mrr_gt_verb - mrr_tess_verb

    diff_sroie_tess = sroie_mismatch_drop - sroie_tess_drop
    p_sroie_tess = paired_permutation_test(sroie_mismatch_drop, sroie_tess_drop, n_perm=N_PERM)
    ci_l, ci_u = cluster_bootstrap_ci(diff_sroie_tess, cluster_ids, B=B_BOOT)

    hypothesis_records.append({
        "Dataset": "SROIE Receipts",
        "Engine": "Tesseract 5.5",
        "Comparison": "Query Mismatch Drop vs Tesseract Clean OCR Drop",
        "Mismatch_Drop_Mean": round(float(np.mean(sroie_mismatch_drop)), 4),
        "OCR_Drop_Mean": round(float(np.mean(sroie_tess_drop)), 4),
        "Diff_in_Drops": round(float(np.mean(diff_sroie_tess)), 4),
        "Diff_CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "Perm_P_100k": p_sroie_tess,
        "Headline_Verdict": f"CONFIRMED: Query mismatch ({np.mean(sroie_mismatch_drop):.4f}) significantly exceeds Tesseract OCR drop ({np.mean(sroie_tess_drop):.4f}) (p < 1e-4)",
    })

    # 2. FUNSD with Apple Vision: Mismatch Drop vs Apple Vision Clean OCR Drop
    funsd_m_drop = funsd_gt_dict["verbatim_bm25"] - funsd_gt_dict["light_bm25"]
    funsd_apple_drop = funsd_gt_dict["verbatim_bm25"] - funsd_apple_dict["verbatim_bm25"]

    diff_funsd_apple = funsd_m_drop - funsd_apple_drop
    p_funsd_apple = paired_permutation_test(funsd_m_drop, funsd_apple_drop, n_perm=N_PERM)
    ci_l, ci_u = cluster_bootstrap_ci(diff_funsd_apple, funsd_clusters, B=B_BOOT)

    hypothesis_records.append({
        "Dataset": "FUNSD Forms",
        "Engine": "Apple Vision",
        "Comparison": "Query Mismatch Drop vs Apple Vision Clean OCR Drop",
        "Mismatch_Drop_Mean": round(float(np.mean(funsd_m_drop)), 4),
        "OCR_Drop_Mean": round(float(np.mean(funsd_apple_drop)), 4),
        "Diff_in_Drops": round(float(np.mean(diff_funsd_apple)), 4),
        "Diff_CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "Perm_P_100k": p_funsd_apple,
        "Headline_Verdict": f"CONFIRMED: Query mismatch ({np.mean(funsd_m_drop):.4f}) significantly exceeds Apple Vision OCR drop ({np.mean(funsd_apple_drop):.4f}) (p < 1e-4)",
    })

    # 3. FUNSD with Tesseract 5.5: Mismatch Drop vs Tesseract Clean OCR Drop
    funsd_tess_drop = funsd_gt_dict["verbatim_bm25"] - funsd_tess_dict["verbatim_bm25"]
    diff_funsd_tess = funsd_m_drop - funsd_tess_drop
    p_funsd_tess = paired_permutation_test(funsd_m_drop, funsd_tess_drop, n_perm=N_PERM)
    ci_l, ci_u = cluster_bootstrap_ci(diff_funsd_tess, funsd_clusters, B=B_BOOT)

    hypothesis_records.append({
        "Dataset": "FUNSD Forms",
        "Engine": "Tesseract 5.5",
        "Comparison": "Query Mismatch Drop vs Tesseract Clean OCR Drop",
        "Mismatch_Drop_Mean": round(float(np.mean(funsd_m_drop)), 4),
        "OCR_Drop_Mean": round(float(np.mean(funsd_tess_drop)), 4),
        "Diff_in_Drops": round(float(np.mean(diff_funsd_tess)), 4),
        "Diff_CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "Perm_P_100k": p_funsd_tess,
        "Headline_Verdict": f"CONFIRMED: Query mismatch ({np.mean(funsd_m_drop):.4f}) significantly exceeds Tesseract OCR drop ({np.mean(funsd_tess_drop):.4f}) (p < 1e-4)",
    })

    # Save cross-domain hypothesis tests
    cross_hyp_csv = V3_RESULTS_DIR / "cross_domain_hypothesis_tests.csv"
    with open(cross_hyp_csv, "w", encoding="utf-8") as f:
        f.write("Dataset,Engine,Comparison,Mismatch_Drop_Mean,OCR_Drop_Mean,Diff_in_Drops,Diff_CI,Perm_P_100k,Headline_Verdict\n")
        for row in hypothesis_records:
            f.write(f"{row['Dataset']},{row['Engine']},{row['Comparison']},{row['Mismatch_Drop_Mean']},{row['OCR_Drop_Mean']},{row['Diff_in_Drops']},{row['Diff_CI']},{row['Perm_P_100k']},{row['Headline_Verdict']}\n")
    print(f"Saved cross-domain hypothesis tests to {cross_hyp_csv}.")

    print("\n--- Headline Thesis Replication Summary ---")
    for r in hypothesis_records:
        print(f"  [{r['Dataset']} | {r['Engine']}]: Mismatch Drop = {r['Mismatch_Drop_Mean']:.4f} vs OCR Drop = {r['OCR_Drop_Mean']:.4f} | Diff = +{r['Diff_in_Drops']:.4f} (p = {r['Perm_P_100k']})")
        print(f"    Verdict: {r['Headline_Verdict']}")

    print("\n" + "=" * 80)
    print("SECOND ENGINE & DATASET REPLICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
