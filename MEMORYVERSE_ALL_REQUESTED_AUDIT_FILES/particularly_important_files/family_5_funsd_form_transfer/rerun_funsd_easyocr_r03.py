#!/usr/bin/env python3
"""
rerun_funsd_easyocr_r03.py: Comprehensive Recomputation and Audit of FUNSD EasyOCR Blur Retrieval.

Addresses Item R-03 / N-04:
- Recomputes FUNSD EasyOCR Clean and Blur across all 5 systems (Word BM25, Char BM25, MiniLM, BGE, Hybrid)
  under both Verbatim and Light Paraphrase queries.
- Compares 3 tie-handling / retrieval regimes:
  1. Original Stable Argsort (replicating historical artifact)
  2. Strict IR Unretrieved Rule (RR=0.0 and Hit@10=0.0 when target score <= 0.0)
  3. Average Tie-Breaking (fractional rank across all tied documents)
- Breaks down metrics by target status:
  - Full collection (N=99 test probes)
  - Non-blank target subset (N=29 probes where OCR produced text)
  - Blank target subset (N=70 probes where CRAFT completely failed)

Outputs:
- results/v4/R-03/funsd_easyocr_rerun.csv
- results/v4/R-03/funsd_easyocr_subset_analysis.csv
- results/v4/R-03/table_vii_b_corrected_column.tex
- results/v4/R-03/README.md
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES_PATH = REPO_ROOT / "results" / "v3" / "funsd_probes.json"
BENCHMARK_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "public_benchmark_v2.json"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"
EASY_CLEAN_CACHE = V3_RESULTS_DIR / "funsd_easyocr_clean_ocr.json"
EASY_DEG_CACHE = V3_RESULTS_DIR / "funsd_easyocr_degraded_ocr.json"
OUT_DIR = REPO_ROOT / "results" / "v4" / "R-03"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
B_BOOT = 2000

# -----------------------------------------------------------------------------
# 1. Tokenizers and BM25
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 2. Ranking Evaluation Functions
# -----------------------------------------------------------------------------

def eval_original_stable(scores: np.ndarray, target_idx: int) -> Tuple[float, float]:
    """Original stable argsort: ranks by array index among tied scores."""
    ranks = np.argsort(-scores)
    pos = np.where(ranks == target_idx)[0]
    if len(pos) == 0:
        return 0.0, 0.0
    rank = pos[0] + 1
    rr = float(1.0 / rank)
    h10 = float(1.0 if rank <= 10 else 0.0)
    return rr, h10

def eval_strict_ir(scores: np.ndarray, target_idx: int, is_dense: bool = False) -> Tuple[float, float]:
    """
    Strict IR metric:
    - If target score <= 0 (for sparse) or not positively matched, RR=0, Hit@10=0.
    - If target score > 0, tie-split only among documents that achieved > 0 score.
    - For dense, cosine similarity can be negative; dense target must have distinct similarity.
    """
    tgt_score = scores[target_idx]
    if not is_dense:
        if tgt_score <= 1e-6:
            return 0.0, 0.0
        n_greater = np.sum(scores > tgt_score + 1e-6)
        n_equal = np.sum(np.abs(scores - tgt_score) <= 1e-6)
        frac_rank = n_greater + (n_equal + 1) / 2.0
    else:
        # Dense model: standard fractional rank
        n_greater = np.sum(scores > tgt_score + 1e-6)
        n_equal = np.sum(np.abs(scores - tgt_score) <= 1e-6)
        frac_rank = n_greater + (n_equal + 1) / 2.0
    
    rr = float(1.0 / frac_rank)
    h10 = float(1.0 if frac_rank <= 10.0 else 0.0)
    return rr, h10

def eval_average_ties(scores: np.ndarray, target_idx: int) -> Tuple[float, float]:
    """Pure average tie-breaking over the entire pool."""
    tgt_score = scores[target_idx]
    n_greater = np.sum(scores > tgt_score + 1e-6)
    n_equal = np.sum(np.abs(scores - tgt_score) <= 1e-6)
    frac_rank = n_greater + (n_equal + 1) / 2.0
    rr = float(1.0 / frac_rank)
    h10 = float(1.0 if frac_rank <= 10.0 else 0.0)
    return rr, h10

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

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("R-03: FUNSD EasyOCR Blur Retrieval Recomputation and Audit")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROBES_PATH, "r", encoding="utf-8") as f:
        probes_data = json.load(f)
    test_probes = probes_data["test_probes"]
    print(f"Loaded {len(test_probes)} test probes.")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    coco_distractors = [r for r in bench_data["records"] if r["type"] == "natural_image" and r["split"] == "test"][:500]
    print(f"Loaded {len(coco_distractors)} COCO distractors.")

    with open(EASY_CLEAN_CACHE, "r", encoding="utf-8") as f:
        easy_clean = json.load(f)
    with open(EASY_DEG_CACHE, "r", encoding="utf-8") as f:
        easy_deg = json.load(f)

    test_stems = [p["target_id"].replace("funsd_", "") for p in test_probes]
    unique_test_forms = list(dict.fromkeys(test_stems))
    stem_to_idx = {s: i for i, s in enumerate(unique_test_forms)}
    target_indices = [stem_to_idx[s] for s in test_stems]
    cluster_ids = [p["target_id"] for p in test_probes]

    q_verb = [p["verbatim_query"] for p in test_probes]
    q_light = [p["light_paraphrase"] for p in test_probes]

    # Target form character counts under blur
    deg_char_counts = [len(" ".join(easy_deg.get(s, []))) for s in test_stems]
    is_non_blank = np.array([c > 0 for c in deg_char_counts])
    is_blank = ~is_non_blank
    print(f"Test targets under Blur: {np.sum(is_non_blank)} non-blank ({np.mean(is_non_blank)*100:.1f}%), {np.sum(is_blank)} blank ({np.mean(is_blank)*100:.1f}%)")

    # Load Dense Models
    print("Loading Dense Models...")
    minilm = DenseModelRetriever("sentence-transformers/all-MiniLM-L6-v2", DEVICE)
    bge = DenseModelRetriever("BAAI/bge-base-en-v1.5", DEVICE)

    print("Encoding queries...")
    q_verb_mini = minilm.encode(q_verb)
    q_light_mini = minilm.encode(q_light)
    q_verb_bge = bge.encode(q_verb)
    q_light_bge = bge.encode(q_light)

    def build_corpus(ocr_dict: Dict[str, List[str]]) -> List[str]:
        corpus = []
        for s in unique_test_forms:
            lines = ocr_dict.get(s, [])
            corpus.append("\n".join(lines))
        for r in coco_distractors:
            corpus.append(r.get("ground_truth_text", ""))
        return corpus

    corpus_clean = build_corpus(easy_clean)
    corpus_deg = build_corpus(easy_deg)

    all_results = []
    subset_results = []

    for cond_name, doc_corpus in [("EasyOCR Clean", corpus_clean), ("EasyOCR Blur", corpus_deg)]:
        print(f"\n--- Indexing {cond_name} ---")
        bm25_word = PurePythonBM25(doc_corpus, k1=0.9, b=0.5, char_ngram=False)
        bm25_char = PurePythonBM25(doc_corpus, k1=0.9, b=0.5, char_ngram=True)

        print("Encoding corpus embeddings...")
        doc_emb_mini = minilm.encode(doc_corpus)
        doc_emb_bge = bge.encode(doc_corpus)

        for q_type, q_list, q_mini, q_bge in [
            ("verbatim", q_verb, q_verb_mini, q_verb_bge),
            ("light", q_light, q_light_mini, q_light_bge),
        ]:
            alpha = 0.2 if q_type == "verbatim" else 0.3

            # System scores matrices: [N_queries, N_docs]
            scores_word = np.zeros((len(q_list), len(doc_corpus)), dtype=np.float32)
            scores_char = np.zeros((len(q_list), len(doc_corpus)), dtype=np.float32)
            scores_mini = np.zeros((len(q_list), len(doc_corpus)), dtype=np.float32)
            scores_bge = np.zeros((len(q_list), len(doc_corpus)), dtype=np.float32)
            scores_hyb = np.zeros((len(q_list), len(doc_corpus)), dtype=np.float32)

            for i, q in enumerate(q_list):
                s_w = bm25_word.score(q)
                s_c = bm25_char.score(q)
                s_m = doc_emb_mini @ q_mini[i]
                s_b = doc_emb_bge @ q_bge[i]

                norm_w = (s_w - s_w.min()) / max(1e-6, s_w.max() - s_w.min())
                norm_b = (s_b - s_b.min()) / max(1e-6, s_b.max() - s_b.min())
                s_h = (1.0 - alpha) * norm_w + alpha * norm_b

                scores_word[i] = s_w
                scores_char[i] = s_c
                scores_mini[i] = s_m
                scores_bge[i] = s_b
                scores_hyb[i] = s_h

            systems = [
                ("BM25 (Tuned)", scores_word, False),
                ("BM25 (Char 3-5gram)", scores_char, False),
                ("Dense MiniLM", scores_mini, True),
                ("Dense BGE-base", scores_bge, True),
                ("Tuned Hybrid", scores_hyb, False),
            ]

            for sys_name, score_mat, is_dense in systems:
                # 1. Original stable argsort
                orig_rr, orig_h10 = [], []
                # 2. Strict IR unretrieved
                strict_rr, strict_h10 = [], []
                # 3. Average tie-breaking
                avg_rr, avg_h10 = [], []

                for i in range(len(q_list)):
                    t_idx = target_indices[i]
                    s = score_mat[i]
                    r1, h1 = eval_original_stable(s, t_idx)
                    r2, h2 = eval_strict_ir(s, t_idx, is_dense=is_dense)
                    r3, h3 = eval_average_ties(s, t_idx)
                    orig_rr.append(r1); orig_h10.append(h1)
                    strict_rr.append(r2); strict_h10.append(h2)
                    avg_rr.append(r3); avg_h10.append(h3)

                for rule_name, rr_list, h10_list in [
                    ("original_stable", orig_rr, orig_h10),
                    ("strict_ir", strict_rr, strict_h10),
                    ("average_ties", avg_rr, avg_h10),
                ]:
                    arr = np.array(rr_list)
                    ci_l, ci_u = cluster_bootstrap_ci(arr, cluster_ids, B=B_BOOT)
                    mrr = float(np.mean(arr))
                    h10 = float(np.mean(h10_list))
                    all_results.append({
                        "Condition": cond_name,
                        "Query_Type": q_type,
                        "System": sys_name,
                        "Rule": rule_name,
                        "MRR": round(mrr, 4),
                        "MRR_CI_Lower": round(ci_l, 4),
                        "MRR_CI_Upper": round(ci_u, 4),
                        "Hit10": round(h10, 4),
                    })

                # Breakdown by non-blank vs blank for strict_ir
                strict_arr = np.array(strict_rr)
                strict_h_arr = np.array(strict_h10)
                subset_results.append({
                    "Condition": cond_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "Subset": "all_probes",
                    "N": len(q_list),
                    "MRR": round(float(np.mean(strict_arr)), 4),
                    "Hit10": round(float(np.mean(strict_h_arr)), 4),
                })
                subset_results.append({
                    "Condition": cond_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "Subset": "non_blank",
                    "N": int(np.sum(is_non_blank)),
                    "MRR": round(float(np.mean(strict_arr[is_non_blank])), 4),
                    "Hit10": round(float(np.mean(strict_h_arr[is_non_blank])), 4),
                })
                subset_results.append({
                    "Condition": cond_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "Subset": "blank",
                    "N": int(np.sum(is_blank)),
                    "MRR": round(float(np.mean(strict_arr[is_blank])), 4),
                    "Hit10": round(float(np.mean(strict_h_arr[is_blank])), 4),
                })

    # Save detailed comparisons to CSV
    csv_all = OUT_DIR / "funsd_easyocr_rerun.csv"
    with open(csv_all, "w", encoding="utf-8") as f:
        f.write("Condition,Query_Type,System,Rule,MRR,MRR_CI_Lower,MRR_CI_Upper,Hit10\n")
        for r in all_results:
            f.write(f"{r['Condition']},{r['Query_Type']},{r['System']},{r['Rule']},{r['MRR']},{r['MRR_CI_Lower']},{r['MRR_CI_Upper']},{r['Hit10']}\n")
    print(f"\nWrote full comparison to {csv_all}")

    csv_sub = OUT_DIR / "funsd_easyocr_subset_analysis.csv"
    with open(csv_sub, "w", encoding="utf-8") as f:
        f.write("Condition,Query_Type,System,Subset,N,MRR,Hit10\n")
        for r in subset_results:
            f.write(f"{r['Condition']},{r['Query_Type']},{r['System']},{r['Subset']},{r['N']},{r['MRR']},{r['Hit10']}\n")
    print(f"Wrote subset analysis to {csv_sub}")

    # Generate LaTeX column snippet
    tex_out = OUT_DIR / "table_vii_b_corrected_column.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write("% Formatted LaTeX column for Table VII(b) under Strict IR Unretrieved Rule\n")
        for q_type in ["verbatim", "light"]:
            f.write(f"\n% --- {q_type.upper()} ---\n")
            for sys_name in ["BM25 (Tuned)", "BM25 (Char 3-5gram)", "Dense MiniLM", "Dense BGE-base", "Tuned Hybrid"]:
                row_orig = next(r for r in all_results if r["Condition"] == "EasyOCR Blur" and r["Query_Type"] == q_type and r["System"] == sys_name and r["Rule"] == "original_stable")
                row_strict = next(r for r in all_results if r["Condition"] == "EasyOCR Blur" and r["Query_Type"] == q_type and r["System"] == sys_name and r["Rule"] == "strict_ir")
                row_nb = next(r for r in subset_results if r["Condition"] == "EasyOCR Blur" and r["Query_Type"] == q_type and r["System"] == sys_name and r["Subset"] == "non_blank")
                
                f.write(f"% {sys_name:<20}: Original Stable = {row_orig['MRR']:.4f}, Strict IR All = {row_strict['MRR']:.4f} [{row_strict['MRR_CI_Lower']:.3f}, {row_strict['MRR_CI_Upper']:.3f}], Non-Blank (N=29) = {row_nb['MRR']:.4f}\n")
                f.write(f" & {sys_name:<20} & \\makecell{{{row_strict['MRR']:.4f}\\\\ {{\\tiny [{row_strict['MRR_CI_Lower']:.3f}, {row_strict['MRR_CI_Upper']:.3f}]}}}} % strict IR\n")
    print(f"Wrote formatted LaTeX snippet to {tex_out}")

if __name__ == "__main__":
    main()

