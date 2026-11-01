#!/usr/bin/env python3
"""
scripts/reconcile_funsd_b08.py
Standardizes and harmonizes FUNSD Form Benchmark Matrix for Table X(b)
across Ground Truth, Apple Vision Clean, Tesseract Clean, EasyOCR Clean, and EasyOCR Blur.
Uses standard Information Retrieval evaluation:
  - If score(target) <= 0 (target is unretrieved / no query terms match), RR = 0, Hit@10 = 0.
  - If score(target) > 0, rank = standard rank with average tie-breaking.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "v3"

class FastBM25:
    def __init__(self, corpus, k1=0.9, b=0.5, char_ngram=False):
        self.k1 = k1
        self.b = b
        self.char_ngram = char_ngram
        self.doc_lens = []
        self.doc_term_freqs = []
        self.df = {}
        self.N = len(corpus)
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_term_freqs.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        self.idfs = {t: np.log((self.N - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()}

    def _tokenize(self, text):
        if self.char_ngram:
            text = text.lower()
            ngrams = []
            for n in (3, 4, 5):
                for i in range(len(text) - n + 1):
                    ngrams.append(text[i:i+n])
            return ngrams if ngrams else ([text] if text else [])
        else:
            return re.findall(r"\w+", text.lower())

    def score(self, query):
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N)
        for t in q_tokens:
            if t in self.idfs:
                idf = self.idfs[t]
                for d_idx, tf_dict in enumerate(self.doc_term_freqs):
                    tf = tf_dict.get(t, 0)
                    if tf > 0:
                        denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[d_idx] / self.avgdl))
                        scores[d_idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

def compute_ranking_metrics(scores: np.ndarray, target_idx: int) -> tuple:
    target_score = scores[target_idx]
    if target_score <= 1e-9:
        # Document was unretrieved (zero terms matched)
        return 0.0, 0.0
    # Standard rank with average tie-breaking
    strictly_better = np.sum(scores > target_score)
    tied = np.sum(np.isclose(scores, target_score))
    rank = strictly_better + 1 + (tied - 1) / 2.0
    rr = 1.0 / rank
    hit10 = 1.0 if rank <= 10.0 else 0.0
    return rr, hit10

def cluster_bootstrap_ci(arr, cluster_ids, B=2000, seed=42):
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
        boot_means.append(np.mean(arr[idx_list]))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def main():
    # Load probes
    with open(RESULTS_DIR / "funsd_probes.json", "r", encoding="utf-8") as f:
        probes_data = json.load(f)
    test_probes = probes_data["test_probes"]

    # Load OCR transcriptions
    with open(RESULTS_DIR / "funsd_apple_vision_ocr.json", "r", encoding="utf-8") as f:
        apple_ocr = json.load(f)
    with open(RESULTS_DIR / "funsd_tesseract_ocr.json", "r", encoding="utf-8") as f:
        tess_ocr = json.load(f)
    tess_deg_cache = RESULTS_DIR / "funsd_tesseract_degraded_ocr.json"
    tess_deg_ocr = {}
    if tess_deg_cache.exists():
        with open(tess_deg_cache, "r", encoding="utf-8") as f:
            tess_deg_ocr = json.load(f)
    with open(RESULTS_DIR / "funsd_easyocr_clean_ocr.json", "r", encoding="utf-8") as f:
        easy_clean_ocr = json.load(f)
    with open(RESULTS_DIR / "funsd_easyocr_degraded_ocr.json", "r", encoding="utf-8") as f:
        easy_deg_ocr = json.load(f)

    # Load Ground Truth lines
    funsd_dir = REPO_ROOT / "ai_engine" / "rag" / "data" / "funsd" / "dataset"
    all_anns = list((funsd_dir / "testing_data" / "annotations").glob("*.json")) + \
               list((funsd_dir / "training_data" / "annotations").glob("*.json"))
    gt_dict = {}
    for p in all_anns:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        form_lines = [item["text"] for item in data.get("form", []) if item.get("text", "").strip()]
        gt_dict[p.stem] = form_lines

    # Load MS-COCO distractors with ground truth caption fallback
    bench_path = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "public_benchmark_v2.json"
    coco_distractors = []
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        coco_distractors = [r.get("ground_truth_text", "") for r in bench_data["records"] 
                            if r["type"] == "natural_image" and r["split"] == "test"][:500]

    test_stems = [p["target_id"].replace("funsd_", "") for p in test_probes]
    unique_forms = sorted(list(set(test_stems)))
    stem_to_idx = {s: i for i, s in enumerate(unique_forms)}
    target_indices = [stem_to_idx[s] for s in test_stems]
    clusters = test_stems

    q_verb = [p["verbatim_query"] for p in test_probes]
    q_light = [p["light_paraphrase"] for p in test_probes]

    all_records = []

    for pool_type in ["Closed 99-Form Pool", "600-Item Distractor Pool"]:
        distractors = [] if pool_type == "Closed 99-Form Pool" else coco_distractors
        
        conditions = {
            "Ground Truth": [ "\n".join(gt_dict.get(s, [])) for s in unique_forms ] + distractors,
            "Apple Vision Clean": [ "\n".join(apple_ocr.get(s, [])) for s in unique_forms ] + distractors,
            "Tesseract Clean": [ "\n".join(tess_ocr.get(s, [])) for s in unique_forms ] + distractors,
            "Tesseract Blur": [ "\n".join(tess_deg_ocr.get(s, [])) for s in unique_forms ] + distractors,
            "EasyOCR Clean": [ "\n".join(easy_clean_ocr.get(s, [])) for s in unique_forms ] + distractors,
            "EasyOCR Blur": [ "\n".join(easy_deg_ocr.get(s, [])) for s in unique_forms ] + distractors,
        }

        print(f"\n================================================================================")
        print(f"POOLS: {pool_type} (Total docs: {len(unique_forms) + len(distractors)})")
        print(f"================================================================================")
        header = f"{'Condition':<20} | {'Query':<10} | {'Word BM25':<24} | {'Char BM25':<24} | {'Hit@10 Word':<12} | {'Hit@10 Char':<12}"
        print(header)
        print("-" * len(header))

        for cond_name, corpus in conditions.items():
            bm_word = FastBM25(corpus, k1=0.9, b=0.5, char_ngram=False)
            bm_char = FastBM25(corpus, k1=0.9, b=0.5, char_ngram=True)
            
            for q_type, q_list in [("Verbatim", q_verb), ("Light", q_light)]:
                rrs_w, rrs_c = [], []
                h10_w, h10_c = [], []
                for i, q in enumerate(q_list):
                    target = target_indices[i]
                    # word
                    sc_w = bm_word.score(q)
                    rr_w, h_w = compute_ranking_metrics(sc_w, target)
                    rrs_w.append(rr_w); h10_w.append(h_w)
                    # char
                    sc_c = bm_char.score(q)
                    rr_c, h_c = compute_ranking_metrics(sc_c, target)
                    rrs_c.append(rr_c); h10_c.append(h_c)
                
                arr_w = np.array(rrs_w)
                arr_c = np.array(rrs_c)
                ci_w = cluster_bootstrap_ci(arr_w, clusters)
                ci_c = cluster_bootstrap_ci(arr_c, clusters)
                
                m_w = f"{np.mean(arr_w):.4f} [{ci_w[0]:.3f}, {ci_w[1]:.3f}]"
                m_c = f"{np.mean(arr_c):.4f} [{ci_c[0]:.3f}, {ci_c[1]:.3f}]"
                hit_w = f"{np.mean(h10_w):.4f}"
                hit_c = f"{np.mean(h10_c):.4f}"
                print(f"{cond_name:<20} | {q_type:<10} | {m_w:<24} | {m_c:<24} | {hit_w:<12} | {hit_c:<12}")

                all_records.append({
                    "Pool": pool_type, "Condition": cond_name, "Query_Type": q_type,
                    "Word_BM25_MRR": float(np.mean(arr_w)), "Word_BM25_CI_Low": ci_w[0], "Word_BM25_CI_High": ci_w[1],
                    "Char_BM25_MRR": float(np.mean(arr_c)), "Char_BM25_CI_Low": ci_c[0], "Char_BM25_CI_High": ci_c[1],
                    "Word_Hit10": float(np.mean(h10_w)), "Char_Hit10": float(np.mean(h10_c)),
                })

    df_out = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / "funsd_standardized_pools_b08.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved standardized metrics to {out_csv}")

if __name__ == "__main__":
    main()
