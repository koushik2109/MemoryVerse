#!/usr/bin/env python3
"""
NEW_EXPERIMENT_SCRIPTS/run_experiment_1_date_ablation.py
MemoryVerse — Experiment 1: Transaction-Date Ablation on Human Queries

Objective:
Test whether including explicit transaction dates in human-authored receipt queries
improves retrieval of the intended receipt and suppresses same-merchant false transactions.

Systems evaluated:
1. Word BM25 (k1=1.5, b=0.75)
2. Character 3-5gram BM25 (k1=1.5, b=0.75)
3. Dense BGE-Base (BAAI/bge-base-en-v1.5)

Evaluated across:
- All eligible date-bearing human queries (N=56)
- Multi-receipt same-merchant subset (N=36)
- Single-receipt merchant control subset (N=20)
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Tokenization & BM25 Retriever
# ---------------------------------------------------------

def tokenize_words(text):
    return re.findall(r"\w+", text.lower())

def tokenize_char_ngrams(text, min_n=3, max_n=5):
    clean = re.sub(r"\s+", " ", text.lower().strip())
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(clean) - n + 1):
            ngrams.append(clean[i:i+n])
    return ngrams

class BM25Retriever:
    def __init__(self, corpus, tokenizer, k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_lens = []
        self.inverted_index = {}
        
        for doc_idx, doc in enumerate(corpus):
            tokens = tokenizer(doc)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = tf
                
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query):
        q_tokens = self.tokenizer(query)
        q_counts = Counter(q_tokens)
        scores = np.zeros(self.N, dtype=np.float32)
        for term, q_cnt in q_counts.items():
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in self.inverted_index[term].items():
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl))
                scores[doc_idx] += idf_val * (num / denom)
        return scores

# ---------------------------------------------------------
# 2. Statistical Testing Helpers
# ---------------------------------------------------------

def paired_permutation_test(a, b, num_permutations=100_000, seed=42):
    rng = np.random.RandomState(seed)
    diff = np.array(a) - np.array(b)
    obs = np.abs(np.mean(diff))
    if obs < 1e-9:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, len(diff)))
    perm_diffs = np.abs(np.mean(signs * diff, axis=1))
    p_val = (np.sum(perm_diffs >= obs) + 1.0) / (num_permutations + 1.0)
    return float(p_val)

def bootstrap_ci_diff(a, b, num_bootstrap=5_000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    diff = np.array(a) - np.array(b)
    indices = rng.randint(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(diff[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(boot_means, alpha * 100)
    high = np.percentile(boot_means, (1.0 - alpha) * 100)
    return float(low), float(high), float(np.mean(diff))

def mcnemar_test(hit_orig, hit_abl):
    # b: hit in orig, miss in abl; c: miss in orig, hit in abl
    b = sum(1 for o, a in zip(hit_orig, hit_abl) if o == 1 and a == 0)
    c = sum(1 for o, a in zip(hit_orig, hit_abl) if o == 0 and a == 1)
    if b + c == 0:
        return 0.0, 1.0
    # McNemar with Edwards continuity correction
    chi2 = (abs(b - c) - 1.0)**2 / (b + c)
    p_val = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p_val)

# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------

def main():
    repo_root = Path(".")
    
    # Load public benchmark test corpus
    bench_path = repo_root / "ai_engine/rag/data/v2/public_benchmark_v2.json"
    with open(bench_path) as f:
        bench = json.load(f)
        
    test_ids = set(bench.get("test_corpus_ids", []))
    test_corpus = [r for r in bench["records"] if r["id"] in test_ids]
    corpus_ids = [r["id"] for r in test_corpus]
    id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    
    # Extract OCR texts
    corpus_texts = [r.get("real_ocr_text", r.get("ground_truth_text", "")) for r in test_corpus]
    print(f"Loaded {len(test_corpus)} test receipts.")
    
    # Corpus merchant map
    all_records = {r["id"]: r for r in bench["records"]}
    merchant_counts = Counter()
    doc_merchants = {}
    for r in bench["records"]:
        ent = r.get("key_entities", {})
        comp = ent.get("company", "").strip().upper()
        if comp:
            merchant_counts[comp] += 1
            doc_merchants[r["id"]] = comp
        else:
            doc_merchants[r["id"]] = "UNKNOWN"

    # Initialize BM25 retrievers
    print("Building Word BM25 index...")
    bm25_word = BM25Retriever(corpus_texts, tokenize_words)
    print("Building Char 3-5gram BM25 index...")
    bm25_char = BM25Retriever(corpus_texts, tokenize_char_ngrams)
    
    # Initialize Dense BGE-Base retriever
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading BGE-Base model on {device}...")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device=device)
    print("Encoding test corpus with BGE-Base...")
    corpus_embs = bge_model.encode(corpus_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    # Load 400 human queries
    queries_path = repo_root / "data/human_queries_adjudicated.csv"
    df_queries = pd.read_csv(queries_path)
    p_df = df_queries[df_queries["query_type"] == "paraphrase"].copy()

    # Exact date excision patterns
    date_patterns = [
        (r",\s*dated\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", ""),
        (r"\s+dated\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", ""),
        (r"\s+dated\s+\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+\d{2,4})?", ""),
        (r"\s+on\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+\d{2,4})?", ""),
        (r"\s+bought\s+on\s+christmas", " bought"),
        (r"\s+on\s+christmas", ""),
        (r"\s+in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?", ""),
        (r"\s+on\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", ""),
    ]

    ablation_items = []
    for idx, r in p_df.iterrows():
        q_orig = r["query_string"]
        tid = r["original_single_target"]
        rec = all_records.get(tid, {})
        ent = rec.get("key_entities", {})
        merchant = ent.get("company", "").strip().upper()
        rec_date = ent.get("date", "").strip()
        num_merchant_docs = merchant_counts.get(merchant, 0)
        
        q_abl = q_orig
        removed_exprs = []
        for pat, repl in date_patterns:
            m = re.findall(pat, q_abl, flags=re.IGNORECASE)
            if m:
                removed_exprs.extend(m)
                q_abl = re.sub(pat, repl, q_abl, flags=re.IGNORECASE).strip()
                
        if removed_exprs:
            ablation_items.append({
                "row_idx": r["row_idx"],
                "target_id": tid,
                "merchant": merchant,
                "num_merchant_docs": num_merchant_docs,
                "receipt_date": rec_date,
                "original_query": q_orig,
                "ablated_query": q_abl,
                "removed_date_expression": "; ".join(removed_exprs).strip()
            })

    print(f"\nIdentified {len(ablation_items)} date-bearing queries out of {len(p_df)} human paraphrase queries.")
    
    # Encode all original and ablated queries with BGE-Base
    orig_queries = [it["original_query"] for it in ablation_items]
    abl_queries = [it["ablated_query"] for it in ablation_items]
    print("Encoding original and ablated queries with BGE-Base...")
    embs_orig = bge_model.encode(orig_queries, normalize_embeddings=True, show_progress_bar=False)
    embs_abl = bge_model.encode(abl_queries, normalize_embeddings=True, show_progress_bar=False)

    # Retrieval evaluation
    detailed_rows = []
    for i, it in enumerate(ablation_items):
        tid = it["target_id"]
        tgt_idx = id_to_idx.get(tid, -1)
        target_merchant = it["merchant"]
        
        # Word BM25
        w_scores_orig = bm25_word.score(it["original_query"])
        w_ranks_orig = np.argsort(-w_scores_orig)
        w_rank_orig = int(np.where(w_ranks_orig == tgt_idx)[0][0]) + 1 if tgt_idx in w_ranks_orig else 9999
        w_top1_orig = corpus_ids[w_ranks_orig[0]]
        w_same_merch_wrong_orig = 1 if (w_top1_orig != tid and doc_merchants.get(w_top1_orig) == target_merchant) else 0

        w_scores_abl = bm25_word.score(it["ablated_query"])
        w_ranks_abl = np.argsort(-w_scores_abl)
        w_rank_abl = int(np.where(w_ranks_abl == tgt_idx)[0][0]) + 1 if tgt_idx in w_ranks_abl else 9999
        w_top1_abl = corpus_ids[w_ranks_abl[0]]
        w_same_merch_wrong_abl = 1 if (w_top1_abl != tid and doc_merchants.get(w_top1_abl) == target_merchant) else 0

        # Char BM25
        c_scores_orig = bm25_char.score(it["original_query"])
        c_ranks_orig = np.argsort(-c_scores_orig)
        c_rank_orig = int(np.where(c_ranks_orig == tgt_idx)[0][0]) + 1 if tgt_idx in c_ranks_orig else 9999
        c_top1_orig = corpus_ids[c_ranks_orig[0]]
        c_same_merch_wrong_orig = 1 if (c_top1_orig != tid and doc_merchants.get(c_top1_orig) == target_merchant) else 0

        c_scores_abl = bm25_char.score(it["ablated_query"])
        c_ranks_abl = np.argsort(-c_scores_abl)
        c_rank_abl = int(np.where(c_ranks_abl == tgt_idx)[0][0]) + 1 if tgt_idx in c_ranks_abl else 9999
        c_top1_abl = corpus_ids[c_ranks_abl[0]]
        c_same_merch_wrong_abl = 1 if (c_top1_abl != tid and doc_merchants.get(c_top1_abl) == target_merchant) else 0

        # Dense BGE-Base
        b_scores_orig = np.dot(corpus_embs, embs_orig[i])
        b_ranks_orig = np.argsort(-b_scores_orig)
        b_rank_orig = int(np.where(b_ranks_orig == tgt_idx)[0][0]) + 1 if tgt_idx in b_ranks_orig else 9999
        b_top1_orig = corpus_ids[b_ranks_orig[0]]
        b_same_merch_wrong_orig = 1 if (b_top1_orig != tid and doc_merchants.get(b_top1_orig) == target_merchant) else 0

        b_scores_abl = np.dot(corpus_embs, embs_abl[i])
        b_ranks_abl = np.argsort(-b_scores_abl)
        b_rank_abl = int(np.where(b_ranks_abl == tgt_idx)[0][0]) + 1 if tgt_idx in b_ranks_abl else 9999
        b_top1_abl = corpus_ids[b_ranks_abl[0]]
        b_same_merch_wrong_abl = 1 if (b_top1_abl != tid and doc_merchants.get(b_top1_abl) == target_merchant) else 0

        detailed_rows.append({
            "target_id": tid,
            "merchant": target_merchant,
            "num_merchant_docs": it["num_merchant_docs"],
            "receipt_date": it["receipt_date"],
            "removed_date_expression": it["removed_date_expression"],
            "original_query": it["original_query"],
            "ablated_query": it["ablated_query"],
            # Word BM25
            "word_rank_orig": w_rank_orig,
            "word_rr_orig": 1.0 / w_rank_orig if w_rank_orig <= 813 else 0.0,
            "word_hit1_orig": 1 if w_rank_orig == 1 else 0,
            "word_hit5_orig": 1 if w_rank_orig <= 5 else 0,
            "word_hit10_orig": 1 if w_rank_orig <= 10 else 0,
            "word_same_merch_wrong_orig": w_same_merch_wrong_orig,
            "word_rank_abl": w_rank_abl,
            "word_rr_abl": 1.0 / w_rank_abl if w_rank_abl <= 813 else 0.0,
            "word_hit1_abl": 1 if w_rank_abl == 1 else 0,
            "word_hit5_abl": 1 if w_rank_abl <= 5 else 0,
            "word_hit10_abl": 1 if w_rank_abl <= 10 else 0,
            "word_same_merch_wrong_abl": w_same_merch_wrong_abl,
            # Char BM25
            "char_rank_orig": c_rank_orig,
            "char_rr_orig": 1.0 / c_rank_orig if c_rank_orig <= 813 else 0.0,
            "char_hit1_orig": 1 if c_rank_orig == 1 else 0,
            "char_hit5_orig": 1 if c_rank_orig <= 5 else 0,
            "char_hit10_orig": 1 if c_rank_orig <= 10 else 0,
            "char_same_merch_wrong_orig": c_same_merch_wrong_orig,
            "char_rank_abl": c_rank_abl,
            "char_rr_abl": 1.0 / c_rank_abl if c_rank_abl <= 813 else 0.0,
            "char_hit1_abl": 1 if c_rank_abl == 1 else 0,
            "char_hit5_abl": 1 if c_rank_abl <= 5 else 0,
            "char_hit10_abl": 1 if c_rank_abl <= 10 else 0,
            "char_same_merch_wrong_abl": c_same_merch_wrong_abl,
            # Dense BGE-Base
            "dense_rank_orig": b_rank_orig,
            "dense_rr_orig": 1.0 / b_rank_orig if b_rank_orig <= 813 else 0.0,
            "dense_hit1_orig": 1 if b_rank_orig == 1 else 0,
            "dense_hit5_orig": 1 if b_rank_orig <= 5 else 0,
            "dense_hit10_orig": 1 if b_rank_orig <= 10 else 0,
            "dense_same_merch_wrong_orig": b_same_merch_wrong_orig,
            "dense_rank_abl": b_rank_abl,
            "dense_rr_abl": 1.0 / b_rank_abl if b_rank_abl <= 813 else 0.0,
            "dense_hit1_abl": 1 if b_rank_abl == 1 else 0,
            "dense_hit5_abl": 1 if b_rank_abl <= 5 else 0,
            "dense_hit10_abl": 1 if b_rank_abl <= 10 else 0,
            "dense_same_merch_wrong_abl": b_same_merch_wrong_abl,
        })

    df_out = pd.DataFrame(detailed_rows)
    out_csv = repo_root / "TRANSACTION_DATE_ABLATION.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} query evaluations to {out_csv}")

    # Compute summary statistics across cohorts
    cohorts = [
        ("All Date-Bearing Queries (N=56)", df_out),
        ("Multi-Receipt Merchants (N=36)", df_out[df_out["num_merchant_docs"] > 1]),
        ("Single-Receipt Merchants (N=20)", df_out[df_out["num_merchant_docs"] == 1]),
    ]

    print("\n" + "="*80)
    print("EXPERIMENT 1: TRANSACTION-DATE ABLATION RESULTS SUMMARY")
    print("="*80)

    summary_rows = []
    for c_name, sub_df in cohorts:
        n = len(sub_df)
        print(f"\n--- Cohort: {c_name} (N={n}) ---")
        
        for sys_prefix, sys_name in [("word", "Word BM25"), ("char", "Char 3-5g BM25"), ("dense", "Dense BGE-Base")]:
            rr_orig = sub_df[f"{sys_prefix}_rr_orig"].values
            rr_abl = sub_df[f"{sys_prefix}_rr_abl"].values
            hit1_orig = sub_df[f"{sys_prefix}_hit1_orig"].values
            hit1_abl = sub_df[f"{sys_prefix}_hit1_abl"].values
            hit5_orig = sub_df[f"{sys_prefix}_hit5_orig"].values
            hit5_abl = sub_df[f"{sys_prefix}_hit5_abl"].values
            hit10_orig = sub_df[f"{sys_prefix}_hit10_orig"].values
            hit10_abl = sub_df[f"{sys_prefix}_hit10_abl"].values
            wrong_orig = sub_df[f"{sys_prefix}_same_merch_wrong_orig"].values
            wrong_abl = sub_df[f"{sys_prefix}_same_merch_wrong_abl"].values

            mrr_o, mrr_a = np.mean(rr_orig), np.mean(rr_abl)
            h1_o, h1_a = np.mean(hit1_orig), np.mean(hit1_abl)
            h5_o, h5_a = np.mean(hit5_orig), np.mean(hit5_abl)
            h10_o, h10_a = np.mean(hit10_orig), np.mean(hit10_abl)
            w_o, w_a = np.mean(wrong_orig), np.mean(wrong_abl)

            low, high, diff = bootstrap_ci_diff(rr_orig, rr_abl)
            p_perm = paired_permutation_test(rr_orig, rr_abl)
            chi2, p_mcnemar = mcnemar_test(hit1_orig, hit1_abl)

            print(f"[{sys_name}]")
            print(f"  MRR: Orig = {mrr_o:.4f} vs. Abl = {mrr_a:.4f} | Diff = {diff:+.4f} [95% CI: {low:+.4f}, {high:+.4f}] | p_perm = {p_perm:.4f}")
            print(f"  Hit@1: Orig = {h1_o:.4f} vs. Abl = {h1_a:.4f} | McNemar p = {p_mcnemar:.4f}")
            print(f"  Hit@5: Orig = {h5_o:.4f} vs. Abl = {h5_a:.4f} | Hit@10: Orig = {h10_o:.4f} vs. Abl = {h10_a:.4f}")
            print(f"  Same-Merchant False-Target Rate at Top-1: Orig = {w_o:.4f} vs. Abl = {w_a:.4f} (Change = {w_a - w_o:+.4f})")

            summary_rows.append({
                "cohort": c_name,
                "N": n,
                "system": sys_name,
                "mrr_orig": mrr_o,
                "mrr_ablated": mrr_a,
                "mrr_diff": diff,
                "mrr_ci_lower": low,
                "mrr_ci_upper": high,
                "p_perm": p_perm,
                "hit1_orig": h1_o,
                "hit1_ablated": h1_a,
                "p_mcnemar": p_mcnemar,
                "hit5_orig": h5_o,
                "hit5_ablated": h5_a,
                "hit10_orig": h10_o,
                "hit10_ablated": h10_a,
                "wrong_target_rate_orig": w_o,
                "wrong_target_rate_ablated": w_a,
                "wrong_target_rate_increase": w_a - w_o
            })

    summary_csv = repo_root / "TRANSACTION_DATE_ABLATION_SUMMARY.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved statistical summary to {summary_csv}")

if __name__ == "__main__":
    main()

