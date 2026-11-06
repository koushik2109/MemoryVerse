#!/usr/bin/env python3
"""
NEW_EXPERIMENT_SCRIPTS/run_experiment_2_same_merchant_hard_negatives.py
MemoryVerse — Experiment 2: Same-Merchant Hard-Negative Evaluation

Objective:
Determine whether retrieval systems distinguish the intended transaction from
other receipts issued by the exact same merchant (hard negatives), compared
to ordinary random distractors.

Systems evaluated:
1. Word BM25 (k1=1.5, b=0.75)
2. Character 3-5gram BM25 (k1=1.5, b=0.75)
3. Dense BGE-Base (BAAI/bge-base-en-v1.5)

Evaluated across:
- All queries targeting multi-receipt merchants (N=276 total: 138 verbatim, 138 paraphrase)
- Within-merchant hard-negative sub-rankings vs. full-corpus rankings
- Date-bearing paraphrase subset (N=36) vs. Non-date paraphrase subset (N=102)
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
# Tokenization & BM25
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
# Bootstrap & Permutation Helpers
# ---------------------------------------------------------

def bootstrap_ci(vals, num_bootstrap=5_000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(vals)
    indices = rng.randint(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(vals[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100)), float(np.mean(vals))

def bootstrap_ci_diff(a, b, num_bootstrap=5_000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    diff = np.array(a) - np.array(b)
    indices = rng.randint(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(diff[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100)), float(np.mean(diff))

def paired_permutation_test(a, b, num_permutations=100_000, seed=42):
    rng = np.random.RandomState(seed)
    diff = np.array(a) - np.array(b)
    obs = np.abs(np.mean(diff))
    if obs < 1e-9:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, len(diff)))
    perm_diffs = np.abs(np.mean(signs * diff, axis=1))
    return float((np.sum(perm_diffs >= obs) + 1.0) / (num_permutations + 1.0))

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    repo_root = Path(".")
    
    # 1. Load public benchmark
    bench_path = repo_root / "ai_engine/rag/data/v2/public_benchmark_v2.json"
    with open(bench_path) as f:
        bench = json.load(f)
        
    test_ids = set(bench.get("test_corpus_ids", []))
    test_corpus = [r for r in bench["records"] if r["id"] in test_ids]
    corpus_ids = [r["id"] for r in test_corpus]
    id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    corpus_texts = [r.get("real_ocr_text", r.get("ground_truth_text", "")) for r in test_corpus]
    print(f"Loaded {len(test_corpus)} test receipts.")

    # Build merchant groupings across corpus
    all_records = {r["id"]: r for r in bench["records"]}
    merchant_counts = Counter()
    doc_merchant = {}
    merchant_to_corpus_indices = defaultdict(list)
    
    for idx, r in enumerate(test_corpus):
        ent = r.get("key_entities", {})
        comp = ent.get("company", "").strip().upper()
        if comp:
            merchant_counts[comp] += 1
            doc_merchant[r["id"]] = comp
            merchant_to_corpus_indices[comp].append(idx)
        else:
            doc_merchant[r["id"]] = "UNKNOWN"

    # 2. Build Retrievers
    print("Building BM25 retrievers...")
    bm25_word = BM25Retriever(corpus_texts, tokenize_words)
    bm25_char = BM25Retriever(corpus_texts, tokenize_char_ngrams)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading BGE-Base model on {device}...")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device=device)
    corpus_embs = bge_model.encode(corpus_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    # 3. Load human queries and select multi-receipt merchant queries
    df_queries = pd.read_csv(repo_root / "data/human_queries_adjudicated.csv")
    
    date_regex = re.compile(
        r'(?:dated\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:on\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+\d{4})?)|(?:in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?)|(?:on\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:on\s+christmas(?:\s+\d{4})?)|(?:bought\s+on\s+christmas)|(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\s+\d{4})?)',
        re.IGNORECASE
    )

    eligible_queries = []
    for idx, r in df_queries.iterrows():
        tid = r["original_single_target"]
        rec = all_records.get(tid, {})
        ent = rec.get("key_entities", {})
        comp = ent.get("company", "").strip().upper()
        num_distractors = len([d_idx for d_idx in merchant_to_corpus_indices.get(comp, []) if corpus_ids[d_idx] != tid])
        
        if num_distractors > 0:
            has_date = bool(date_regex.findall(r["query_string"]))
            eligible_queries.append({
                "row_idx": r["row_idx"],
                "query_type": r["query_type"],
                "query_string": r["query_string"],
                "target_id": tid,
                "merchant": comp,
                "target_date": ent.get("date", "").strip(),
                "num_same_merchant_distractors": num_distractors,
                "has_date_expression": has_date
            })

    print(f"Identified {len(eligible_queries)} queries targeting merchants with hard distractors in test corpus.")
    
    # 4. Run retrieval and compute metrics
    eval_rows = []
    query_texts = [q["query_string"] for q in eligible_queries]
    query_embs = bge_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)

    for i, q in enumerate(eligible_queries):
        tid = q["target_id"]
        tgt_idx = id_to_idx.get(tid, -1)
        comp = q["merchant"]
        same_merchant_indices = merchant_to_corpus_indices.get(comp, [])
        
        # Word BM25
        w_scores = bm25_word.score(q["query_string"])
        w_ranks_full = np.argsort(-w_scores)
        w_rank_full = int(np.where(w_ranks_full == tgt_idx)[0][0]) + 1 if tgt_idx in w_ranks_full else 9999
        w_top1 = corpus_ids[w_ranks_full[0]]
        w_top1_is_target = (w_top1 == tid)
        w_top1_is_same_merch_false = (w_top1 != tid and doc_merchant.get(w_top1) == comp)
        w_top1_is_random_false = (w_top1 != tid and doc_merchant.get(w_top1) != comp)

        # Within-merchant hard negative ranking
        w_same_scores = [w_scores[idx] for idx in same_merchant_indices]
        w_same_sorted = np.argsort(-np.array(w_same_scores))
        w_within_rank = int(np.where(np.array(same_merchant_indices)[w_same_sorted] == tgt_idx)[0][0]) + 1

        # Char BM25
        c_scores = bm25_char.score(q["query_string"])
        c_ranks_full = np.argsort(-c_scores)
        c_rank_full = int(np.where(c_ranks_full == tgt_idx)[0][0]) + 1 if tgt_idx in c_ranks_full else 9999
        c_top1 = corpus_ids[c_ranks_full[0]]
        c_top1_is_target = (c_top1 == tid)
        c_top1_is_same_merch_false = (c_top1 != tid and doc_merchant.get(c_top1) == comp)
        c_top1_is_random_false = (c_top1 != tid and doc_merchant.get(c_top1) != comp)

        c_same_scores = [c_scores[idx] for idx in same_merchant_indices]
        c_same_sorted = np.argsort(-np.array(c_same_scores))
        c_within_rank = int(np.where(np.array(same_merchant_indices)[c_same_sorted] == tgt_idx)[0][0]) + 1

        # Dense BGE-Base
        b_scores = np.dot(corpus_embs, query_embs[i])
        b_ranks_full = np.argsort(-b_scores)
        b_rank_full = int(np.where(b_ranks_full == tgt_idx)[0][0]) + 1 if tgt_idx in b_ranks_full else 9999
        b_top1 = corpus_ids[b_ranks_full[0]]
        b_top1_is_target = (b_top1 == tid)
        b_top1_is_same_merch_false = (b_top1 != tid and doc_merchant.get(b_top1) == comp)
        b_top1_is_random_false = (b_top1 != tid and doc_merchant.get(b_top1) != comp)

        b_same_scores = [b_scores[idx] for idx in same_merchant_indices]
        b_same_sorted = np.argsort(-np.array(b_same_scores))
        b_within_rank = int(np.where(np.array(same_merchant_indices)[b_same_sorted] == tgt_idx)[0][0]) + 1

        eval_rows.append({
            "target_id": tid,
            "merchant": comp,
            "target_date": q["target_date"],
            "query_type": q["query_type"],
            "has_date_expression": q["has_date_expression"],
            "num_same_merchant_distractors": q["num_same_merchant_distractors"],
            "query_string": q["query_string"],
            # Word BM25 Full
            "word_rank_full": w_rank_full,
            "word_rr_full": 1.0 / w_rank_full if w_rank_full <= 813 else 0.0,
            "word_hit1_full": 1 if w_top1_is_target else 0,
            "word_hit5_full": 1 if w_rank_full <= 5 else 0,
            "word_hit10_full": 1 if w_rank_full <= 10 else 0,
            "word_same_merch_false": 1 if w_top1_is_same_merch_false else 0,
            "word_random_false": 1 if w_top1_is_random_false else 0,
            "word_within_rank": w_within_rank,
            "word_within_rr": 1.0 / w_within_rank,
            "word_within_hit1": 1 if w_within_rank == 1 else 0,
            # Char BM25 Full
            "char_rank_full": c_rank_full,
            "char_rr_full": 1.0 / c_rank_full if c_rank_full <= 813 else 0.0,
            "char_hit1_full": 1 if c_top1_is_target else 0,
            "char_hit5_full": 1 if c_rank_full <= 5 else 0,
            "char_hit10_full": 1 if c_rank_full <= 10 else 0,
            "char_same_merch_false": 1 if c_top1_is_same_merch_false else 0,
            "char_random_false": 1 if c_top1_is_random_false else 0,
            "char_within_rank": c_within_rank,
            "char_within_rr": 1.0 / c_within_rank,
            "char_within_hit1": 1 if c_within_rank == 1 else 0,
            # Dense BGE-Base Full
            "dense_rank_full": b_rank_full,
            "dense_rr_full": 1.0 / b_rank_full if b_rank_full <= 813 else 0.0,
            "dense_hit1_full": 1 if b_top1_is_target else 0,
            "dense_hit5_full": 1 if b_rank_full <= 5 else 0,
            "dense_hit10_full": 1 if b_rank_full <= 10 else 0,
            "dense_same_merch_false": 1 if b_top1_is_same_merch_false else 0,
            "dense_random_false": 1 if b_top1_is_random_false else 0,
            "dense_within_rank": b_within_rank,
            "dense_within_rr": 1.0 / b_within_rank,
            "dense_within_hit1": 1 if b_within_rank == 1 else 0,
        })

    df_out = pd.DataFrame(eval_rows)
    out_csv = repo_root / "SAME_MERCHANT_HARD_NEGATIVES.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} hard-negative evaluations to {out_csv}")

    # Summary analysis
    cohorts = [
        ("All Multi-Receipt Queries (N=276)", df_out),
        ("Verbatim Queries (N=138)", df_out[df_out["query_type"] == "verbatim"]),
        ("Paraphrase Queries (N=138)", df_out[df_out["query_type"] == "paraphrase"]),
        ("Paraphrase with Dates (N=36)", df_out[(df_out["query_type"] == "paraphrase") & (df_out["has_date_expression"])]),
        ("Paraphrase without Dates (N=102)", df_out[(df_out["query_type"] == "paraphrase") & (~df_out["has_date_expression"])]),
    ]

    print("\n" + "="*80)
    print("EXPERIMENT 2: SAME-MERCHANT HARD-NEGATIVE EVALUATION SUMMARY")
    print("="*80)

    summary_rows = []
    for c_name, sub_df in cohorts:
        n = len(sub_df)
        print(f"\n--- Cohort: {c_name} (N={n}) ---")
        for sys_prefix, sys_name in [("word", "Word BM25"), ("char", "Char 3-5g BM25"), ("dense", "Dense BGE-Base")]:
            rr_full = sub_df[f"{sys_prefix}_rr_full"].values
            h1_full = sub_df[f"{sys_prefix}_hit1_full"].values
            h5_full = sub_df[f"{sys_prefix}_hit5_full"].values
            h10_full = sub_df[f"{sys_prefix}_hit10_full"].values
            same_false = sub_df[f"{sys_prefix}_same_merch_false"].values
            rand_false = sub_df[f"{sys_prefix}_random_false"].values
            within_rr = sub_df[f"{sys_prefix}_within_rr"].values
            within_h1 = sub_df[f"{sys_prefix}_within_hit1"].values

            low_rr, high_rr, mrr_full = bootstrap_ci(rr_full)
            low_wrr, high_wrr, mrr_within = bootstrap_ci(within_rr)

            print(f"[{sys_name}]")
            print(f"  Full Corpus MRR: {mrr_full:.4f} [95% CI: {low_rr:.4f}, {high_rr:.4f}] | Hit@1: {np.mean(h1_full):.4f} | Hit@5: {np.mean(h5_full):.4f} | Hit@10: {np.mean(h10_full):.4f}")
            print(f"  Top-1 Outcome Breakdown: Target = {np.mean(h1_full):.4f} | Same-Merchant False = {np.mean(same_false):.4f} | Random False = {np.mean(rand_false):.4f}")
            print(f"  Within-Merchant Discrimination MRR: {mrr_within:.4f} [95% CI: {low_wrr:.4f}, {high_wrr:.4f}] | Within-Merchant Top-1: {np.mean(within_h1):.4f}")

            summary_rows.append({
                "cohort": c_name,
                "N": n,
                "system": sys_name,
                "full_mrr": mrr_full,
                "full_mrr_ci_lower": low_rr,
                "full_mrr_ci_upper": high_rr,
                "full_hit1": np.mean(h1_full),
                "full_hit5": np.mean(h5_full),
                "full_hit10": np.mean(h10_full),
                "same_merchant_false_rate": np.mean(same_false),
                "random_distractor_false_rate": np.mean(rand_false),
                "within_merchant_mrr": mrr_within,
                "within_merchant_mrr_ci_lower": low_wrr,
                "within_merchant_mrr_ci_upper": high_wrr,
                "within_merchant_hit1": np.mean(within_h1)
            })

    summary_csv = repo_root / "SAME_MERCHANT_HARD_NEGATIVES_SUMMARY.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved hard-negative summary to {summary_csv}")

if __name__ == "__main__":
    main()
