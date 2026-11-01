# T3-10: Granular Error Analysis & Failure Mode Attribution

## Overview
This study provides an exhaustive diagnostic failure attribution across all $N=306$ held-out probes and 813 corpus items, categorizing every retrieval failure (Rank > 10 / Hit@10 = 0) into distinct physical and semantic root causes.

- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Models Benchmarked:** Word BM25, Char 3–5g BM25, all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5
- **Regimes Evaluated:** Apple Vision Real-OCR vs. Ground-Truth Clean Text
- **Query Slices:** Verbatim Probes vs. Light Paraphrase Probes

## Diagnostic Failure Taxonomy
1. **OCR Target Absence:** The target document's OCR output completely omits key query tokens (<15% word recall).
2. **Sparse Distractor Collision:** The target document contains the tokens, but distractors contain identical or higher-frequency matches.
3. **Dense Numeric/Short Dispersion:** Dense dual-encoders collapse on numeric dates/amounts or short entity strings ($\le 4$ tokens) despite token presence in target.
4. **Paraphrase Entity Gap:** Synthetic paraphrase replaces specific entity anchors with non-overlapping descriptive phrasing (0 verbatim words in target).

## Key Headline Insights

### 1. Verbatim Regime: Character BM25 Eliminates Distractor Collisions
On Real OCR verbatim queries:
- **Char 3–5g BM25:** Achieves **97.06% Hit@10** (only 9 failures out of 306). Failures are driven almost entirely by extreme OCR target absence (7/9).
- **Word BM25:** Suffers **51 failures** (83.33% Hit@10). 38 of 51 failures (74.5%) are caused by OCR token fragmentation causing false distractor collisions.
- **Dense Models (BGE & MiniLM):** Suffer **189 failures** (BGE Hit@10 = 38.24%) and **219 failures** (MiniLM Hit@10 = 28.43%). Over 72% of dense failures occur on numeric-heavy and short entity queries where embedding pooling loses specificity.

### 2. Paraphrase Regime: Why Word BM25 Reverses
Under light paraphrasing:
- **Char 3–5g BM25 failures surge to 196** (35.95% Hit@10), with 82% caused by *Paraphrase Entity Gap* (character n-grams matching irrelevant substrings when lexical anchors are stripped).
- **Word BM25 retains 58.17% Hit@10** (128 failures), as word-level IDF assigns high weight to shared topical stems while ignoring noisy character subwords.
