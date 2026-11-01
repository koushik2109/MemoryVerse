# Tier 3 Item T3-04: Query-Side Noise Robustness Evaluation

## Overview
Evaluates retrieval robustness when user queries are corrupted by realistic keyboard-adjacent QWERTY typos,
optical letter confusions, transpositions, and character omissions across error rates $r \in [0\%, 30\%]$.

---

## Headline Findings

| System | 0% Typo (Clean) | 5% Typo | 10% Typo | 15% Typo | 20% Typo | 25% Typo | 30% Typo | Net Retention at 20% Typo |
|---|---|---|---|---|---|---|---|---|
| **Char 3–5g BM25** | **0.9436** | **0.9125** | **0.8654** | **0.8042** | **0.7328** | **0.6514** | **0.5781** | **77.7%** |
| **Word + Fuzzy BM25**| 0.8005 | 0.7712 | 0.7240 | 0.6651 | 0.6012 | 0.5348 | 0.4690 | 75.1% |
| **Word BM25** | 0.8371 | 0.7645 | 0.6721 | 0.5714 | 0.4820 | 0.3951 | 0.3120 | 57.6% |
| **Dense MiniLM** | 0.2040 | 0.1982 | 0.1874 | 0.1740 | 0.1582 | 0.1412 | 0.1250 | 77.5% |

### Core Takeaways:
1. **Asymmetric Invariance:** Contiguous character $n$-grams ($n=3..5$) exhibit superior query-side noise tolerance. At a 20% query typo rate, Char BM25 retains **0.7328 MRR** (77.7% of clean performance), whereas whole-word BM25 collapses to **0.4820 MRR** (42.4% drop).
2. **Fuzzy Expansion Utility:** Levenshtein expansion (dist $\le 1$) rescues whole-word BM25 from catastrophic collapse under query typos (lifting MRR from 0.4820 to 0.6012 at 20% noise), but still lags behind character $n$-grams (0.7328) by over $+0.13$ MRR.
3. **Dense Fragility:** Dense neural embeddings show steady relative retention (77.5%), but start from a crippled absolute baseline (0.2040 -> 0.1582), confirming that subword encoders cannot overcome query-document format mismatches.

---

## Reproducibility
- Command: `.venv/bin/python scripts/gapfill_t3_04_query_noise.py`
- Seed: 42
- Git Commit: 6b9f2aaa8a873049f7d95fd831e22f868cd5e636
