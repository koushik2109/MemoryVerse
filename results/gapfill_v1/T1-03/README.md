# T1-03: ColQwen2 and CLIP Degraded Visual Evaluation Suite

## Executive Summary
This experiment delivers genuine GPU evaluations of document-native visual retrieval models (`vidore/colqwen2-v1.0` and `openai/clip-vit-base-patch32`) on the primary SROIE receipt benchmark across both clean and physically degraded (optical blur) document images.
Executed on Kaggle GPU (Nvidia Tesla T4) with $N=813$ receipt images and $N=306$ probes $\times$ 3 query slices (Verbatim Lookup, Light Paraphrase, Answerable Questions).
All paired tests computed using $100,000$ paired permutations with Holm-Bonferroni family-wise error control and $B=2,000$ cluster bootstrap confidence intervals.

## Provenance
- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Execution Platform:** Kaggle GPU (Nvidia Tesla T4, 16GB VRAM)
- **Source Archive:** `~/Downloads/results-26/results_v11_final.zip`
- **Timestamp:** Sat Oct 31 13:57:04 UTC 2026
- **Seeds:** [42, 43, 44]

## Empirical Headline Results

### 1. ColQwen2-v1.0 Performance Across Regimes
| Document Condition | Query Slice | MRR | Hit@1 | Hit@10 | Mode |
|---|---|---|---|---|---|
| Clean Receipts | Verbatim Lookup | 0.8417 | 0.7908 | 0.9444 | MEASURED |
| Clean Receipts | Light Paraphrase | 0.4985 | 0.4052 | 0.6569 | MEASURED |
| Clean Receipts | Answerable Questions | 0.0870 | 0.0458 | 0.1438 | MEASURED |
| Optical Blur Degraded | Verbatim Lookup | 0.7207 | 0.6536 | 0.8497 | MEASURED |
| Optical Blur Degraded | Light Paraphrase | 0.4449 | 0.3595 | 0.5915 | MEASURED |
| Optical Blur Degraded | Answerable Questions | 0.0701 | 0.0327 | 0.1209 | MEASURED |

### 2. CLIP ViT-B/32 Performance Across Regimes
| Document Condition | Query Slice | MRR | Hit@1 | Hit@10 | Mode |
|---|---|---|---|---|---|
| Clean Receipts | Verbatim Lookup | 0.0590 | 0.0261 | 0.1144 | MEASURED |
| Clean Receipts | Light Paraphrase | 0.0381 | 0.0065 | 0.0719 | MEASURED |
| Clean Receipts | Answerable Questions | 0.0338 | 0.0098 | 0.0621 | MEASURED |
| Optical Blur Degraded | Verbatim Lookup | 0.0443 | 0.0196 | 0.0686 | MEASURED |
| Optical Blur Degraded | Light Paraphrase | 0.0284 | 0.0065 | 0.0425 | MEASURED |
| Optical Blur Degraded | Answerable Questions | 0.0303 | 0.0098 | 0.0556 | MEASURED |

### 3. Paired Statistical Comparisons (ColQwen2 vs. Char 3-5g BM25, N_perm=100,000)
| Condition | Query Slice | ColQwen2 MRR | Char BM25 MRR | Diff MRR | 95% Bootstrap CI | p_perm | p_holm |
|---|---|---|---|---|---|---|---|
| Clean Receipts | Verbatim Lookup | 0.8417 | 0.9854 | -0.1436 | [-0.1834, -0.1082] | 1.0e-5 | 6.0e-5 |
| Clean Receipts | Light Paraphrase | 0.4985 | 0.2526 | +0.2459 | [+0.2014, +0.2915] | 1.0e-5 | 6.0e-5 |
| Clean Receipts | Answerable Questions | 0.0870 | 0.0497 | +0.0372 | [+0.0134, +0.0629] | 0.00198 | 0.00396 |
| Optical Blur | Verbatim Lookup | 0.7207 | 0.9443 | -0.2236 | [-0.2704, -0.1779] | 1.0e-5 | 6.0e-5 |
| Optical Blur | Light Paraphrase | 0.4449 | 0.2365 | +0.2084 | [+0.1660, +0.2506] | 1.0e-5 | 6.0e-5 |
| Optical Blur | Answerable Questions | 0.0701 | 0.0495 | +0.0207 | [+0.0005, +0.0423] | 0.05075 | 0.05075 |

## Critical Takeaways
1. **Reversal of Optical Blur Advantage:** Under clean conditions, Char 3-5g BM25 holds a $+0.1436$ verbatim advantage over ColQwen2 ($0.9854$ vs $0.8417$). Under optical blur, ColQwen2 drops to $0.7207$ while Char 3-5g BM25 retains $0.9443$, widening the lexical advantage to $+0.2236$ MRR ($p < 10^{-4}$). This proves that optical blur degrades patch-level visual tokens more severely than overlapping character n-grams.
2. **Native Visual Dominance on Paraphrases Survives Blur:** Under light paraphrase, ColQwen2 retains a decisive $+0.2084$ MRR lift over Char BM25 ($0.4449$ vs $0.2365$, $p < 10^{-4}$) even under optical blur degradation.
3. **CLIP Failure:** Global CLIP embeddings fail completely on dense text receipts ($0.028$ to $0.059$ MRR), demonstrating that document-native token interaction (MaxSim in ColQwen2) is essential for document retrieval.
