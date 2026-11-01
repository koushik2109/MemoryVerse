# T1-05: Multiple Split Seeds Robustness Evaluation

## Overview
This experiment systematically tests the stability of all headline empirical claims across **20 independent Dev/Test split seeds** (seeds 42 through 61), extending the primary frozen split (seed 42).

## Key Headline Findings & Ordering Hold Rates
Across all 20 seeds evaluated:
1. **Verbatim Inverted Index Dominance (Char BM25 > Word BM25):**
   - **Hold Rate: 20/20 (100.0%)**
   - Char BM25 Mean MRR: **0.9443 ± 0.0062** [Range: 0.9305 – 0.9570]
   - Word BM25 Mean MRR: **0.8272 ± 0.0118** [Range: 0.8049 – 0.8463]
   - Mean Lift: **+0.1171 ± 0.0120** [Range: +0.0841 to +0.1365]

2. **Paraphrase Reversal (Word BM25 > Char BM25 under Light Paraphrase):**
   - **Hold Rate: 20/20 (100.0%)**
   - Word BM25 Mean MRR: **0.4115 ± 0.0313** [Range: 0.3644 – 0.4679]
   - Char BM25 Mean MRR: **0.2658 ± 0.0308** [Range: 0.2203 – 0.3470]
   - Mean Reversal Lift: **+0.1457 ± 0.0211** [Range: +0.0930 to +0.1831]

3. **Hybrid Robustness (Tuned Hybrid >= Char BM25 under Light Paraphrase):**
   - **Hold Rate: 20/20 (100.0%)**
   - Tuned Hybrid Mean MRR: **0.4115 ± 0.0313** [Range: 0.3644 – 0.4679]
   - Mean Lift over Char: **+0.1457 ± 0.0211**

4. **ColQwen2 Document-Native Visual Superiority under Paraphrase:**
   - **Hold Rate: 20/20 (100.0%)** vs Char BM25
   - **Hold Rate: 20/20 (100.0%)** vs Word BM25
   - **Hold Rate: 20/20 (100.0%)** vs Tuned Hybrid
   - ColQwen2 Mean MRR: **0.4971 ± 0.0190**
   - Lift over Char BM25: **+0.2548 ± 0.0137** [Range: +0.2248 to +0.2761]
   - Lift over Word BM25: **+0.0846 ± 0.0125** [Range: +0.0559 to +0.1096]
   - Lift over Hybrid: **+0.2447 ± 0.0138** [Range: +0.2142 to +0.2680]

## Paper Impact
- **Tables Affected:** Extends Table II and Table X discussion with formal multi-seed variance bounds.
- **Claims Strengthened:** Confirms with 100% split consistency that:
  * Verbatim character $n$-gram dominance is not an artifact of split seed 42.
  * Paraphrase reversal (Word BM25 > Char BM25) is an invariant property across all 20 splits.
  * ColQwen2's lift over text proxies under semantic paraphrase holds across 100% of evaluated splits.

## Provenance
- **Git Commit:** `6b9f2aaa8a873049f7d95fd831e22f868cd5e636`
- **Seeds Evaluated:** `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]`
- **Total Runtime:** `239.16s`
