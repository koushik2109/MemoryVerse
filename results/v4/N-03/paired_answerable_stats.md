# N-03 Diagnostic Report: Paired Significance and Confidence Intervals for Char BM25 vs. ColQwen2 on Answerable Questions

## 1. Executive Summary
- **Issue:** In the manuscript, the p-value $p = 0.00012$ is attached to the Clean Ground-Truth difference ($+0.0876$) in Section 5.E, but to the Real-OCR difference ($+0.0733$) in Section 5.G.2 and Table XIII.
- **Root Cause:** In `results/v6/colqwen_vs_proxies_paired.csv`, the paired permutation test ($N_{\text{perm}} = 100,000$) was executed on the Clean Ground-Truth text proxy (Char BM25 MRR = 0.9450/0.9471 vs. ColQwen2 MRR = 0.8595), which yielded:
  - Difference: $+0.0855$ to $+0.0876$
  - 95% Cluster Bootstrap CI: $[+0.0413, +0.1284]$
  - Paired Permutation p-value: $p = 0.00012$.
  In subsequent drafting, this exact p-value ($0.00012$) was incorrectly pasted next to the Real-OCR difference ($+0.0733$, Char BM25 0.9328 vs. ColQwen2 0.8595).
- **Corrected Statistics:**
  1. **Primary Headline (Real-OCR, Apple Clean):**
     - Char BM25 MRR: $0.9328$ [0.906, 0.957]
     - ColQwen2-v1.0 MRR: $0.8595$ [0.825, 0.892]
     - Paired Difference: $+0.0733$
     - 95% Bootstrap CI: $[+0.0292, +0.1174]$
     - Paired Permutation p-value: $p = 0.0011$ ($p \approx 0.001$).
  2. **Secondary/Oracle Reference (Clean Ground-Truth Text):**
     - Char BM25 MRR: $0.9471$ [0.923, 0.969]
     - ColQwen2-v1.0 MRR: $0.8595$ [0.825, 0.892]
     - Paired Difference: $+0.0876$
     - 95% Bootstrap CI: $[+0.0413, +0.1284]$
     - Paired Permutation p-value: $p = 0.00012$.

## 2. Manuscript Paper Locations to Update (Phase 5)
1. **Key Contribution 1 (line 22):** Lead with Real-OCR diff $+0.0733$ ($p = 0.0011$), followed by Clean GT diff $+0.0876$ ($p = 0.00012$).
2. **Section 5.E (line 530):** Clarify both comparisons with their distinct p-values and 95% CIs.
3. **Table XIII (line 573):** Fix diff $+0.0733$ to cite $p = 0.0011$, keeping Clean GT diff $+0.0876$ at $p = 0.00012$.
4. **Section 5.G.2 (line 591):** State Real-OCR diff $+0.0733$ with $p = 0.0011$.

## 3. Impact Assessment
- Does this weaken or reverse a headline claim? **No.** Both Real-OCR ($p = 0.0011$) and Clean GT ($p = 0.00012$) remain highly statistically significant ($p < 0.01$). Character BM25 solidly outperforms ColQwen2 on answerable questions under both regimes.

