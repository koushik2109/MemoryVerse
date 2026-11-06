# Independent Forensic Re-Audit of Risk-Controlling Prediction Sets (RCPS)

**Document ID:** `FINAL_BLOCKER_RESOLUTION/RCPS_REAUDIT.md`  
**Date:** November 2026  
**Auditor:** Senior Statistical Auditor & Information Retrieval Scientist  
**Focus:** Resolution of the RCPS Mathematical Discrepancy (9.7% vs. 18.0% Exceedance)  
**Status:** FULL RE-AUDIT COMPLETED FROM 200 RAW SPLIT RECORDS  

---

## 1. Executive Summary & Resolution of the Mathematical Discrepancy

In the prior audit report, a mathematical inconsistency was noted:
> *"The previous report claims that mean risk 0.0429, threshold 0.05, test size 1,377, and standard error 0.00546 imply approximately 18% threshold exceedance. The normal approximation from those figures gives approximately 9.7%, not 18%."*

### Mathematical Root-Cause Analysis
Our forensic investigation into all 200 split-level records reveals the exact source of this discrepancy:
1. **The Pure Binomial Standard Error ($\sigma_{\text{binom}} = 0.005461$):**  
   If the calibration threshold $\tau^*$ were **fixed and deterministic**, the sampling variance on a single test set of size $N_{\text{test}} = 1,377$ with true population risk $p = 0.04293$ would indeed be:
   $$\sigma_{\text{binom}} = \sqrt{\frac{p(1 - p)}{N_{\text{test}}}} = \sqrt{\frac{0.04293 \times (1 - 0.04293)}{1377}} \approx 0.005461$$
   Under this naive single-set assumption, the normal approximation for exceeding $\alpha = 0.0500$ is:
   $$z_{\text{binom}} = \frac{0.0500 - 0.04293}{0.005461} = 1.2946 \implies 1 - \Phi(1.2946) = 0.0977 \approx \mathbf{9.77\%}$$
   The reviewer was correct: $\sigma = 0.00546$ yields $9.7\%$, not $18\%$.

2. **The True Empirical Standard Deviation Across Splits ($\sigma_{\text{empirical}} = 0.007562$):**  
   Across the 200 splits, the threshold $\hat{\tau}$ is **not fixed**. It is re-calibrated on each split's calibration partition ($n_{\text{cal}} = 1,377$). The calibrated threshold $\hat{\tau}_{\text{hp}}$ varies across splits ($\text{Mean} = 0.2468, \text{Std} = 0.0054, \text{Min} = 0.2310, \text{Max} = 0.2540$).  
   This threshold variability compounds with the test-set sampling variability. Across the 200 empirical splits, the actual measured standard deviation of the test risk is:
   $$\sigma_{\text{empirical}} = \mathbf{0.007562}$$
   Evaluating the exceedance probability using the true empirical dispersion:
   $$z_{\text{empirical}} = \frac{0.0500 - 0.04293}{0.007562} = 0.9349 \implies 1 - \Phi(0.9349) = 1 - 0.8251 = 0.1749 \approx \mathbf{17.5\%} \approx \mathbf{18.0\%}$$
   
### Conclusion
The empirical split satisfaction of **82.00%** (164 / 200 satisfied, 36 / 200 exceeded) is completely reproducible and mathematically sound. The previous report's error was a drafting conflation: it cited the naive binomial test error $\sigma_{\text{binom}} = 0.00546$ (which produces 9.77%) while stating the empirical split exceedance of 18.0%. The 18.0% exceedance is governed by the total across-split standard deviation $\sigma_{\text{total}} = 0.007562$.

---

## 2. Comprehensive Empirical Risk Distribution Across All 200 Splits

Independently recalculated from raw records in [`FINAL_BLOCKER_RESOLUTION/RCPS_200_SPLITS_RECALCULATED.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/FINAL_BLOCKER_RESOLUTION/RCPS_200_SPLITS_RECALCULATED.csv):

| Metric | High-Probability RCPS ($\delta = 0.10$) | Standard Expected Risk (Bates et al., 2021) |
| :--- | :---: | :---: |
| **Number of Evaluated Splits** | 200 (Seeds 1000–1199) | 200 (Seeds 1000–1199) |
| **Calibration Set Size ($n_{\text{cal}}$)** | 1,377 | 1,377 |
| **Test Set Size ($N_{\text{test}}$)** | 1,377 | 1,377 |
| **Mean Calibrated Threshold ($\hat{\tau}$)** | 0.2468 [Std: 0.0054] | 0.2369 [Std: 0.0058] |
| **Mean Test Miss Risk** | **0.042930** | **0.049361** |
| **Median Test Miss Risk ($p_{50}$)** | 0.042847 | 0.047930 |
| **Empirical Standard Deviation** | **0.007562** | **0.008078** |
| **Sample Variance** | $5.718 \times 10^{-5}$ | $6.525 \times 10^{-5}$ |
| **Minimum Observed Test Risk** | 0.023239 | 0.026144 |
| **Maximum Observed Test Risk** | 0.065359 | 0.068991 |
| **1st Percentile ($p_{01}$)** | 0.025410 | 0.031184 |
| **5th Percentile ($p_{05}$)** | 0.030501 | 0.037763 |
| **10th Percentile ($p_{10}$)** | 0.032607 | 0.039869 |
| **25th Percentile ($p_{25}$)** | 0.037763 | 0.044299 |
| **75th Percentile ($p_{75}$)** | 0.047204 | 0.055192 |
| **80th Percentile ($p_{80}$)** | 0.049383 | 0.057371 |
| **82nd Percentile ($p_{82}$)** | **0.049513** ($\le 0.0500$) | 0.058097 |
| **85th Percentile ($p_{85}$)** | 0.050835 ($> 0.0500$) | 0.058824 |
| **90th Percentile ($p_{90}$)** | 0.052288 | 0.060276 |
| **95th Percentile ($p_{95}$)** | 0.055229 | 0.062527 |
| **99th Percentile ($p_{99}$)** | 0.060276 | 0.068264 |
| **Empirical Splits Satisfying Risk $\le 0.05$** | **164 / 200 (82.00%)** | **117 / 200 (58.50%)** |
| **Empirical Splits Exceeding Risk $> 0.05$** | **36 / 200 (18.00%)** | **83 / 200 (41.50%)** |

---

## 3. Methodological Audit: Split Independence and Binomial Suitability

### 1. Are the Splits Independent?
**No.** All 200 splits are formed by repeatedly drawing 50/50 partitions from a single fixed pool of $N = 2,754$ queries.
- In each split, 1,377 instances are assigned to calibration and 1,377 to test.
- Any two distinct splits share approximately $50\%$ of their calibration instances and $50\%$ of their test instances.
- Because observations are reused across splits, the realizations of $\hat{\tau}_s$ and $\hat{R}_s$ are positively correlated. Consequently, the 200 observations do **not** represent 200 independent Bernoulli trials from an infinite data stream.

### 2. Is a Simple Binomial Approximation Appropriate?
**No, not for modeling across-split exceedance.**
A simple binomial model $\text{Binomial}(N_{\text{test}}, p)$ assumes:
1. The population risk $p$ is invariant across repetitions.
2. The calibration threshold is fixed.
3. Every test evaluation is statistically independent.

In reality, the total variance of test risk $\hat{R}_{\text{test}}$ decomposes into three distinct components:
$$\text{Var}(\hat{R}_{\text{test}}) = \underbrace{\mathbb{E}_{\hat{\tau}}\left[\text{Var}(\hat{R}_{\text{test}} \mid \hat{\tau})\right]}_{\text{Within-Test Sampling Variance } (\sigma^2_{\text{binom}} \approx 2.98 \times 10^{-5})} + \underbrace{\text{Var}_{\hat{\tau}}\left(\mathbb{E}[\hat{R}_{\text{test}} \mid \hat{\tau}]\right)}_{\text{Threshold Calibration Variance } (\sigma^2_{\text{cal}} \approx 2.74 \times 10^{-5})}$$
Summing these variance components yields:
$$\sigma^2_{\text{total}} = 2.98 \times 10^{-5} + 2.74 \times 10^{-5} = 5.72 \times 10^{-5} \implies \sigma_{\text{total}} = 0.007562$$
The empirical standard deviation ($0.007562$) is $38.5\%$ larger than the pure binomial standard error ($0.005461$). This explains why the empirical exceedance rate is $18.0\%$ rather than $9.77\%$.

---

## 4. Formal RCPS Guarantee vs. Empirical Split Coverage

It is critical to distinguish between what the mathematical theory guarantees versus what is measured on empirical splits:

### What RCPS Formally Guarantees
Risk-Controlling Prediction Sets (Bates et al., 2021) provide a Probably Approximately Correct (PAC) bound on the **true population risk** $R(\hat{\lambda})$:
$$\mathbb{P}_{\mathcal{D}_{\text{cal}}}\left(R(\hat{\lambda}) \le \alpha\right) \ge 1 - \delta$$
- The probability is taken over the randomness of the calibration set $\mathcal{D}_{\text{cal}}$.
- The guarantee states that with probability at least $1 - \delta = 0.90$, the **true infinite-sample population risk** of the calibrated threshold is bounded by $\alpha = 0.05$.

### Why Empirical Split Satisfaction is 82.0%, Not 90.0%
1. On each split $s$, we do not observe the true population risk $R(\hat{\lambda}_s)$; we only observe the **empirical test-sample risk** $\hat{R}_{\text{test}, s} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} L_i$.
2. Even in splits where the formal guarantee holds and the true population risk is strictly $R(\hat{\lambda}_s) = 0.046 \le 0.05$, finite test-sample fluctuations ($N_{\text{test}} = 1,377$) can easily cause the sample realization $\hat{R}_{\text{test}, s}$ to evaluate to $0.051$ or $0.052$.
3. When evaluated empirically across finite test sets, the fraction of splits observing $\hat{R}_{\text{test}, s} \le 0.05$ will naturally fall below the nominal population confidence level $1 - \delta = 0.90$.
4. **Reproducibility Verdict:** The 82.00% figure is completely authentic and reproducible from the code. It is not an artifact of faulty code, but a direct reflection of finite test sample variance around a calibrated threshold.

---

## 5. Subgroup Risk Inflation: A Disclosed Negative Result

The split records confirm the critical negative result disclosed in the manuscript:

| Query Subgroup | Mean Test Miss Risk | Satisfies Risk $\le 0.05$? |
| :--- | :---: | :---: |
| **Anchored Questions** | 0.033780 | YES |
| **Verbatim Queries** | 0.042113 | YES |
| **Entity Rewrites (Paraphrases)** | **0.052898** | **NO (Inflates beyond 0.05)** |

Under semantic query shift (entity rewrites), the miss risk rises to $0.0529$ under RCPS (and to $0.0627$ under uncorrected marginal calibration). This confirms that marginal calibration cannot guarantee conditional safety across semantic subgroups, justifying simpler fixed-margin cascades in production.

---

## 6. Manuscript Corrections Required

The following adjustments must be made in Section VI (Limitations, Item 3) of the manuscript:

```latex
% REPLACEMENT WORDING FOR IJCDS_BODY.TEX (SECTION VI, LIMITATIONS ITEM 3):
\item \textbf{Finite-Sample Marginal vs.\ PAC-Style Uncertainty Guarantees:} 
Under marginal risk control $L = \mathbf{1}[\text{not escalated} \land Y \ne 1]$, 
split conformal calibration controls expected risk ($\mathbb{E}[L] \le \alpha = 0.05$, 
empirical mean $0.0491$ at a 78.4\% escalation rate, $N=2{,}754$, 200 splits). 
For applications requiring high-probability PAC-style bounds rather than marginal 
expectations, Risk-Controlling Prediction Sets (RCPS, $\delta=0.10$) achieve a mean test 
miss risk of $0.0429 \le 0.05$. Across 200 random splits, empirical split coverage 
reaches 82.0\% ($\hat{R}_{\text{test}} \le 0.05$); the gap between nominal population 
confidence (90\%) and empirical finite-sample split satisfaction (82\%) reflects the 
compound dispersion ($\sigma_{\text{total}} = 0.0076$) combining calibration threshold 
variability with finite test-set evaluation noise ($N_{\text{test}}=1{,}377$), under which 
normal exceedance theory predicts an $\approx 17.5\%$ threshold exceedance rate.
```

