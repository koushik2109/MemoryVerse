# DEEP THEORETICAL & STATISTICAL AUDIT: RCPS CONFORMAL RISK CONTROL
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Auditor:** Independent Senior Statistical Reviewer & Reproducibility Auditor  
**Date:** November 2026  
**Audit Directory:** `IJCDS_SECOND_AUDIT/`  

---

## 1. Executive Summary & Verdict

### Final Scientific Classification:
### **VERDICT: B. VALID IMPLEMENTATION BUT MISLEADING INTERPRETATION**

```
+----------------------------------------------------------------------------------------------------+
|                                     RCPS AUDIT CLASSIFICATION                                      |
|                                                                                                    |
|                        [ B. VALID IMPLEMENTATION BUT MISLEADING INTERPRETATION ]                    |
|                                                                                                    |
| 1. Code Implementation: The Python implementation of marginal conformal prediction (Bates et al.,   |
|    2021) and high-probability Risk-Controlling Prediction Sets (RCPS via Clopper-Pearson inversion; |
|    Angelopoulos et al., 2021) in `scripts/gapfill_g02_conditional_conformal.py` is mathematically   |
|    correct, rigorously cluster-stratified, and completely free of calibration/test leakage.       |
| 2. Paper Interpretation: The manuscript reports an 82.0% empirical split coverage rate under a     |
|    nominal 90% guarantee (delta = 0.10) without explaining the distinction between true population |
|    risk coverage and empirical test-set binomial sampling noise. This creates a misleading         |
|    impression that the calibration algorithm slightly underperformed its theoretical bound.       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Algorithm Verification & Theoretical Procedure

### 2.1 Cited Procedures
The manuscript relies on two theoretical frameworks from distribution-free uncertainty quantification:
1. **Marginal Conformal Risk Control (CRC):** Bates, Angelopoulos, Lei, Malik, and Jordan (2021), *Distribution-Free, Risk-Controlling Prediction Sets*.
2. **High-Probability Learn-Then-Test / RCPS:** Angelopoulos, Bates, Candès, Jordan, and Lei (2021), *Learn-Then-Test: High-Probability Risk Control for Machine Learning*.

### 2.2 Mathematical Implementation in Source Code
In `scripts/gapfill_g02_conditional_conformal.py`:

#### A. Marginal Conformal Risk Control (`calibrate_marginal_bates`, lines 66–77):
```python
def calibrate_marginal_bates(m_cal: np.ndarray, h1_cal: np.ndarray, eps: float = TARGET_EPSILON) -> float:
    n = len(m_cal)
    for tau in TAU_GRID:
        loss = (m_cal >= tau) & (h1_cal == 0)
        r_hat = np.mean(loss)
        r_adj = (n * r_hat + 1.0) / (n + 1.0)
        if r_adj <= eps:
            return float(tau)
    return 0.50
```
- **Mathematical Form:**
  $$R_{\text{adj}}(\tau) = \frac{n \hat{R}(\tau) + 1}{n + 1} \le \alpha$$
- **Theoretical Assessment:** This is the exact finite-sample correction proven by Bates et al. (2021). It rigorously guarantees marginal expectation control:
  $$\mathbb{E}_{D_{\text{cal}}, (X, Y)}\left[\ell(\hat{\tau}; X, Y)\right] \le \alpha$$
  under exchangeability.

#### B. High-Probability RCPS via Clopper-Pearson Inversion (`calibrate_rcps_clopper`, lines 80–92):
```python
def calibrate_rcps_clopper(m_cal: np.ndarray, h1_cal: np.ndarray, eps: float = TARGET_EPSILON, delta: float = 0.10) -> float:
    n = len(m_cal)
    for tau in TAU_GRID:
        loss = (m_cal >= tau) & (h1_cal == 0)
        k = int(np.sum(loss))
        ub = 1.0 if k == n else float(beta.ppf(1.0 - delta, k + 1, n - k))
        if ub <= eps:
            return float(tau)
    return 0.50
```
- **Mathematical Form:**
  The upper confidence bound $UB_{1-\delta}(k, n)$ is the exact Clopper-Pearson (1934) upper bound for a binomial parameter, evaluated via the quantile function of the Beta distribution:
  $$UB_{1-\delta} = \text{Beta}^{-1}(1 - \delta; k + 1, n - k)$$
  The threshold $\hat{\tau}$ is chosen as:
  $$\hat{\tau} = \inf \left\{ \tau \in \mathcal{T} : UB_{1-\delta}(k(\tau), n) \le \alpha \right\}$$
- **Theoretical Assessment:** For bounded binary or monotone losses, the Clopper-Pearson upper bound is an exact, non-asymptotic concentration bound. It guarantees that over random draws of the calibration set $D_{\text{cal}}$:
  $$\mathbb{P}_{D_{\text{cal}}}\left( R(\hat{\tau}) \le \alpha \right) \ge 1 - \delta$$
  where $R(\hat{\tau}) = \mathbb{E}[\ell(\hat{\tau}; X, Y)]$ is the true population risk.

---

## 3. Data Partitioning, Leakage, and Independence Audit

### 3.1 Cluster-Stratified Splitting (`gapfill_g02_conditional_conformal.py`, lines 224–234)
- **Data Structure:** The pooled cache consists of $N = 2{,}754$ query instances across 306 document clusters (3 optical noise conditions $\times$ 3 query formulations per document).
- **Partitioning Protocol:**
  ```python
  shuffled_clusters = rng.permutation(unique_clusters)
  split_point = len(shuffled_clusters) // 2
  cal_c = set(shuffled_clusters[:split_point])
  test_c = set(shuffled_clusters[split_point:])
  ```
- **Audit Findings:**
  1. **Zero Data Leakage:** The 50/50 split is performed at the **document cluster level** ($153$ clusters in calibration, $153$ clusters in test). All 9 representations of any given document reside exclusively on one side of the split.
  2. **Zero Threshold Reuse:** A fresh threshold $\hat{\tau}_s$ is computed for each split $s \in \{1, \dots, 200\}$ using only $D_{\text{cal}, s}$.
  3. **Strict Separation:** Evaluation on $D_{\text{test}, s}$ uses the frozen $\hat{\tau}_s$.

---

## 4. Split-Level Numerical Recalculation across 200 Splits

Extracting all 200 splits from `results/v6/conformal_high_prob_v6.csv` and `results/v7/conformal_retest_v7.csv` (archived in [`RCPS_SPLIT_LEVEL_AUDIT.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/IJCDS_SECOND_AUDIT/RCPS_SPLIT_LEVEL_AUDIT.csv)):

| Evaluation Metric | Marginal CRC ($\alpha = 0.05$) | High-Probability RCPS ($\alpha = 0.05, \delta = 0.10$) |
|---|---|---|
| Calibration Sample Size ($N_{\text{cal}}$) | 1,377 | 1,377 |
| Test Sample Size ($N_{\text{test}}$) | 1,377 | 1,377 |
| **Mean Calibrated Threshold ($\hat{\tau}^*$)** | **0.2347** | **0.2468** |
| Threshold Standard Deviation | 0.0094 | 0.0102 |
| **Mean Test Miss Risk** | **0.0491** | **0.0429** |
| Median Test Miss Risk | 0.0488 | 0.0428 |
| Test Miss Risk Standard Deviation | 0.0081 | 0.0076 |
| Test Risk 10th Percentile | 0.0385 | 0.0326 |
| Test Risk 50th Percentile | 0.0488 | 0.0428 |
| Test Risk 82nd Percentile | 0.0560 | 0.0495 |
| Test Risk 90th Percentile | 0.0592 | 0.0523 |
| Test Risk 95th Percentile | 0.0632 | 0.0552 |
| **Split Coverage Rate ($\hat{R}_{\text{test}} \le 0.05$)** | **58.5% (117 / 200)** | **82.0% (164 / 200)** |
| **Mean Escalation Rate** | **78.4%** | **79.5%** |
| **Mean Realized Cascade MRR** | **0.3579** | **0.3569** |

---

## 5. Mathematical Resolution of the 82.0% vs. 90.0% Empirical Coverage Paradox

### 5.1 The Theoretical Guarantee vs. Empirical Observation
The mathematical guarantee provided by Clopper-Pearson inversion is:
$$\mathbb{P}_{D_{\text{cal}}}\left( R(\hat{\tau}) \le \alpha \right) \ge 1 - \delta = 0.90$$
where $R(\hat{\tau})$ is the **true population risk** (the expected risk over an infinite test set).

In the empirical experiment, what is evaluated is **NOT** the population risk $R(\hat{\tau})$, but the **finite test-set empirical risk**:
$$\hat{R}_{\text{test}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \ell(\hat{\tau}; X_i, Y_i), \quad N_{\text{test}} = 1{,}377$$

### 5.2 Binomial Sampling Noise Proof
Because $N_{\text{test}} = 1{,}377$ is finite, $\hat{R}_{\text{test}}$ fluctuates around the true population risk $R(\hat{\tau})$ according to a scaled binomial distribution:
$$N_{\text{test}} \hat{R}_{\text{test}} \sim \text{Binomial}\left(N_{\text{test}}, R(\hat{\tau})\right)$$

Suppose the calibration algorithm successfully chooses $\hat{\tau}$ such that the population risk is $R(\hat{\tau}) = 0.0450 \le 0.0500$ (satisfying the guarantee!).
What is the probability that the empirical test risk exceeds $0.0500$ simply due to sampling noise?
$$\mathbb{P}\left(\hat{R}_{\text{test}} > 0.0500 \mid R(\hat{\tau}) = 0.0450\right) = \mathbb{P}\left(\text{Binomial}(1377, 0.045) \ge 69\right)$$
Using the normal approximation to the binomial:
$$\mu = 1377 \times 0.045 = 61.965, \quad \sigma = \sqrt{1377 \times 0.045 \times 0.955} = \sqrt{59.176} = 7.693$$
$$z = \frac{68.5 - 61.965}{7.693} = \frac{6.535}{7.693} \approx 0.849 \implies \mathbb{P}\left(\hat{R}_{\text{test}} > 0.05\right) \approx 1 - \Phi(0.849) \approx 0.198 \text{ (approx. 20%)!}$$

### 5.3 Deduction
Even if the calibration procedure bounds the true population risk $R(\hat{\tau}) \le 0.05$ on $90\%$ of calibration sets, in approximately $15\text{--}20\%$ of those successful cases, finite test-set binomial sampling noise causes the observed test risk $\hat{R}_{\text{test}}$ to exceed $0.0500$.
Therefore, observing an empirical split satisfaction of **$82.0\%$** on a test set of size $N=1{,}377$ is **theoretically consistent with a true coverage probability of $1 - \delta = 0.90$**!

### 5.4 Why the Manuscript Language is Misleading
The manuscript states:
> "high-probability risk-controlling prediction sets (RCPS, $\delta=0.10$) bound empirical risk at 0.0429 with 82.0\% split coverage."

This phrasing conflates the theoretical $1-\delta = 0.90$ guarantee on population risk with the observed $82.0\%$ test-split coverage. A reviewer or auditor reading this without context would conclude that the algorithm failed to meet its nominal $90\%$ guarantee by $8.0$ percentage points.

---

## 6. Required Manuscript Replacement Text

### Section 5.C / Section 6.1 Replacement:
```latex
Under marginal conformal risk control \cite{bates2021distribution}, the calibrated threshold ($\bar{\tau}^* = 0.2347$) achieves a mean test risk of $0.0491 \le 0.05$ across 200 splits, satisfying marginal expectation control ($\mathbb{E}[R] \le \alpha$). Because test risks fluctuate symmetrically around their expectation on finite samples, 58.5\% of individual test splits achieve empirical risk $\le 0.05$. Under high-probability RCPS via exact Clopper-Pearson inversion ($\delta=0.10$), which certifies population risk control with probability $1-\delta = 0.90$, the mean test risk is conservatively bounded at $0.0429 < 0.05$. Across the 200 evaluation splits, 82.0\% of splits achieve an empirical test risk $\hat{R}_{\text{test}} \le 0.05$; the difference between observed 82.0\% test coverage and the nominal 90.0\% population bound is quantitatively accounted for by finite-sample binomial evaluation variance on the test partitions ($N_{\text{test}} = 1{,}377$).
```

---
**Auditor Signature:** Independent Statistical & Reproducibility Auditor  
**Date:** November 4, 2026

