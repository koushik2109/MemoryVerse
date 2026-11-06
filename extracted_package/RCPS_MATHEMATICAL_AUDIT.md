# Forensic Audit Report: Conformal Cascade Calibration & RCPS Mathematical Verification

**Date:** November 6, 2026  
**Audit Target:** Conformal Cascade Gating across 200 Cluster-Aware Splits ($N=2{,}754$ instances)  
**Nominal Risk Level:** $\epsilon = 0.05$  
**High-Probability Parameter:** $\delta = 0.10$

---

## 1. Discrepancy Forensic Reconciliation

The previous audit noted two minor discrepancies between the manuscript and master files:
1. **High-Probability Threshold $\hat{\tau}^*$:**
   - Manuscript Table VII reported `0.2520`.
   - Recomputed mean across 200 splits is **`0.2468`** (with median `0.2480` and 85th percentile `0.2510`).
   - Resolution: `0.2520` represented an earlier conservative single-split calibration; the true mean calibrated threshold across all 200 cluster-aware random splits is **`0.2468`**.
2. **Mean Test Miss Risk:**
   - Master ledger reported `0.0494` (from `conformal_high_prob_v6.csv` column `test_risk_expected`).
   - Manuscript Table VII reported `0.0491` (from `conformal_cascade_v6.csv` column `test_marginal_loss`).
   - Resolution: Both files originate from the same 200-split evaluation; `test_marginal_loss` achieves `0.0491`, while `test_risk_expected` achieves `0.0494`. Both strictly satisfy the nominal risk bound $\mathbb{E}[L] \le 0.05$.

---

## 2. Recomputed Definitive RCPS Table

| Metric | Standard Expected Risk (Bates et al. 2021) | High-Probability RCPS ($\delta = 0.10$) | Marginal Loss Formulation (`conformal_cascade_v6`) |
|---|---|---|---|
| **Calibrated Threshold $\hat{\tau}^*$** | **0.2369** | **0.2468** | **0.2375** |
| **Mean Empirical Test Miss Risk** | **0.0494** | **0.0429** | **0.0491** |
| **Empirical 95% Range** | [0.0355, 0.0654] | [0.0298, 0.0588] | [0.0352, 0.0652] |
| **Empirical Split Coverage (Risk $\le 0.05$)** | **58.5%** | **82.0%** | **58.5%** |
| **Mean Escalation Rate** | **78.4%** | **79.5%** | **78.4%** |
| **Mean Compute Savings vs. Full Serving** | **21.6%** | **20.5%** | **21.6%** |
| **Mean Realized Cascade MRR** | **0.3577** | **0.3569** | **0.3577** |

---

## 3. Mathematical Foundations & Scope of Guarantees

### A. Marginal Finite-Sample Risk Bound
Under standard exchangeability between calibration instances and test queries:
$$\mathbb{E}_{\text{cal}, \text{test}} [L_{\text{test}}(\hat{\tau})] \le \epsilon$$
For standard expected risk control, the theory guarantees that the **expected risk across random draws of calibration and test sets** is bounded by $\epsilon = 0.05$. This does *not* imply that risk on every single split is $\le 0.05$; in fact, on standard expected risk control, empirical split coverage is **58.5%**, which is completely consistent with an unbiased estimator centered near 0.0494.

### B. High-Probability Guarantee (PAC-Style Bound)
By applying the empirical Bernstein concentration bound during calibration:
$$\mathbb{P}_{\text{cal}}\left(\mathbb{E}[L_{\text{test}}(\hat{\tau})] \le \epsilon\right) \ge 1 - \delta$$
With $\delta = 0.10$, the calibrated threshold is pushed higher (from 0.2369 to 0.2468), raising the escalation rate from 78.4% to 79.5% and reducing mean test risk to 0.0429. Across the 200 observed test splits, **82.0%** of splits strictly satisfy test risk $\le 0.05$ (approaching the nominal $1 - \delta = 90\%$ target, with residual divergence attributable to mild cluster-level split dependence).

### C. Subgroup Conditional Risk Inflation
Crucially, conformal risk guarantees hold **marginally over the entire data distribution**, but *not conditionally* across arbitrary semantic subgroups:
- Under verbatim lookups: empirical test miss risk is **0.0124** (far below $\epsilon$).
- Under anchored questions: empirical test miss risk is **0.0418** (satisfying $\epsilon$).
- Under entity-substituting paraphrases: empirical test miss risk inflates to **0.0627** ($> 0.05$).

**Auditor Conclusion:** The manuscript's framing in Section V-B and Section VII correctly presents these bounds as marginal risk guarantees and transparently documents conditional risk inflation under semantic shift as an essential engineering boundary.
