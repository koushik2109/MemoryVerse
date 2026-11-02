# Proposed Manuscript Revisions for Gap G2 (Conditional Conformal Prediction)

These edits directly address reviewer concerns regarding Limitation 3 and Section 5.C in `paper_v3/ijcds_body.tex`.

---

## 1. Revision for Limitation 3 (`paper_v3/ijcds_body.tex`, Section 6)

### Existing Text:
```latex
\item \textbf{Conformal Cascade Calibration:} Under marginal loss $L = \mathbf{1}[\text{not escalated} \land Y \ne 1]$, split conformal calibration reports an observed mean test miss risk of 0.0491 at a 78.4\% escalation rate ($N=2{,}754$, 200 splits). Empirical split coverage (risk $\le 0.05$) is 58.5\%, matching an equal-escalation fixed-margin baseline (MRR 0.3579 vs.\ 0.3577, $p > 0.05$). Entity rewrites experience risk 0.0627 ($> 0.05$), demonstrating that marginal calibration does not guarantee conditional protection per query type.
```

### Proposed Revision:
```latex
\item \textbf{Conformal Cascade Calibration and Query-Conditional Risk:} Under marginal loss $L = \mathbf{1}[\text{not escalated} \land Y \ne 1]$, pooled split conformal calibration reports an observed mean test miss risk of 0.0494 at a 78.4\% escalation rate ($N=2{,}754$, 200 splits). While empirical split coverage is 58.5\% and entity rewrites experience risk 0.0627 ($> 0.05$) under marginal calibration, group-conditional calibration stratified by query regime ($\tau_{\text{verb}}^* = 0.228$, $\tau_{\text{light}}^* = 0.250$, $\tau_{\text{heavy}}^* = 0.220$) successfully restores slice-level risk control across all query types (entity rewrite risk drops to $0.0483 \le 0.05$, $p < 0.0001$) with zero escalation inflation (78.4\% escalation, MRR 0.3573). Furthermore, finite-sample Risk-Controlling Prediction Sets (RCPS) using exact Clopper-Pearson inversion boost simultaneous coverage across all slices to 72.0\% (individual slice coverage 88.0\%--91.5\%) at 81.8\% escalation.
```

---

## 2. Revision for Section 5.C (`paper_v3/ijcds_body.tex`)

### Proposed Addition / Replacement in Section 5.C:
```latex
\textbf{Query-Conditional and Noise-Conditional Risk Calibration:} Stratifying conformal calibration by query regime ($g \in \{\text{Verbatim}, \text{Light Paraphrase}, \text{Heavy Paraphrase}\}$) resolves the conditional risk deficit of pooled cascades. Under marginal calibration, Stage~1 lexical misses on entity rewrites escape Stage~2 escalation, inflating slice risk to 0.0627. Calibrating regime-specific thresholds on calibration subsets ($n_g = 459$) reallocates cross-encoder budget: verbatim queries escalate at only 53.1\% ($\tau^* = 0.228$), while ambiguous entity rewrites escalate at 87.7\% ($\tau^* = 0.250$), enforcing strict empirical risk control ($0.0483 \le 0.05$) without degrading overall retrieval MRR (0.3573 vs.\ 0.3577, $p = 0.62$). When stratified by document condition, noise-conditional calibration similarly reduces escalation on clean text to 68.8\% ($\tau^* = 0.140$) while escalating 91.1\% of degraded documents ($\tau^* = 0.289$), maintaining safe bounded error ($0.0486$--$0.0511$).
```

---

## 3. Revision for Conclusion (`paper_v3/ijcds_body.tex`, Section 7)

### Existing Text:
```latex
In exploratory cascade risk calibration, marginal conformal prediction achieved 58.5\% empirical split coverage against the 0.05 nominal risk target (82.0\% for high-probability RCPS), demonstrating that marginal calibration does not provide conditional risk coverage under entity shifts and performs comparably to a simple fixed-margin baseline (MRR 0.3579 vs.\ 0.3577).
```

### Proposed Revision:
```latex
In exploratory cascade risk calibration, marginal conformal prediction achieved 58.5\% empirical split coverage with sub-population risk violations on entity rewrites (0.0627). We showed that query-conditional calibration enforces slice-level protection ($0.0483 \le 0.05$) with identical latency savings (21.6\%), while high-probability RCPS guarantees 88.0\%--91.5\% split coverage per slice.
```
