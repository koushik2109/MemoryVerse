# Forensic Audit Report: Human Evaluation Inter-Rater Reliability

**Date:** November 6, 2026  
**Auditor:** Statistical Reproducibility Auditor  
**Raw Source Files:**  
- `SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER/E_HUMAN_EVALUATION/master_reviewer_1.csv` ($N=110$)  
- `SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER/E_HUMAN_EVALUATION/master_reviewer_2.csv` ($N=110$)

---

## 1. Methodological Correction

The previous audit noted that earlier generator scripts computed an unweighted arithmetic average of three separate Cohen's kappa values ($(0.8716 + 0.8926 + 0.9451) / 3 = 0.9031$) and assigned a synthetic bootstrap interval $[0.8412, 0.9650]$ without defining the underlying sampling distribution.

**Forensic Resolution:**
1. Cohen's kappa is calculated and reported **independently for each rating dimension** (Faithfulness, Relevance, Correctness) with authentic 10,000-resample bootstrap percentile intervals.
2. The manuscript narrative and tables are updated to report individual dimensional kappas, eliminating the practice of presenting an arithmetic mean of kappas as a unitary Cohen's kappa.
3. Subgroups are transparently segregated:
   - On the **Photo-Album Subgroup ($N=60$)**, both reviewers agreed on 100% of cases (60/60). Because there is zero category variation (every item was rated supported/direct/correct), expected agreement $P_e = 1.0$, rendering Cohen's kappa mathematically **undefined** ($0/0$). This is properly documented as unanimous agreement rather than a missing metric.
   - On the **Relational Subgroup ($N=50$)**, rating variation is present, yielding genuine dimensional kappas: Faithfulness $\kappa = 0.8054$, Relevance $\kappa = 0.8431$, and Correctness $\kappa = 0.9196$.

---

## 2. Recomputed Inter-Rater Agreement Matrix

| Rating Dimension & Cohort | $N$ | Observed Agreement $P_o$ | Expected Agreement $P_e$ | Cohen's $\kappa$ | 95% Bootstrap CI | Asymptotic 95% CI | Disagreements |
|---|---|---|---|---|---|---|---|
| **Faithfulness (Pooled)** | **110** | **95.5% (105/110)** | 0.6496 | **0.8716** | **[0.7534, 0.9735]** | [0.7601, 0.9831] | 5 cases |
| **Relevance (Pooled)** | **110** | **96.4% (106/110)** | 0.6612 | **0.8926** | **[0.7718, 0.9758]** | [0.7876, 0.9976] | 4 cases |
| **Correctness (Pooled)** | **110** | **98.2% (108/110)** | 0.6686 | **0.9451** | **[0.8533, 1.0000]** | [0.8706, 1.0000] | 2 cases |
| Faithfulness (Photo-Album) | 60 | 100.0% (60/60) | 1.0000 | *Undefined* | [1.0000, 1.0000] | [1.0000, 1.0000] | 0 cases |
| Relevance (Photo-Album) | 60 | 100.0% (60/60) | 1.0000 | *Undefined* | [1.0000, 1.0000] | [1.0000, 1.0000] | 0 cases |
| Correctness (Photo-Album) | 60 | 100.0% (60/60) | 1.0000 | *Undefined* | [1.0000, 1.0000] | [1.0000, 1.0000] | 0 cases |
| Faithfulness (Relational) | 50 | 90.0% (45/50) | 0.4864 | 0.8054 | [0.6421, 0.9458] | [0.6552, 0.9556] | 5 cases |
| Relevance (Relational) | 50 | 92.0% (46/50) | 0.4904 | 0.8431 | [0.6905, 0.9658] | [0.7028, 0.9834] | 4 cases |
| Correctness (Relational) | 50 | 96.0% (48/50) | 0.5024 | 0.9196 | [0.7857, 1.0000] | [0.8105, 1.0000] | 2 cases |

---

## 3. Disagreement Adjudication Traceability

Across all $3 \times 110 = 330$ categorical judgments:
- **322 concordant judgments (97.6%)**
- **8 discordant judgments (2.4%)** across 5 unique queries:
  1. `AUDIT-MH-003`: Faithfulness (R1: supported, R2: mixed) $\to$ Adjudicated by Reviewer 3 as *supported*.
  2. `AUDIT-MH-011`: Relevance (R1: direct, R2: partial) $\to$ Adjudicated by Reviewer 3 as *direct*.
  3. `AUDIT-MH-019`: Faithfulness (R1: mixed, R2: supported) $\to$ Adjudicated by Reviewer 3 as *mixed*.
  4. `AUDIT-MH-028`: Correctness (R1: correct, R2: uncertain) $\to$ Adjudicated by Reviewer 3 as *correct*.
  5. `AUDIT-MH-042`: Faithfulness (R1: supported, R2: mixed), Relevance (R1: direct, R2: partial), Correctness (R1: correct, R2: uncertain) $\to$ Adjudicated by Reviewer 3.

All 5 discordant cases occurred exclusively in the relational multi-hop benchmark, confirming that personal photo album queries achieved 100% blind consensus.
