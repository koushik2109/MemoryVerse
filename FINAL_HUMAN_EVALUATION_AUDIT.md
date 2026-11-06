# FINAL HUMAN EVALUATION & REVIEWER INDEPENDENCE AUDIT

**Audit Scope:** 110-answer dual-rater multimodal retrieval study (`ai_engine/rag/data/master_reviewer_1.csv` & `master_reviewer_2.csv`)  
**Comparison:** Independent audit separating the 110-answer multimodal study from the 150-case automated document QA benchmark  
**Lead Auditor:** Senior Research Integrity & Reproducibility Auditor  
**Date:** November 2026  
**Status:** FULLY VERIFIED, REPRODUCED FROM RAW ANNOTATION RECORDS  

---

## 1. Executive Summary & Audit Verification

1. **Cohort Separation:**
   - **The 110-Answer Multimodal Study:** 100% authentic human evaluation derived from `master_reviewer_1.csv` and `master_reviewer_2.csv`. Comprises $N=60$ personal photo-album answers and $N=50$ relational multi-hop answers.
   - **The 150-Query Document-RAG Benchmark:** 100% automated evaluation based on Average Normalized Levenshtein Similarity (ANLS) and Exact Match against ground-truth SROIE/DocVQA fields. The simulated rater generation code in `scripts/gapfill_t2_05_e2e_rag_accuracy.py` (lines 620–648) has been completely decommissioned and excised from all human evaluation claims.

2. **Reviewer Identity and Independence:**
   - **Reviewer 1:** Primary researcher and system developer.
   - **Reviewer 2:** Second affiliated researcher.
   - **Independence Level:** Internal independent rater (blinded to retrieval model and candidate pipeline), NOT an "independent external expert" or professional commercial auditing service. The manuscript must use the accurate terminology: *"two blinded evaluators: the primary system developer and a second evaluator"*.

3. **Blinding Protocol & Architecture Concealment:**
   - Both reviewers scored responses independently via standardized HTML forms (`master_reviewer_form.html`, `album*_audit/reviewer_*_form.html`).
   - The underlying retrieval architecture (e.g., whether the context was retrieved via Extractive PRF, Generative LLM bridging, or BM25) was completely concealed from both evaluators.

4. **Resolution of Disagreements:**
   - No post-hoc forced reconciliation was applied. Raw independent discordant annotations are preserved in the checked-in CSVs, providing genuine inter-rater variance.

---

## 2. Mathematical Reproduction of Inter-Rater Reliability

All agreement statistics and Cohen's $\kappa$ coefficients have been independently recalculated using scikit-learn (`cohen_kappa_score` and `confusion_matrix`) directly from `master_reviewer_1.csv` and `master_reviewer_2.csv`:

### A. Photo-Album Cohort ($N=60$, Probes `L10-A1-001` through `L10-A3-020`)
- **Observed Agreement:**
  - Faithfulness: **100.00%** (60/60 concordant; both raters assigned `supported` to all 60 cases).
  - Relevance: **100.00%** (60/60 concordant; both raters assigned `direct` to all 60 cases).
  - Correctness: **100.00%** (60/60 concordant; both raters assigned `correct` to all 60 cases).
- **Cohen's $\kappa$ Determination:**
  Because both raters assigned 100% of observations to a single class, expected agreement $p_e = 1.0 \times 1.0 = 1.0$, rendering Cohen's $\kappa = \frac{p_o - p_e}{1 - p_e} = \frac{0}{0}$ **mathematically undefined (degenerate)**.
- **Reporting Rule:** The manuscript must report **100% observed agreement** directly, explicitly noting that chance-adjusted agreement ($\kappa$) is undefined due to zero marginal variance. Reporting $\kappa = 1.0$ is scientifically invalid and prohibited.

---

### B. Relational Multi-Hop Cohort ($N=50$, Probes `AUDIT-001` through `AUDIT-050`)
Unlike the photo albums, relational queries introduced genuine semantic complexity, resulting in natural rater discordance:

#### 1. Faithfulness (Observed Agreement: 90.00%, Cohen's $\kappa = \mathbf{0.8054}$)
- 45 concordant pairs, 5 discordant pairs:
  - Rater 1: 23 `supported`, 27 `mixed`, 0 `unsupported`.
  - Rater 2: 27 `supported`, 22 `mixed`, 1 `unsupported`.
- **Confusion Matrix:**
  $$\begin{pmatrix} & \text{R2: mixed} & \text{R2: supported} & \text{R2: unsupported} \\ \text{R1: mixed} & 22 & 4 & 1 \\ \text{R1: supported} & 0 & 23 & 0 \\ \text{R1: unsupported} & 0 & 0 & 0 \end{pmatrix}$$

#### 2. Relevance (Observed Agreement: 92.00%, Cohen's $\kappa = \mathbf{0.8431}$)
- 46 concordant pairs, 4 discordant pairs:
  - Rater 1: 25 `direct`, 25 `partial`, 0 `off_topic`.
  - Rater 2: 28 `direct`, 21 `partial`, 1 `off_topic`.
- **Confusion Matrix:**
  $$\begin{pmatrix} & \text{R2: direct} & \text{R2: off\_topic} & \text{R2: partial} \\ \text{R1: direct} & 25 & 0 & 0 \\ \text{R1: off\_topic} & 0 & 0 & 0 \\ \text{R1: partial} & 3 & 1 & 21 \end{pmatrix}$$

#### 3. Correctness (Observed Agreement: 96.00%, Cohen's $\kappa = \mathbf{0.9196}$)
- 48 concordant pairs, 2 discordant pairs:
  - Rater 1: 26 `correct`, 24 `uncertain`, 0 `incorrect`.
  - Rater 2: 28 `correct`, 22 `uncertain`, 0 `incorrect`.
- **Confusion Matrix:**
  $$\begin{pmatrix} & \text{R2: correct} & \text{R2: uncertain} \\ \text{R1: correct} & 26 & 0 \\ \text{R1: uncertain} & 2 & 22 \end{pmatrix}$$

- **Cohort Composite Mean $\kappa$:** $\frac{0.8054 + 0.8431 + 0.9196}{3} = \mathbf{0.8560}$.

---

### C. Pooled Cohort ($N=110$, Full Benchmark)
Calculated directly by pooling all 110 records (not by averaging across cohorts):
- **Faithfulness:** Observed Agreement = **95.45%**, Cohen's $\kappa = \mathbf{0.8716}$ [95% Bootstrap CI: 0.753, 0.974].
- **Relevance:** Observed Agreement = **96.36%**, Cohen's $\kappa = \mathbf{0.8926}$ [95% Bootstrap CI: 0.772, 0.976].
- **Correctness:** Observed Agreement = **98.18%**, Cohen's $\kappa = \mathbf{0.9451}$ [95% Bootstrap CI: 0.853, 1.000].
- **Arithmetic Composite Mean $\kappa$:** $\frac{0.8716 + 0.8926 + 0.9451}{3} = \mathbf{0.9031}$.

---

## 3. Mandatory Manuscript Disclosures

1. In **Section VI-B (Page 14)**:
   - State clearly that the evaluation evaluated responses from personal photo albums ($N=60$) and relational retrieval ($N=50$).
   - Explicitly report that photo-album responses yielded 100% observed agreement across all three dimensions, while relational queries yielded $\kappa = 0.8054$ for faithfulness, $0.8431$ for relevance, and $0.9196$ for correctness.
   - Disclose that raters were the primary system developer and a second blinded evaluator.
2. In the **Abstract (Line 26)**:
   - Ensure the distinction between the automated document QA audit ($N=150$) and the dual-reviewer multimodal audit ($N=110$, mean $\kappa = 0.9031$) is maintained with zero ambiguity.

