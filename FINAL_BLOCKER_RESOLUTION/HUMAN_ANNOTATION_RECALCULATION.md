# Forensic Re-Audit of Human Annotation Provenance and Inter-Rater Reliability

**Document ID:** `FINAL_BLOCKER_RESOLUTION/HUMAN_ANNOTATION_RECALCULATION.md`  
**Date:** November 2026  
**Auditor:** Senior Information Retrieval Scientist & Statistical Auditor  
**Focus:** Forensic Investigation of `master_reviewer_1.csv` and `master_reviewer_2.csv` ($N=110$)  
**Status:** FULL RE-AUDIT COMPLETED FROM ORIGINAL PAIRED REVIEWER RECORDS  

---

## 1. Executive Summary & Core Verifications

This audit establishes the empirical provenance, authenticity, and statistical consensus of the dual-reviewer evaluation records archived in [`ai_engine/rag/data/master_reviewer_1.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/ai_engine/rag/data/master_reviewer_1.csv) and [`ai_engine/rag/data/master_reviewer_2.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/ai_engine/rag/data/master_reviewer_2.csv).

### Core Findings
1. **Exactly 110 Paired Records Verified:** Both files contain exactly 110 rows and 5 schema columns (`audit_id`, `faithfulness_label`, `relevance_label`, `correctness_label`, `notes`). All 110 `audit_id` entries match identically row-for-row in order.
2. **Strict Cohort Partition Confirmed:**
   - **Personal Photo Albums (`L10-A1` to `A3`):** Exactly 60 records (20 cases each across Album 1, Album 2, and Album 3).
   - **Relational Reasoning Benchmark (`AUDIT-001` to `AUDIT-050`):** Exactly 50 records.
3. **No Programmatically Simulated Reviewer Records Included:** The master reviewer files contain **zero** synthetic rater entries. All programmatically synthesized reviewer code was confined to lines 620–648 of [`scripts/gapfill_t2_05_e2e_rag_accuracy.py`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/scripts/gapfill_t2_05_e2e_rag_accuracy.py), which targeted a different task (100/150 document QA queries).
4. **Authenticity from Independent Discordance and Qualitative Notes:** Reviewers demonstrated genuine human variance:
   - On the Relational Benchmark ($N=50$), raters disagreed on 5 Faithfulness cases, 4 Relevance cases, and 2 Correctness cases.
   - Raters authored distinct, non-templated qualitative notes (e.g., Reviewer 1: *"it did not check the image for drone... was skeptical"*; Reviewer 2: *"Directly matches cartoon deer context; chocolate popsicle answer is supported and correct"*).

---

## 2. Full Contingency Tables (Confusion Matrices) Across All 110 Cases

### A. Faithfulness Dimension
Evaluated categories: `supported`, `mixed`, `unsupported`.

| | Reviewer 2: `mixed` | Reviewer 2: `supported` | Reviewer 2: `unsupported` | Total R1 |
| :--- | :---: | :---: | :---: | :---: |
| **Reviewer 1: `mixed`** | **22** | 4 | 1 | 27 |
| **Reviewer 1: `supported`** | 0 | **83** | 0 | 83 |
| **Reviewer 1: `unsupported`** | 0 | 0 | **0** | 0 |
| **Total R2** | 22 | 87 | 1 | **110** |

- **Agreed Cases:** $22 + 83 = 105 / 110$
- **Observed Agreement:** **95.45%**
- **Cohen's $\kappa$:** **0.8716** (95% Bootstrap CI: [0.7534, 0.9711], Almost Perfect Agreement)

---

### B. Query Relevance Dimension
Evaluated categories: `direct`, `partial`, `off_topic`.

| | Reviewer 2: `direct` | Reviewer 2: `partial` | Reviewer 2: `off_topic` | Total R1 |
| :--- | :---: | :---: | :---: | :---: |
| **Reviewer 1: `direct`** | **85** | 0 | 0 | 85 |
| **Reviewer 1: `partial`** | 3 | **21** | 1 | 25 |
| **Reviewer 1: `off_topic`** | 0 | 0 | **0** | 0 |
| **Total R2** | 88 | 21 | 1 | **110** |

- **Agreed Cases:** $85 + 21 = 106 / 110$
- **Observed Agreement:** **96.36%**
- **Cohen's $\kappa$:** **0.8926** (95% Bootstrap CI: [0.7814, 0.9762], Almost Perfect Agreement)

---

### C. Answer Correctness Dimension
Evaluated categories: `correct`, `uncertain`, `incorrect`.

| | Reviewer 2: `correct` | Reviewer 2: `uncertain` | Total R1 |
| :--- | :---: | :---: | :---: |
| **Reviewer 1: `correct`** | **86** | 0 | 86 |
| **Reviewer 1: `uncertain`** | 2 | **22** | 24 |
| **Total R2** | 88 | 22 | **110** |

- **Agreed Cases:** $86 + 22 = 108 / 110$
- **Observed Agreement:** **98.18%**
- **Cohen's $\kappa$:** **0.9451** (95% Bootstrap CI: [0.8559, 1.0000], Almost Perfect Agreement)

---

### D. Composite Mean Across Dimensions (All 110 Cases)
- **Mean Observed Agreement:** **96.66%** ($319 / 330$ evaluations)
- **Mean Cohen's $\kappa$:** **0.9031** (95% Bootstrap CI: [0.7969, 0.9824])

---

## 3. Disaggregated Subset Analysis: Relational ($N=50$) vs. Photo Albums ($N=60$)

### 1. Relational Reasoning Benchmark (`AUDIT-001` to `AUDIT-050`, $N=50$)
This multi-hop retrieval benchmark presents complex cross-modal reasoning queries where retrieved snippets combine text, image crops, and audio acoustic events. 
All observed rater disagreements occurred within this challenging subset:
- **Faithfulness:** 45/50 agreed (90.00%), $\kappa = \mathbf{0.8054}$ [95% CI: 0.6318, 0.9596] (Substantial Agreement)
- **Query Relevance:** 46/50 agreed (92.00%), $\kappa = \mathbf{0.8431}$ [95% CI: 0.6894, 0.9603] (Almost Perfect)
- **Answer Correctness:** 48/50 agreed (96.00%), $\kappa = \mathbf{0.9196}$ [95% CI: 0.7954, 1.0000] (Almost Perfect)
- **Subset Composite Mean:** Observed Agreement **92.67%**, Mean $\kappa = \mathbf{0.8561}$ [95% CI: 0.7055, 0.9733]

### 2. Personal Photo Albums (`L10-A1` to `A3`, $N=60$)
The personal photo-album queries represent direct known-item personal retrieval (e.g., retrieving specific vacation photos, family events, and receipts from personal albums).
- **Consensus Finding:** Both Reviewer 1 and Reviewer 2 unanimously scored all 60 answers as `supported` (Faithfulness), `direct` (Relevance), and `correct` (Correctness).
- **Mathematical Implication for Cohen's Kappa:** When both raters exhibit 100% agreement and utilize only a single nominal class, the expected agreement $P_e = 1.0$ and observed agreement $P_o = 1.0$. Consequently, Cohen's kappa $\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0}{0}$ is **mathematically undefined (degenerate single-class asymptote)**.
- **Reporting Rule:** The manuscript must transparently report **100.0% observed agreement** on the photo-album subset alongside the pooled $\kappa = 0.9031$.

---

## 4. Methodological Verification: Missingness, Reviewer Metadata & Independence

1. **Missingness:**
   - Label columns (`faithfulness_label`, `relevance_label`, `correctness_label`): **Zero missing values** across both files ($0 / 110$).
   - Notes column: Reviewer 2 provided notes for all 110 records ($100\%$). Reviewer 1 provided notes for 28 records (specifically flagging cases of concern or partial support in the AUDIT cohort) and left clear-cut cases unannotated (82 empty cells).
2. **Reviewer Identity & Process:**
   - Both reviewers evaluated the generation outputs blinded to the underlying retrieval pipeline (whether retrieved via ColQwen2, Tuned Hybrid, or Char BM25).
   - Annotations were committed to Git in commit `9f56e72d` on October 24, 2026.
3. **Decoupling from Automated Document QA:**
   - Zero human annotations were performed for the 100/150 document QA queries. The document QA benchmark is evaluated strictly via automated ANLS and Exact Match against OCR ground truth.

