# FINAL FORENSIC VERIFICATION SUITE — MEMORYVERSE / IJCDS

**Repository:** MemoryVerse  
**Manuscript Target:** International Journal of Computing and Digital Systems (IJCDS)  
**Verification Date:** November 2026  
**Audit Standard:** Independent Evidence-First Forensic Verification  

---

## 1. Directory Structure and Deliverables Inventory

This directory (`FINAL_FORENSIC_AUDIT/`) houses the complete independent forensic verification deliverables:

```
FINAL_FORENSIC_AUDIT/
├── README.md                                  # Package guide and index (this file)
├── FINAL_EXECUTIVE_DECISION.md                # Formal submission verdict (CONDITIONAL GO, 94/100)
│
├── 1. EVIDENCE INVENTORY & CLAIM LEDGERS
│   ├── PROJECT_EVIDENCE_INVENTORY.csv         # 7,876 project files with sizes, hashes, and roles
│   └── COMPLETE_CLAIM_EVIDENCE_LEDGER.csv     # Complete manuscript claim-to-evidence traceability
│
├── 2. QUERY & HUMAN EVALUATION INTEGRITY
│   ├── HUMAN_QUERY_PROVENANCE_REPORT.md       # Audit of 1,930 queries (GPT-4o, Human, Qwen2.5)
│   ├── HUMAN_QUERY_METRICS.csv                # Recalculated bootstrap metrics across overlap bins
│   ├── HUMAN_ANNOTATION_PROVENANCE_REPORT.md  # Audit of authentic N=110 vs simulated rater code
│   └── HUMAN_ANNOTATION_RECALCULATION.csv     # Cohen's kappa recalculations by subset and dimension
│
├── 3. STATISTICAL & MATHEMATICAL VERIFICATION
│   ├── STATISTICAL_AUDIT_REPORT.md            # Audit of 35 hypotheses, Holm-Bonferroni & TOST
│   ├── STATISTICAL_RECALCULATION.csv          # Corrected Holm-Bonferroni ledger (MiniLM & LLM Bridge)
│   ├── CROSSOVER_AUDIT_REPORT.md              # Analysis of overlap dose-response and generator effect
│   ├── RCPS_VERIFICATION_REPORT.md            # Finite-sample proof of 82% RCPS coverage
│   └── RCPS_SPLIT_RESULTS.csv                 # Complete 200 calibration/test split records
│
├── 4. RETRIEVAL & EXPERIMENT VALIDITY
│   ├── RETRIEVAL_RAG_CLAIM_AUDIT.csv          # Audit of retrieval claims across SROIE, COCO, DocVQA, FUNSD
│   └── EXPERIMENT_VALIDITY_REPORT.md          # Comprehensive empirical validity report
│
├── 5. REPRODUCIBILITY & GPU EXECUTION
│   ├── GPU_EXECUTION_STATUS.csv               # Audit of GPU models and latency status
│   ├── KAGGLE_RUN_INSTRUCTIONS.md             # Guide for executing GPU experiments on Kaggle
│   └── KAGGLE_NOTEBOOKS/                      # Self-contained Kaggle GPU notebooks
│       ├── KAGGLE_COLQWEN2_CASCADE_BENCHMARK.ipynb
│       └── KAGGLE_EXTERNAL_DATASET_GPU_VALIDATION.ipynb
│
├── 6. MANUSCRIPT COMPLIANCE & EDITORIAL INTEGRITY
│   ├── MANUSCRIPT_COMPLIANCE_REPORT.md        # LaTeX label/ref audit and citation checks
│   └── MANUSCRIPT_CLAIM_CORRECTIONS.csv       # Itemized text corrections and status
│
├── 7. THREE ADVERSARIAL PEER REVIEWS
│   ├── REVIEW_IR.md                           # Information Retrieval expert review
│   ├── REVIEW_STATISTICS.md                   # Mathematical & statistical auditor review
│   └── REVIEW_REPRODUCIBILITY.md              # Software engineering & reproducibility review
│
└── SHAREABLE_AUDIT_PACKAGE.zip                # Compact zip bundle for archiving and distribution
```

---

## 2. Executive Summary of Core Findings

1. **Query Counts Reconciled:** The manuscript draft's $1,730$ queries represented an incomplete count omitting the companion handwritten human queries. The true count is exactly **$1,930$** evaluations ($1,224$ GPT-4o-mini + $400$ Human-authored + $306$ Qwen2.5-7B).
2. **Human Evaluation Quarantined:** Lines 620–648 of `scripts/gapfill_t2_05_e2e_rag_accuracy.py` programmatically synthesized raters for document QA. In contrast, `master_reviewer_1.csv` and `master_reviewer_2.csv` contain authentic human ratings on $N=110$ multimodal personal photo and relational queries ($\kappa = 0.9031$). The manuscript strictly decouples automated document QA ($N=150$, ANLS/EM) from the authentic human evaluation ($N=110$).
3. **Statistical Corrections:** Correcting an unsorted table bug confirms that MiniLM reranking is **statistically significant** ($p_{\text{Holm}} = 0.0450 \le 0.05$). Correcting a family multiplier bug confirms that LLM Bridging on $N=50$ is **non-significant** ($p_{\text{Holm}} = 0.1075 > 0.05$). Both TOST equivalence tests (SigLIP on COCO, PRF on relational multi-hop) are verified at $\alpha=0.01$.
4. **Latency Truth:** The 16.20 ms cascade latency is an **analytical model sum** (0.87 ms CPU + scaled GPU reranking) and is transparently designated as such in the revised manuscript.
5. **Crossover Re-interpretation:** The 52% crossover threshold is generator-dependent and driven by synthetic entity substitution. On authentic human queries, Character BM25 and Word BM25 perform comparably across all overlap levels (MRRs 0.555–0.577).

---

## 3. Recommended Submission Action

Apply the single citation patch (`\cite{zheng2023judging}`) to Section IV-F of `ijcds_body.tex`, synchronize the submission package source, and submit with full scientific confidence.

