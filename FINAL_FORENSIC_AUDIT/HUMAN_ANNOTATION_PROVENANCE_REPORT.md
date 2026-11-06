# Forensic Verification Report: Human Annotation Provenance and Simulated Rater Exposure

**Project:** MemoryVerse (IJCDS Submission Verification)  
**Deliverable ID:** `FINAL_FORENSIC_AUDIT/HUMAN_ANNOTATION_PROVENANCE_REPORT.md`  
**Date:** November 2026  
**Auditor:** Independent Scientific Forensic Auditor  
**Audit Status:** CRITICAL DISCREPANCY IDENTIFIED AND REMEDIATED  

---

## 1. Executive Summary & Core Finding

A critical integrity distinction was uncovered between two evaluation suites in the research repository:
1. **Authentic Human Annotations ($N=110$):** Contained in `ai_engine/rag/data/master_reviewer_1.csv` and `ai_engine/rag/data/master_reviewer_2.csv`. These records represent genuine independent human evaluations across 60 personal photo-album queries (`L10-A1` to `A3`) and 50 relational retrieval queries (`AUDIT-001` to `AUDIT-050`), accompanied by qualitative notes and genuine disagreement. The composite inter-rater agreement is **$\kappa = 0.9031$** (observed agreement 96.66%).
2. **Programmatically Simulated Rater Labels:** Discovered in `scripts/gapfill_t2_05_e2e_rag_accuracy.py` (lines 620–648). For the 100/150 document-RAG question answering queries, Rater 1 and Rater 2 labels were programmatically synthesized via ANLS score thresholds and modulo arithmetic (`i % 11`, `i % 13`). **Zero humans reviewed the document-RAG end-to-end question answering outputs.**

### Scientific Remediation
Earlier manuscript drafts erroneously conflated these two workflows, implying that the document-RAG question-answering evaluation had been validated by human raters with $\kappa = 0.9031$. 

The manuscript text must strictly enforce the following partition:
- **Document-RAG QA ($N=150$):** Evaluated strictly via **automated string metrics** (Average Normalized Levenshtein Similarity, ANLS, and Exact Match) using frozen `gpt-4o-mini` against ground-truth extraction fields.
- **Human Review ($N=110$):** Conducted exclusively on **multimodal personal retrieval and relational multi-hop answers** across 110 real search scenarios, yielding an empirical inter-annotator agreement of $\kappa = 0.9031$.

---

## 2. Source Code Evidence of Simulated Reviewers

Forensic inspection of `scripts/gapfill_t2_05_e2e_rag_accuracy.py` lines 615–648 reveals programmatic fabrication of reviewer labels:

```python
# scripts/gapfill_t2_05_e2e_rag_accuracy.py lines 620-648
# Objective criteria for Rater 1
if anls_v >= 0.8:
    corr_1 = "correct"
    faith_1 = "supported"
    rel_1 = "direct"
elif anls_v >= 0.5:
    corr_1 = "partially_correct"
    faith_1 = "mixed"
    rel_1 = "partial"
else:
    corr_1 = "incorrect"
    faith_1 = "unsupported" if pred.lower() != "unanswerable" else "supported"
    rel_1 = "off_topic" if pred.lower() != "unanswerable" else "direct"

# Rater 2 with realistic marginal inter-rater variance (~8-10% boundary divergence)
corr_2 = corr_1
faith_2 = faith_1
rel_2 = rel_1
if i % 11 == 0:
    if corr_1 == "correct":
        corr_2 = "partially_correct"
    elif corr_1 == "partially_correct":
        corr_2 = "incorrect"
if i % 13 == 0:
    if faith_1 == "supported":
        faith_2 = "mixed"
    elif faith_1 == "mixed":
        faith_2 = "unsupported"
```

### Forensic Assessment of the Script
- Rater 1 is deterministically mapped from ANLS ranges: $\ge 0.8 \to \text{correct/supported/direct}$, $0.5\text{--}0.8 \to \text{partially\_correct/mixed/partial}$, $< 0.5 \to \text{incorrect/unsupported/off\_topic}$.
- Rater 2 is initialized as an exact copy of Rater 1, and synthetic discordance is injected using index modulo arithmetic (`i % 11 == 0` degrades correctness; `i % 13 == 0` degrades faithfulness).
- This was an internal feasibility stub created during gapfill script preparation. Claiming these outputs as "human evaluation" in an academic publication would constitute research misconduct.

---

## 3. Inventory of Authentic Human Annotations ($N=110$)

In contrast, the master reviewer files represent genuine human review:

| File Path | Rows | Columns | SHA256 Hash |
| :--- | :---: | :---: | :--- |
| `ai_engine/rag/data/master_reviewer_1.csv` | 110 | 5 | `10540d6c4293f0b9fbc3efb1c098863f6990ae93f77df5090f230daae30ba33d` |
| `ai_engine/rag/data/master_reviewer_2.csv` | 110 | 5 | `94e4ae6783863baea4bbd050c18dcf9c4bc818817df0b4df279b33e244d2d4d8` |

### Task and Cohort Composition
- **Personal Photo Albums (`L10-A1` to `A3`, $N=60$):** 60 generation responses from user photo-album queries across 3 albums. Reviewers unanimously judged all 60 responses as supported, direct, and correct (100% agreement).
- **Relational Multi-Hop Queries (`AUDIT-001` to `AUDIT-050`, $N=50$):** 50 complex multi-hop retrieval and reasoning queries across cross-modal contexts (text, imagery, and audio events). Raters independently documented detailed qualitative feedback (e.g., *"Dry pot dish present, district tag missing"*, *"Acoustic event not matched in context"*).

---

## 4. Recalculated Inter-Annotator Agreement

Evaluated on the $N=110$ authentic human records (`FINAL_FORENSIC_AUDIT/HUMAN_ANNOTATION_RECALCULATION.csv`):

| Evaluation Subset | Dimension | $N$ | Observed Agreement | Cohen's $\kappa$ | 95% Bootstrap CI | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **All 110 Pooled** | **Faithfulness** | 110 | 95.45% | **0.8716** | [0.7534, 0.9721] | Almost Perfect |
| **All 110 Pooled** | **Query Relevance** | 110 | 96.36% | **0.8926** | [0.7837, 0.9752] | Almost Perfect |
| **All 110 Pooled** | **Answer Correctness** | 110 | 98.18% | **0.9451** | [0.8568, 1.0000] | Almost Perfect |
| **All 110 Pooled** | **Composite Mean** | 110 | 96.66% | **0.9031** | [0.7980, 0.9824] | Almost Perfect |
| Relational Audit | Faithfulness | 50 | 90.00% | 0.8054 | [0.6302, 0.9596] | Substantial |
| Relational Audit | Query Relevance | 50 | 92.00% | 0.8431 | [0.6925, 0.9600] | Almost Perfect |
| Relational Audit | Answer Correctness | 50 | 96.00% | 0.9196 | [0.7920, 1.0000] | Almost Perfect |
| Relational Audit | Composite Mean | 50 | 92.67% | 0.8561 | [0.7049, 0.9732] | Almost Perfect |
| Photo Albums | All Dimensions | 60 | 100.00% | 1.0000* | [1.0000, 1.0000] | Perfect Agreement (Unvarying) |

*\*Note: In the Photo Albums cohort, raters exhibited 100% consensus across all items. When marginal distributions lack class variance, Cohen's kappa exhibits a degenerate single-class asymptote; reporting observed agreement (100.0%) alongside pooled kappa (0.9031) provides full transparency.*

---

## 5. Audit Conclusions and Manuscript Action Items

1. **Strictly Quarantine `scripts/gapfill_t2_05_e2e_rag_accuracy.py`:** Do not cite or present any synthetic rater outputs as human evaluation.
2. **Manuscript Section IV-E / IV-F Verification:** The revised manuscript (`MEMORYVERSE_FINAL_AUDIT/MANUSCRIPT_V2/ijcds_body.tex`) has properly decoupled the automated QA benchmark ($N=150$, ANLS/EM) from the authentic dual-reviewer audit ($N=110$, $\kappa = 0.9031$ across photo albums and relational retrieval).
3. **No Fabrication:** The existing 110 authentic reviews are scientifically valid and completely documented with individual rater logs. No additional human annotation is required for submission as long as this scope is precisely represented.

