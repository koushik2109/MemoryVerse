# HUMAN INTERVENTION REGISTER
## MemoryVerse — International Journal of Computing and Digital Systems (IJCDS)
**Workspace:** `IJCDS_RESEARCH_EXECUTION_01/`  
**Date:** November 2026  
**Auditor:** Multidisciplinary Senior Research & Engineering Panel  

---

### Operating Principle (Section 3.5)
You must stop and request human intervention whenever proceeding would require guessing, inventing, interpreting ambiguous evidence, or making a decision that should belong to the researcher.

---

### Registered Interventions

| Intervention ID | Issue & Relevant Files | Evidence & Ambiguity | Decision Required | Available Options | Recommended Action | Researcher Response | Final Resolution | Affected Manuscript Sections |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **HI-01** | **Independent Human Participant Recruitment vs. Authoritative 400 Queries**<br>Files: `data/human_queries_200.csv`, `04_PARTICIPANT_PROTOCOL.md` | Phase 4 outlines recruiting 2–5 external participants (100–150 queries). However, the author clarified in Section 2: *"The existing 400 human-authored query instances were written by the researcher and a friend. Do not describe them as programmatically generated or templated. Do not reopen this authorship issue... The final queries are the authoritative evaluation queries."* | Does the researcher want to recruit new human participants (requiring consent, task design, and timeline extension), or maintain the existing authoritative 400-query suite with transparent authorship disclosure? | **Option A:** Maintain the authoritative 400-query suite with transparent disclosure that queries were authored by the researcher and a friend, and transaction dates were used for disambiguation.<br>**Option B:** Recruit 2–5 external participants for 100–150 new queries under a formal ethics and privacy protocol. | **Option A (Recommended).** The 400 queries are already completed, verified, and provide a strong sample ($N=400$). Transparent disclosure satisfies academic integrity without artificial delays. | **Approved Option A** (2026-11-05) | **RESOLVED:** Authoritative 400 human queries maintained with transparent disclosure (written by researcher and friend; transaction dates for disambiguation). | Section IV-B, Section V-A, Table XI |
| **HI-02** | **Relevance Ground Truth Policy on Repeated Queries**<br>Files: `data/human_queries_200.csv`, `data/human_queries_adjudicated.csv` | 37 query strings are repeated across 147 instances because multiple receipts originate from the same corporate chain (Gardenia, 99 Speedmart, MR DIY). The researcher clarified that transaction dates were used to distinguish the intended receipt and revised queries accordingly. | Should the primary reporting in Table XI report Single-Target ground truth, Adjudicated Multi-Relevance, or both side-by-side as a sensitivity analysis? | **Option A:** Report Single-Target (Word 0.5900, Char 0.5952), Adjudicated Multi-Relevance (Word 0.6740, Char 0.6825), and Unique Subset (Word 0.7599, Char 0.7658) side-by-side in Table XI.<br>**Option B:** Report strict Single-Target evaluation only. | **Option A (Recommended).** Side-by-side reporting is the standard of scientific rigor; it proves that Character BM25 tracks Word BM25 under both ground-truth assumptions. | **Approved Option A** (2026-11-05) | **RESOLVED:** Single-Target, Multi-Relevance, and Unique subsets reported side-by-side in Table XI. | Section V-A, Table XI |
| **HI-03** | **Adaptive Retrieval Selector vs. Simpler Fixed Cascade**<br>Files: `scripts/run_tier_a_baselines.py`, `scripts/gapfill_t2_04_fusion_cascade.py` | Phase 8 suggests an adaptive selector. Prior experiments (Table IV, H4.5–H4.7) evaluated a learned router and found non-significant gains ($p_{\text{Holm}} = 0.1260$ to $0.8200$). Section 11 states: *"If the selector fails, report the result and retain the simpler architecture."* | Confirm whether to retain the negative router result as an honest ablation and recommend the fixed two-stage cascade ($K_1=50$), or to spend compute developing another complex ML classifier. | **Option A:** Retain the negative router result as an empirical ablation, and advocate the fixed two-stage cascade ($K_1=50$) as the primary practical architecture ($15.5\times$ speedup, 97.2% visual retention).<br>**Option B:** Train a new multi-feature logistic or decision-tree classifier on Dev splits. | **Option A (Recommended).** Fixed cascade avoids router misclassification risk and runtime latency overhead, directly fulfilling the governing principle: a narrower, fully defensible contribution. | **Approved Option A** (2026-11-05) | **RESOLVED:** Retain negative router ablation; advocate fixed two-stage cascade (K1=50). | Section IV-D, Section V-C, Table XII, Table XIII |

---

### Procedure for Next Steps
The panel will pause execution of any new human query collection (HI-01) and complex ML router training (HI-03) until the researcher reviews and answers these three targeted decisions. All other non-contingent discovery and traceability tasks continue autonomously.

