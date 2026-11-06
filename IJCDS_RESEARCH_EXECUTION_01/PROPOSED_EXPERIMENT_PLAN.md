# PROPOSED EXPERIMENT PLAN & DUPLICATION AUDIT
## MemoryVerse — International Journal of Computing and Digital Systems (IJCDS)
**Workspace:** `IJCDS_RESEARCH_EXECUTION_01/`  
**Date:** November 2026  
**Auditor:** Senior Information Retrieval Researcher & Experimental Reproducibility Panel  

---

### Operating Principles (Section 3.1 & 3.2)
1. **Audit Before Experimenting:** Before running any experiment, determine whether it has already been completed.
2. **No Redundant Experiments:** Do not repeat experiments merely to generate activity. A new experiment is justified only if it tests an unsupported claim, resolves a contradiction, or addresses a credible reviewer objection.

---

## 1. Inventory of Completed Experiments (No Rerun Needed)

The complete project discovery confirms that 14 core experimental suites have already been executed, independently recalculated, verified from raw files, and typeset into the manuscript. In accordance with Section 3.2, **no reruns are justified for these suites**:

| Suite ID | Empirical Experiment | Primary Findings & Metrics | Verification & Reproducibility Status | Action & Recommendation |
| :---: | :--- | :--- | :---: | :--- |
| **EXP-01** | SROIE Optical Character Degradation Matrix | Char 3–5g BM25 dominates Word BM25 across Clean (0.9436 vs 0.8371), Blur (0.8248 vs 0.6972), Tesseract (0.8412 vs 0.7364), and EasyOCR (0.8519 vs 0.7649), all $p < 10^{-4}$. | **Category A (Fully Verified)** | **DO NOT RERUN.** Existing raw evidence is complete and reproducible. |
| **EXP-02** | Continuous Overlap Dose-Response & Crossover | Reversal ($o^* \approx 0.51-0.55$) is driven by synthetic `gpt-4o-mini` entity substitution; does NOT occur on human queries where proper nouns persist. | **Category A (Fully Verified)** | **DO NOT RERUN.** 1,930 queries analyzed; logistic regression verified ($R^2=0.962$). |
| **EXP-03** | 400 Human-Authored Query Evaluation | Single-target: Word 0.5900, Char 0.5952; Multi-relevance: Word 0.6740, Char 0.6825; Unique subset: Word 0.7599, Char 0.7658. | **Category A (Fully Verified)** | **DO NOT RERUN.** Author clarified provenance (researcher + friend, transaction dates for disambiguation). |
| **EXP-04** | FUNSD Scanned Administrative Forms Transfer | Char BM25 dominates Word BM25 under paraphrase (0.2228 vs 0.1573, $p_{\text{Holm}}=0.0480$). | **Category A (Fully Verified)** | **DO NOT RERUN.** Raw form annotations and evaluation logs preserved. |
| **EXP-05** | DocVQA Complex Visual Layout Retrieval | ColQwen2 achieves 0.6450 MRR vs text proxies $\le 0.3507$. In-domain ViDoRe overlap scoped transparently. | **Category B (Verified & Scoped)** | **DO NOT RERUN.** Raw rankings verified; Kaggle GPU notebook available for independent execution. |
| **EXP-06** | MS-COCO Photographic Scene Text-to-Image | SigLIP (0.6071) decisively beats captions (BLIP 0.3728, Florence-2 0.4420, $p < 10^{-4}$); matches 4-ref gold ($p_{\text{TOST}}=0.0050$). | **Category A (Fully Verified)** | **DO NOT RERUN.** Full 1,000-image evaluation preserved and verified. |
| **EXP-07** | Cross-Encoder Reranking & MaxP Chunking | BGE Reranker lift $+0.0475$ ($p_{\text{Holm}}=0.0066$); MiniLM lift $+0.0279$ ($p_{\text{Holm}}=0.0450$). | **Category A (Fully Verified)** | **DO NOT RERUN.** All 306 probes evaluated with verified Holm multipliers. |
| **EXP-08** | Conformal Miss Risk Control (RCPS 200 Splits) | Expected risk $\bar{R}_{\text{test}} = 0.0429 \le 0.05$; 82.0% split satisfaction mathematically derived ($\sigma_{\text{total}} = 0.00756$). | **Category A (Fully Verified)** | **DO NOT RERUN.** All 200 split-level thresholds and test risks recalculated. |
| **EXP-09** | 600× Corpus Scaling to 500,000 Documents | Char BM25 maintains verbatim retrieval advantage at 500K pool (0.8945 vs 0.7254, diff $+0.1691, z=8.12, p < 10^{-15}$). | **Category A (Fully Verified)** | **DO NOT RERUN.** Full scaling logs and slope estimates verified. |
| **EXP-10** | Relational Multi-Hop Reasoning & Extractive PRF | Extractive PRF matches Generative LLM bridging (Hit@10 0.820 vs 0.820, $p_{\text{TOST}}=0.0073$) at $394\times$ lower latency. | **Category A (Fully Verified)** | **DO NOT RERUN.** 100 multi-hop queries verified. |
| **EXP-11** | Dual-Reviewer Multimodal RAG Audit ($N=110$) | Photo-album: 100% agreement ($\kappa$ undefined); Relational: mean $\kappa = 0.8560$; Pooled: mean $\kappa = 0.9031$. | **Category A (Fully Verified)** | **DO NOT RERUN.** Raw reviewer CSVs verified; rater identities disclosed. |
| **EXP-12** | Automated Document-RAG QA Evaluation ($N=150$) | ANLS 0.842, Exact Match 0.720. Purely programmatic evaluation. | **Category A (Fully Verified)** | **DO NOT RERUN.** Programmatic metrics verified; simulated code excised. |
| **EXP-13** | Two-Stage Retrieval Cascade ($K_1=50$) | Retains 97.2% visual MRR (0.4822 vs 0.4962). Modeled: 16.20 ms; Measured M2 CPU: 12.18 ms ($15.5\times$ speedup); MPS: 1.26 ms. | **Category A (Fully Verified)** | **DO NOT RERUN.** Physical live benchmarks completed and corroborated. |
| **EXP-14** | Standalone Systems & Index Memory Profiles | BM25 CPU 0.87 ms (4.2 MB RAM); ColQwen2 GPU 249.21 ms (48.0 MB index, 184 s indexing). | **Category A (Fully Verified)** | **DO NOT RERUN.** Hardware profiles verified and annotated on Table XII. |

---

## 2. Evaluation of Proposed New Experiments

### EXP-15: Adaptive ML Retrieval Selector / Router (Phase 8)
- **Proposed Objective:** Train a machine learning classifier (logistic regression or decision tree) to dynamically route queries between BM25, dense bi-encoder, and ColQwen2 based on query length, OCR confidence, and character overlap.
- **Audit Findings on Existing Evidence:**
  1. An empirical learned router was already evaluated on SROIE development and test splits (Table IV, Hypotheses H4.5, H4.6, H4.7).
  2. The empirical gains over a static baseline were **statistically non-significant**:
     - On 20% synthetic noise: diff $+0.0149$, unadjusted $p=0.0420$, $p_{\text{Holm}} = \mathbf{0.1260} > 0.05$.
     - On optical blur: diff $+0.0038$, unadjusted $p=0.4100$, $p_{\text{Holm}} = \mathbf{0.8200} > 0.05$.
     - On clean OCR: diff $+0.0014$, unadjusted $p=0.8200$, $p_{\text{Holm}} = \mathbf{0.8200} > 0.05$.
  3. Section 11 of the Master Execution Protocol explicitly instructs: *"If the selector fails, report the result and retain the simpler architecture."*
- **Panel Recommendation:** **DO NOT TRAIN A NEW COMPLEX SELECTOR.**  
  Retain the negative router finding as an honest, published architectural ablation. Recommend the fixed two-stage cascade ($K_1=50$), which is validated, robust, and achieves an empirical $15.5\times$ speedup without router misclassification latency.

---

### EXP-16: New Independent External Human Participant Study (Phase 4 / Phase 7)
- **Proposed Objective:** Recruit 2–5 external human participants to author 100–150 new receipt retrieval queries to test whether findings generalize beyond the researcher and their friend.
- **Audit Findings & Context:**
  1. The researcher explicitly clarified in Section 2:
     *"The existing 400 human-authored query instances were written by the researcher and a friend. Do not describe them as programmatically generated or templated. Do not reopen this authorship issue... The final queries are the authoritative evaluation queries. When multiple receipts originate from the same store, the researcher used transaction dates to distinguish the intended receipt and revised queries accordingly."*
  2. The existing 400-query benchmark is substantially larger ($N=400$) than the proposed 100–150 query pilot and already contains complete, verified rankings and adjudicated sensitivity partitions.
  3. External recruitment involves formal ethics, consent, and participant privacy compliance (Section 7.4), extending project scope without guaranteeing superior task compliance.
- **Panel Recommendation:** **RECOMMEND OPTION A (RETAIN AUTHORITATIVE 400 QUERIES WITH TRANSPARENT DISCLOSURE).**  
  Registered as **HI-01** in the Human Intervention Register. We recommend preserving the authoritative 400 queries with transparent manuscript disclosure of authorship and transaction-date disambiguation. If the researcher explicitly approves Option B, a 100-query participant protocol is drafted and ready in `04_PARTICIPANT_PROTOCOL.md`.

---

## 3. Summary of Experiment Execution Plan

```
====================================================================================================
PROPOSED EXPERIMENT EXECUTION STATUS
====================================================================================================
Suite ID     Topic / Objective                   Status Classification        Recommended Action
----------------------------------------------------------------------------------------------------
EXP-01–14    Core Regime & Systems Suites (14)   A / B (Completed & Verified) DO NOT RERUN (Reused)
EXP-15       Adaptive ML Selector Routing        A (Completed Negative Result) RETAIN NEGATIVE ABLATION
EXP-16       New External Human Participant Study E (Pending Researcher Call)  RECOMMEND OPTION A (PAUSE)
====================================================================================================
```
**Conclusion:** All 14 experimental foundations of the manuscript are complete, verified, and backed by authentic raw records. No redundant experiments will be executed.

