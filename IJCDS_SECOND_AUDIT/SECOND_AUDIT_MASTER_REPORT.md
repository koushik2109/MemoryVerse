# MASTER INDEPENDENT SCIENTIFIC & STATISTICAL AUDIT REPORT (STAGE 2)
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Authors:** Anurag Krishna Vennela and G. Sai Balaji Koushik (NIT Warangal, India)  
**Auditor:** Independent Senior Research Scientist, Statistical Reviewer & Reproducibility Auditor  
**Audit Completed:** November 2026  
**Audit Directory:** `IJCDS_SECOND_AUDIT/`  

---

## 1. Executive Summary & Final Verdict

### Formal Classification:
### **B. MOSTLY VERIFIED – Specific Technical Corrections Required**

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 2 INDEPENDENT AUDIT VERDICT                               |
|                                                                                                    |
|                       [ B. MOSTLY VERIFIED – SPECIFIC TECHNICAL CORRECTIONS REQUIRED ]              |
|                                                                                                    |
| 1. Empirical Grounding: The core scientific conclusions of the research—character n-gram BM25      |
|    superiority under OCR noise, subword token fragmentation driving dense retrieval collapse,      |
|    zero-shot SigLIP visual equivalence to human descriptions, ColQwen2 late-interaction superiority|
|    on 2D document layouts, and finite-sample conformal risk calibration—are verified and           |
|    computationally reproducible from original datasets and probe outputs.                         |
| 2. Required Technical Corrections: Several critical reporting defects and methodological flaws     |
|    must be corrected prior to publication:                                                         |
|    a. Holm-Bonferroni Family 4 step-down sorting bug (MiniLM is significant at p_Holm = 0.0450).   |
|    b. Holm-Bonferroni Family 6 multiplier bug (LLM bridging is non-significant at p_Holm = 0.1075).|
|    c. Conflation of human evaluations: 110-answer photo album audit misattributed as document RAG; |
|       document RAG QA rater ratings were programmatically simulated via modulo index flipping.    |
|    d. Two-stage cascade latency (16.20 ms) is an analytical model estimate based on cross-hardware  |
|       profiling (Apple M2 CPU + Tesla T4 GPU), not an end-to-end live runtime measurement.         |
|    e. Query count discrepancy: 1,930 queries executed, not 1,730.                                  |
|    f. Crossover boundary (0.51-0.55) is an overfitted model artifact; human queries show no        |
|       crossover effect (Char and Word BM25 are tied across bins).                                  |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Independent Investigation of Preliminary Audit Findings

### 2.1 Query-Count Discrepancy ($1{,}730$ vs. $1{,}930$)
- **Investigation:** We inspected the query-generation files (`results/v6/dose_response_queries_v6.json`, `data/human_queries_200.csv`, `ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json`) and the execution script (`scripts/gapfill_t3_02_generator_robustness.py`).
- **Exact Counts:**
  - `GPT-4o-mini`: $306 \text{ receipts} \times 4 \text{ bands} = 1{,}224\text{ queries}$.
  - `Human-Authored`: $200 \text{ rows in } \texttt{human\_queries\_200.csv}$. However, lines 134–162 extract **both** `handwritten_verbatim` and `handwritten_paraphrase`, yielding $200 \times 2 = 400\text{ queries}$.
  - `Qwen2.5-7B`: $306\text{ heavy paraphrase queries}$.
  - **Total Scored Queries in `generator_query_evaluations.csv`:** exactly **$1{,}930\text{ queries}$**.
- **Root Cause:** The manuscript authors counted 200 human receipts and added $1{,}224 + 306 + 200 = 1{,}730$, failing to realize that the code extracted two queries per row.
- **Uniqueness & Duplicates:** While all 1,930 records are uniquely identified by `(generator, target_id, source_band)`, 229 instances share identical query strings across different receipts (e.g., `"99 Speed Mart Klang"` repeats for multiple receipts from that vendor).
- **Verdict:** Reporting error in manuscript text and captions ($1{,}730 \to 1{,}930$).

---

### 2.2 Human Evaluation Sample Discrepancy (100 vs. 110 Cases)
- **Investigation:** We audited all human annotation sheets (`master_reviewer_1.csv`, `master_reviewer_2.csv`, `scripts/gapfill_t2_05_e2e_rag_accuracy.py`).
- **Findings:**
  1. **Authentic Human Audit ($N=110$):** In `ai_engine/rag/data/master_reviewer_1.csv` and `master_reviewer_2.csv`, two human reviewers evaluated 60 personal photo album queries (`L10-A1` to `A3`) and 50 multi-hop relational queries (`AUDIT-001` to `050`). Independent recalculation of Cohen's kappa yields:
     - Faithfulness: observed agreement $= 0.9545, \kappa = 0.8716$
     - Relevance: observed agreement $= 0.9636, \kappa = 0.8926$
     - Correctness: observed agreement $= 0.9818, \kappa = 0.9451$
     - **Composite Mean $\kappa = 0.9031$**.
     This is an authentic human audit, but it evaluated **photo album and relational retrieval**, NOT the 150 document QA queries.
  2. **Simulated "Human Audit" ($N=100$):** In `scripts/gapfill_t2_05_e2e_rag_accuracy.py` (lines 620–648), the 100 document RAG audit was generated programmatically:
     - "Reviewer 1" was an automated ANLS rule (`anls >= 0.8 -> correct`, `anls >= 0.5 -> partially_correct`).
     - "Reviewer 2" was generated by deterministically flipping labels using modulo arithmetic: `if i % 11 == 0` for correctness; `if i % 13 == 0` for faithfulness; `rel_2 = rel_1` always.
- **Verdict:** The manuscript conflated these two separate studies. The claim of dual human review on document RAG must be retracted, and the $\kappa = 0.9031$ agreement must be correctly attributed to the 110-answer photo album study.

---

### 2.3 Holm-Bonferroni Hypothesis Multiplicity Corrections
We audited all 35 confirmatory hypotheses across the 6 families registered in `results/v4/R-02/holm_family_complete_table.csv`:

#### A. Family 4 Sorting Error (RAMP Architectural Ablation, $m = 7$)
- In `holm_family_complete_table.csv`, Hypothesis 4.7 (`Char BM25 + Cross-Encoder Degradation`, $p_{\text{raw}} = 0.0001$) was placed at position 7 instead of position 1.
- This placed the MiniLM reranker ($p_{\text{raw}} = 0.0090$) at Rank 2 with critical threshold $\alpha_2 = 0.05/6 = 0.00833$, reporting $p_{\text{Holm}} = 0.054$ (declared non-significant).
- When sorted strictly by ascending $p_{\text{raw}}$:
  - Rank 1 ($p=0.00010$): $\alpha_1 = 0.05/7 = 0.00714 \implies p_{\text{Holm}} = 0.0007$ (Significant)
  - Rank 2 ($p=0.00110$): $\alpha_2 = 0.05/6 = 0.00833 \implies p_{\text{Holm}} = 0.0066$ (Significant)
  - **Rank 3 (MiniLM Reranker, $p=0.00900$):**
    $$\alpha_3 = \frac{0.05}{7 - 3 + 1} = \frac{0.05}{5} = \mathbf{0.01000}$$
    Since $p_{\text{raw}} = 0.00900 \le 0.01000$:
    $$p_{\text{Holm}} = 5 \times 0.00900 = \mathbf{0.04500} \le 0.05$$
- **Corrected Finding:** MiniLM cross-encoder degradation is **STATISTICALLY SIGNIFICANT**.

#### B. Family 6 Multiplier Error (Relational Retrieval, $m = 5$)
- Hypothesis 6.1 (LLM bridge vs Sparse BM25, $p_{\text{raw}} = 0.02150$) was multiplied by $2$ instead of $5$ ($m=5$), reporting $p_{\text{Holm}} = 0.04300$.
- With the correct step-down threshold ($\alpha_1 = 0.05/5 = 0.01000$), because $0.02150 > 0.01000$:
  $$p_{\text{Holm}} = 5 \times 0.02150 = \mathbf{0.10750} > 0.05$$
- **Corrected Finding:** Generative LLM bridging is **NOT STATISTICALLY SIGNIFICANT** under family-wise control.

---

## 3. Independent TOST Equivalence Verification

We investigated the discrepancy between the reported $p = 0.0050$ and the preliminary audit's $p \approx 0.0067$:
1. **$0.0050$ is Strictly Parametric:** In `results/v4/R-02/coco_siglip_tost_recomputed.csv`, using $\bar{d} = -0.0139, \text{SE} = 0.0140$:
   $$t_1 = \frac{-0.0139 - (-0.05)}{0.0140} = 2.5825 \implies p_1 = 0.00498 \approx \mathbf{0.0050}$$
2. **No Permutation Test Exists:** The hypothesis of a permutation TOST is refuted; no such code exists in the repository.
3. **Equivalence Holds:** Under both $\text{SE} = 0.0140$ ($p=0.0050$) and the sample SD $\text{SE} = 0.01458$ ($p=0.0067$), the null hypothesis of non-equivalence is rejected at $\alpha = 0.01$, and both 90% CIs lie entirely within $[-0.05, +0.05]$.

---

## 4. Deep RCPS & Conformal Risk Control Audit

1. **Algorithm:** Uses exact Clopper-Pearson binomial inversion on calibration loss indicators ($ub = \text{beta.ppf}(1-\delta, k+1, n-k) \le \epsilon$).
2. **Partitioning:** Cluster-stratified by document ID across 200 random splits; zero data leakage.
3. **Resolution of 82.0% Coverage:**
   The nominal $1-\delta = 0.90$ guarantee applies to the **true population risk** $R(\hat{\tau})$.
   On each test split ($N_{\text{test}} = 1{,}377$), the empirical test risk $\hat{R}_{\text{test}}$ has binomial sampling variance. Binomial sampling theory proves that even when population risk is controlled at $0.045 \le 0.05$, empirical test risk exceeds $0.05$ in approximately $15\text{--}20\%$ of splits.
   Thus, **82.0% empirical split coverage is mathematically expected and validates the algorithm**.
4. **Verdict:** Valid implementation; requires textual clarification.

---

## 5. Rebuilding Crossover Dynamics

1. **Model Overfitting:** The manuscript's 6-parameter logistic model was fit on only 12 summary points with hardcoded coordinates ($0.35, 0.65, 1.00$).
2. **Query-Level Reality ($N=1{,}930$):**
   - On GPT-4o-mini, crossover occurs at $\theta \approx 0.68\text{--}0.70$.
   - On authentic human queries ($N=400$), **no crossover exists** ($\Delta \text{MRR} \le 0.027$ across all bins; slope is $-0.0181$).
3. **Invariance Retraction:** Concluding invariance from an ANOVA $F=0.413, p=0.6617$ is invalid. Crossover dynamics are syntax- and generator-dependent.

---

## 6. Cascade Latency Profiling

In `scripts/gapfill_t2_04_fusion_cascade.py` (lines 105–125):
$$\text{Latency} = 0.87\text{ ms (Apple M2 CPU)} + 50 \times \left(\frac{249.21\text{ ms (Tesla T4 GPU)}}{813}\right) = \mathbf{16.20\text{ ms}}$$
- **Finding:** Latency was **never measured end-to-end**. It is an analytical model summing CPU time on one machine with GPU time on another, omitting PCIe and IPC overheads.
- **Reranking:** Evaluated via proxy lookup (`np.where(rank_word <= 50, rr_col, 0.0)`), not live candidate scoring.
- **Verdict:** Must be labeled as an analytical model in the manuscript.

---

## 7. Headline Claims Verification Matrix

| Claim Item | Nominal Manuscript Value | Raw Artifact Value | Recalculated Value | Independent Audit Status | Notes & Evidence |
|---|---|---|---|---|---|
| **1. Primary Benchmark** | 1,626 items, 813/813 split | 1,626 items | 1,626 items | **VERIFIED** | 0 cross-split leakage |
| **2. Real OCR BM25 Lift** | $+0.0733$ MRR ($p=0.0011$) | $+0.0733$ | $+0.0733$ | **VERIFIED** | Paired permutation test |
| **3. FUNSD Transfer** | $+0.0763$ (Tess), $+0.0768$ (Easy) | $+0.0763, +0.0768$ | $+0.0763, +0.0768$ | **VERIFIED** | Transfer confirmed |
| **4. DocVQA Visual Gain** | $+0.2472$ MRR ($p=0.0001$) | $+0.2472$ | $+0.2472$ | **VERIFIED** | Disclose pretraining overlap |
| **5. 500K Corpus Scaling** | Word 0.9412, Char 0.9385 | $0.9412, 0.9385$ | $0.9412, 0.9385$ | **VERIFIED WITH CAVEATS** | Distractors repeat 20 vendors |
| **6. SigLIP Visual TOST** | $p_{\text{TOST}} = 0.0050$ | $0.00498$ | $0.00498$ (parametric) | **VERIFIED WITH CAVEATS** | Parametric $t$-test, not permutation |
| **7. Relational Multi-Hop** | Hit@10 = $0.8200$ ($p=0.0073$) | $0.8200$ | $0.8200$ | **VERIFIED** | Statistically equivalent |
| **8. PRF vs LLM Latency** | $394\times$ speedup ($12.0$ vs $4736.8$) | $394.7\times$ | $394.7\times$ | **VERIFIED** | Measured latency ratio |
| **9. Dense Collapse Cause** | Subwords $+122.3\%$, Truncation $<0.007$ | $+122.3\%, <0.007$ | $+122.3\%, <0.007$ | **VERIFIED** | Token fragmentation verified |
| **10. ColQwen2 Standalone** | $0.4962$ MRR | $0.4962$ | $0.4962$ | **VERIFIED** | Test corpus baseline |
| **11. Cascade Retention** | $97.2\%$ MRR retention | $97.18\%$ | $97.18\%$ | **VERIFIED** | Evaluated via proxy lookup |
| **12. Cascade Latency** | $16.20\text{ ms}$ ($15.4\times$ speedup) | $16.20\text{ ms}$ | Modeled estimate | **UNVERIFIED (MODELED)** | CPU+GPU analytical sum |
| **13. RAG Human Audit** | $N=110, \kappa = 0.9031$ | $N=110$ (Albums) | $N=110, \kappa = 0.9031$ | **MISATTRIBUTED SAMPLE** | 110 were photo albums; RAG was simulated |
| **14. RCPS Conformal** | Mean risk $=0.0429, 82.0\%$ coverage | $0.0429, 82.0\%$ | $0.0429, 82.0\%$ | **VERIFIED WITH CAVEATS** | $82\%$ explained by $N_{\text{test}}$ variance |
| **15. Overlap Crossover** | $[0.51, 0.55]$, invariant ($p=0.6617$) | Overfitted surface | Crossover absent on human | **UNVERIFIED (MODEL ARTIFACT)**| Retract universal invariance |

---

## 8. Summary of Second-Stage Deliverables

All audit artifacts and data tables are archived in [`IJCDS_SECOND_AUDIT/`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/IJCDS_SECOND_AUDIT):
1. `SECOND_AUDIT_MASTER_REPORT.md`: This comprehensive master audit report.
2. `RAW_METRIC_RECALCULATION.csv`: 62 metrics verified against raw artifacts.
3. `HOLM_RECALCULATION.csv`: Complete 35-hypothesis registry with corrected step-down ranks.
4. `TOST_VERIFICATION.md`: Independent parametric TOST mathematical audit.
5. `RCPS_SPLIT_LEVEL_AUDIT.csv`: Per-split calibration logs for all 200 splits.
6. `RCPS_THEORETICAL_AUDIT.md`: Theoretical assessment and binomial sampling variance proof.
7. `CROSSOVER_REANALYSIS.md`: Continuous query-level crossover evaluation on 1,930 queries.
8. `HUMAN_EVALUATION_AUDIT.csv`: Itemized review of authentic vs simulated human audits.
9. `CASCADE_LATENCY_AUDIT.md`: Analysis of the analytical cascade latency model.
10. `CLAIM_TRACEABILITY.csv`: Full claim-to-code traceability matrix.
11. `MANUSCRIPT_REPLACEMENTS.md`: Complete copy-pasteable LaTeX replacement text.
12. `UNVERIFIED_CLAIMS.md`: Itemized ledger of all unverified and modeled claims.
13. `FINAL_SUBMISSION_VERDICT.md`: Formal verdict (`B. MOSTLY VERIFIED`) and top 5 peer-review risks.

---
**Auditor Signature:** Independent Senior Research Scientist & Statistical Reviewer  
**Audit Completed:** November 4, 2026

