# LEDGER OF UNVERIFIED & METHODOLOGICALLY DEFECTIVE CLAIMS
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Auditor:** Independent Second-Stage Senior Scientific Reviewer & Reproducibility Auditor  
**Date:** November 2026  
**Audit Directory:** `IJCDS_SECOND_AUDIT/`  

---

## 1. Overview of Unverified Claims

In accordance with Absolute Rule #9, when an empirical result is derived from analytical models, proxy lookups, synthetic heuristics, or overfitted approximations rather than direct experimental measurement, it is formally classified as **UNVERIFIED**.

This document lists all 7 claims that cannot be established from existing empirical evidence in the repository, along with the precise missing artifacts required to establish them.

---

## 2. Exhaustive Itemized Audit of Unverified Claims

### 1. Two-Stage Cascade Measured Latency (16.20 ms)
- **Manuscript Location:** Abstract (line 18), Table IX (page 10), Section 5.F (line 648).
- **Exact Claim:** Two-stage cascade achieves a mean query latency of $16.20\text{ ms}$, reducing latency by $15.4\times$ (from $249.21\text{ ms}$).
- **Audit Investigation:**
  - Source file: `scripts/gapfill_t2_04_fusion_cascade.py` (lines 105–115).
  - Formula: $\text{Latency} = 0.87 + 50 \times (249.21 / 813) = 16.20\text{ ms}$.
  - The latency was mathematically calculated by combining Apple Silicon M2 CPU sparse retrieval time with a scaled fraction of NVIDIA Tesla T4 GPU time.
- **Why Unverified:** The cascade was never executed end-to-end with an active timer. Inter-process communication, PCIe memory transfer, CUDA kernel launch overhead, and image loading were omitted.
- **Missing Artifact:** A dedicated Python benchmark executing `time.perf_counter()` or `torch.cuda.Event` measuring end-to-end latency on a unified host with GPU acceleration.
- **Required Action:** Reclassify as an analytical estimate in manuscript text.

---

### 2. Live ColQwen2 Cascade Candidate Reranking
- **Manuscript Location:** Section 5.F, Table IX.
- **Exact Claim:** Two-stage cascade reranks top-50 candidates using ColQwen2 late interaction.
- **Audit Investigation:**
  - Source file: `scripts/gapfill_t2_04_fusion_cascade.py`, line 120:
    ```python
    casc_word_rr = np.where(in_k_word, rr_col, 0.0)
    ```
- **Why Unverified:** The code never passed the top-50 candidates to ColQwen2 for scoring. It simply checked if the ground-truth document was in the Word BM25 top-50 list, and if so, assigned the document its standalone full-corpus ColQwen2 reciprocal rank.
- **Missing Artifact:** A live reranking script that extracts the top-50 candidate image embeddings and computes MaxSim late interaction exclusively across those 50 candidates.
- **Required Action:** Disclose that cascade ranking was evaluated via proxy ranking simulation.

---

### 3. Universal Lexical Overlap Crossover at [0.51, 0.55]
- **Manuscript Location:** Abstract (line 27), Section 5.B (line 498), Figure 3.
- **Exact Claim:** The crossover boundary where Word BM25 surpasses Character BM25 occurs at lexical overlap $\theta \in [0.51, 0.55]$.
- **Audit Investigation:**
  - Source file: `scripts/gapfill_t2_01_regime_map.py` (lines 85–96).
  - The 6-parameter logistic model was fit on only **12 aggregated summary points** using 3 hardcoded coordinates ($0.35, 0.65, 1.00$).
  - When recomputed query-by-query on 1,930 instances (`generator_query_evaluations.csv`), the crossover occurs at $\theta \approx 0.68\text{--}0.70$ for GPT-4o-mini, and is completely absent on authentic human queries.
- **Why Unverified:** The $[0.51, 0.55]$ boundary is a mathematical artifact of an overfitted 6-parameter model on 12 points.
- **Missing Artifact:** An empirical query-level continuous model that cross-validates the crossover point without hardcoded bin coordinates.
- **Required Action:** Remove the claim of a universal $[0.51, 0.55]$ threshold; describe the boundary as generator- and syntax-dependent.

---

### 4. Cross-Generator Crossover Invariance
- **Manuscript Location:** Section 5.B (line 512).
- **Exact Claim:** One-way ANOVA across generator slopes confirms that the crossover location is statistically invariant across generators ($F=0.413, p=0.6617$).
- **Audit Investigation:**
  - ANOVA tests whether differences exist, not whether equivalence holds ($p=0.6617$ is failure to reject $H_0$).
  - On human queries ($N=400$), the slope of $(\text{Char} - \text{Word})$ is $-0.0181$ (flat), whereas for GPT-4o-mini it is $+0.3589$ (steep).
- **Why Unverified:** Conflating failure to reject difference with proof of invariance is a methodological defect. No equivalence test was performed.
- **Missing Artifact:** A Two One-Sided Tests (TOST) equivalence procedure across generator interaction slopes with a pre-specified equivalence margin.
- **Required Action:** Retract the claim of universal generator invariance.

---

### 5. Dual-Human Review on Document RAG QA (150 Queries)
- **Manuscript Location:** Section 1 (line 112), Section 6.2 (line 720).
- **Exact Claim:** Dual-reviewer human auditing was conducted on document RAG answers, achieving mean Cohen's $\kappa = 0.9031$.
- **Audit Investigation:**
  - In `scripts/gapfill_t2_05_e2e_rag_accuracy.py` (lines 620–648), "Reviewer 1" was an automated ANLS heuristic (`anls >= 0.8`), and "Reviewer 2" was generated by deterministically flipping labels via modulo arithmetic (`if i % 11 == 0`, `if i % 13 == 0`).
  - The actual human review with $\kappa = 0.9031$ was conducted on an entirely different 110-answer suite over **personal photo albums and multi-hop reasoning** (`master_reviewer_1.csv` and `master_reviewer_2.csv`).
- **Why Unverified:** No human raters ever evaluated the 150 document QA answers. Presenting a deterministic code loop as a "dual-rater human audit" is scientifically invalid.
- **Missing Artifact:** Independent, blinded human annotation sheets for the 150 document QA queries.
- **Required Action:** Explicitly re-attribute the 110-answer human audit to the photo album evaluation and remove claims of human review on the 150 document QA queries.

---

### 6. Permutation-Based TOST for MS-COCO SigLIP ($p = 0.0050$)
- **Manuscript Location:** Section 5.C, Table VI.
- **Exact Claim:** Zero-shot SigLIP visual retrieval equivalence was confirmed via a permutation-based TOST procedure yielding $p = 0.0050$.
- **Audit Investigation:**
  - The value $0.00498 \approx 0.0050$ was computed parametrically in `coco_siglip_tost_recomputed.csv` using $t_1 = 2.5825$.
  - Zero permutation scripts for TOST exist in the repository.
- **Why Unverified:** The reported $p$-value was generated by a standard Student's $t$ distribution, not a permutation distribution.
- **Missing Artifact:** A permutation script resampling paired signs under composite TOST null hypotheses.
- **Required Action:** Correct description to "parametric Student's $t$ TOST".

---

### 7. Diverse Open-Domain 500,000-Document Scaling Corpus
- **Manuscript Location:** Section 5.E, Table XII.
- **Exact Claim:** Retrieval scalability was evaluated over a diverse corpus scaling up to 500,000 documents.
- **Audit Investigation:**
  - In `scripts/gapfill_t1_04_corpus_scaling.py`, the 450,000 synthetic distractors were generated by randomly sampling combinations from only **20 retail company names**.
  - These 20 names include the exact merchant strings of the evaluation queries (`"99 SPEEDMART"`, `"MR D.I.Y."`).
- **Why Unverified:** The corpus contains tens of thousands of duplicate merchant names that act as unlabelled false-negative distractors.
- **Missing Artifact:** A deduplicated, web-scale, multi-domain distractor corpus (e.g., Common Crawl or Wikipedia).
- **Required Action:** Disclose synthetic distractor vocabulary repetition in Section 7 (Limitations).

---
**Auditor Signature:** Independent Senior Scientific Reviewer  
**Date:** November 4, 2026

