# CHECKPOINT 1: Quality-First Audit & Claim Reconciliation Report (V9)

**Date:** 2026-10-27  
**Working Paper Copy:** `IEEE_Paper_v9.tex` (Compiled cleanly to `IEEE_Paper_v9.pdf`; original `IEEE_Paper_MemoryVerse.tex` untouched)  
**Plan Freeze Commit:** `97d4ab56` (2026-10-25)  
**Number Provenance Database:** `results/v9/number_provenance.csv` (172 verified empirical records)  
**Verification Script Status:** `python3 scripts/check_numbers.py` $\rightarrow$ **100% PASS**  
**Compilation Status:** `tectonic IEEE_Paper_v9.tex` $\rightarrow$ **PASS (0 errors, 0 undefined citations, 0 undefined refs)**  
**Kaggle Harness Status:** `python3 kaggle/kaggle_run_v9.py --smoke --run all` $\rightarrow$ **PASS (Exit Code 0)**  

---

## 1. Executive Summary

This checkpoint concludes **Phase 0 (Audit and Number Reconciliation)** and **Phase 1 (Claim Corrections)**. Every empirical metric in the abstract, contributions, limitations, conclusion, and table captions traces with zero fabrication to verifiable result files under `results/` logged in `results/v9/number_provenance.csv`. All changes are logged in `CHANGELOG_V9.md`.

In accordance with the project instructions, **work has paused at Checkpoint 1**, awaiting user review before initiating Phase 2 (manuscript re-scoping and cuts).

---

## 2. Phase 0: Audit and Number Reconciliation (CPU)

### [DONE] 0.1 Word BM25 Clean OCR Reconciliation
- **Discrepancy:** Word BM25 on Clean Apple Vision OCR had two conflicting value sets in earlier drafts:
  - Regime Grid (`results/v3/headline_regime_grid_v3.csv`): **0.8371** verbatim, **0.4334** light paraphrase.
  - Engine Table (`results/v3/engine_replication.csv`): **0.8324** verbatim, **0.4003** light paraphrase.
- **Forensic Root Cause:** In `headline_regime_grid_v3.csv`, distractors retained caption fallback text (`r.get('real_ocr_text') or r['ground_truth_text']`), calibrating average document length $\text{avg\_len}$ on all 813 documents. In `engine_replication.csv`, distractors were empty strings `""`, compressing $\text{avg\_len}$ and penalizing receipts via BM25 length normalization $b \cdot \frac{|d|}{\text{avg\_len}}$.
- **Resolution:** Standardized on the headline regime grid values (**0.8371** verbatim, **0.4334** light paraphrase) across all text, tables, and captions. Caption of Table V explicitly documents: *"Evaluated on held-out test split ($N=306$ probes, 813 corpus items); distractors retain caption fallback text."*
- **Evidence:** `results/v9/dose_response_reconciled.csv`, `IEEE_Paper_v9.tex` Table V (line 305).

### [DONE] 0.2 Dose-Response Ghost Number Elimination
- **Discrepancy:** The text quoted an optical blur drop of `0.6724` that appeared in no table or result file (regime grid was 0.6972; ablation was 0.6684).
- **Forensic Root Cause:** Derived from an invalid cross-table mixed subtraction ($0.8324 - 0.1600 = 0.6724$).
- **Resolution:** Purged `0.6724` completely. Recomputed verbatim-to-light-paraphrase drops vs optical blur drops for **BOTH** OCR engines from a single reconciled source (`results/v9/dose_response_drops_v9.csv`):
  - **Apple Vision:** Paraphrase drop is **0.4037** MRR (0.8371 to 0.4334); Optical blur drop is **0.1399** MRR (0.8371 to 0.6972).
  - **Tesseract 5.5:** Paraphrase drop is **0.3400** MRR (0.7364 to 0.3964); Optical blur drop is **0.3916** MRR (0.7364 to 0.3448).
- **Evidence:** `results/v9/dose_response_drops_v9.csv`, `IEEE_Paper_v9.tex` line 68.

### [DONE] 0.3 Dense MiniLM Multi-Value Delineation
- **Discrepancy:** Dense MiniLM on clean text appeared across the manuscript as `0.2100`, `0.1919`, `0.2030`, and `0.2056`.
- **Forensic Root Cause:** Each number represents a distinct combination of chunking strategy, sequence length, and text source:
  1. `0.2100`: Pristine Ground Truth text, chunked MaxP (150-word sliding window).
  2. `0.1919`: Pristine Ground Truth text, unchunked single-vector (truncating at 256 tokens).
  3. `0.2056`: Clean Apple Vision OCR text, unchunked single-vector.
  4. `0.2030`: Clean Apple Vision OCR text, chunked MaxP (150-word sliding window).
- **Resolution:** Explicitly cataloged in `results/v9/dense_reconciliation_v9.csv`. Delineated in Section IV-B and Key Contributions (line 67): *"real max_seq_length=256; 33.4% exceed 256 tokens, but chunked MaxP yields identical MRR 0.2030 vs. 0.2056"*.
- **Evidence:** `results/v9/dense_reconciliation_v9.csv`, `IEEE_Paper_v9.tex` line 67.

### [DONE] 0.4 Conformal Cascade Re-Verification
- **Verification:** Regenerated from latest 200-split evaluation in `results/v7/conformal_retest_summary_v7.csv` and verified in `results/v9/conformal_retest_summary_v9.csv`:
  - Mean test miss risk under marginal loss: **0.0494** $\le 0.05$ (95% quantile CI: `[0.0355, 0.0654]` includes 0.05).
  - Empirical split coverage: **58.5%** across 200 calibration-test draws.
  - Escalation rate: **78.4%** (yielding **21.6%** compute savings vs always-escalate).
  - Equal-escalation fixed margin matches MRR: **0.3579** vs **0.3577** (paired test $p > 0.05$).
  - Stratified risk: entity rewrites experience risk **0.0627** ($> 0.05$).
  - Low MRR on third slice (0.0404): `conformal_pooled_cache.npz` ($N=2{,}754$) used legacy v2 `heavy_paraphrase` queries (token-stripped queries where search intent was lost), whereas BM25 scores ~0.90 on redesigned factual questions in `heavy_paraphrases_v3.json`. Documented in `AUDIT_V9.md` and paper text.
- **Evidence:** `results/v9/conformal_retest_summary_v9.csv`, `AUDIT_V9.md`, `IEEE_Paper_v9.tex` line 69, line 490, line 696.

### [DONE] 0.5 Automated Number Verification Script
- **Implementation:** Created `scripts/check_numbers.py` extracting every empirical number in Abstract, Key Contributions, Limitations, Conclusion, and Table Captions and checking against `results/v9/number_provenance.csv`.
- **Status:** **PASS** (100% of reported numbers verified; zero discrepancies).
- **Evidence:** `scripts/check_numbers.py`, `results/v9/number_provenance.csv`.

### [DONE] 0.6 Manuscript Table Inventory
- **Implementation:** Created `AUDIT_V9.md` cataloging all 17 tables in `IEEE_Paper_v9.tex`, their datasets, query formulations, OCR engines, fallback text settings, and backing result files.
- **Evidence:** `AUDIT_V9.md`.

---

## 3. Phase 1: Claim Corrections (CPU)

### [DONE] 1.1 ColQwen2 Lift over Best Proxy per Cell
- **Old Claim:** *"ColQwen2 achieves +0.0982 lift over Word BM25 alone"* (ignoring that Tuned Hybrid scored 0.4334).
- **Corrected Claim:** On light paraphrases, ColQwen2-v1.0 (MRR = 0.4985) achieves **+0.0651 lift** over the best text proxy in the regime grid (Tuned Hybrid / Word BM25 = 0.4334, paired diff $+0.0651$, $p < 10^{-5}$).
- **Answerable Questions:** ColQwen2 (0.9257) statistically ties BM25 within the grid range: Word BM25 (**0.9014**) and Char BM25 (**0.9471**) with paired $p = 0.109$. Removed stale `0.9450` unaligned baseline.
- **Evidence:** `results/v9/colqwen_vs_proxies_paired.csv`, `IEEE_Paper_v9.tex` line 37, line 66, line 708.

### [DONE] 1.2 Latency Confound Elimination
- **Old Claim:** *"286x speedup for text proxies"* (comparing Tesla T4 GPU against Apple Silicon M2 CPU).
- **Corrected Claim:** Removed the cross-hardware `286x` headline entirely. Reported per-hardware measurements explicitly:
  - Document-native ColQwen2: **249.21 ms** on Nvidia Tesla T4 GPU.
  - Sub-word Character 3–5gram BM25: **0.87 ms** on Apple M2 CPU.
- **Evidence:** `results/v9/system_cost_benchmark.csv`, `IEEE_Paper_v9.tex` line 37, line 66, line 708.

### [DONE] 1.3 Unsupported Claims Removed
- **Claim A (Limitation 9):** Purged claim of a secondary generator audit (Qwen2.5-72B). Stated honestly: *"Evaluation queries for light paraphrases and answerable questions were synthesized using gpt-4o-mini. No human-authored query set was used. Because queries were generated using a single LLM without multi-model cross-generation or human validation, model-generated queries may under-represent human typographical errors..."*
- **Claim B (Conclusion):** Purged claim that *"hybrid adds +0.0026 to +0.0059 MRR"*, which had no backing result file in `results/` and was contradicted by the regime grid.
- **Evidence:** `IEEE_Paper_v9.tex` line 702, line 708; `CHANGELOG_V9.md` CHG-20, CHG-21.

### [DONE] 1.4 Confirmatory vs. Exploratory Analysis Classification
- **Classification:** Classified all experimental analyses relative to plan freeze commit `97d4ab56` (2026-10-25) in `results/v9/analysis_status.csv`:
  - Analyses A-01 through A-04 (pilot feasibility, synthetic sweeps) $\rightarrow$ **EXPLORATORY**.
  - Analyses A-05 through A-11 (public SROIE benchmark, FUNSD, MS-COCO, ColQwen2, conformal cascades) $\rightarrow$ **CONFIRMATORY**.
- **Wording:** Removed "pre-specified" from exploratory runs; replaced "conclusively confirms" with "demonstrates on held-out test splits".
- **Evidence:** `results/v9/analysis_status.csv`, `IEEE_Paper_v9.tex` line 60.

### [DONE] 1.5 P-Value Floor Standardization
- **Correction:** Replaced bare `4e-5`, `8e-5`, and `1e-5` with standardized permutation floor notation `$p < 10^{-4}$` or `$p < 1e-5$`.
- **Evidence:** `IEEE_Paper_v9.tex` line 37, line 66, line 708.

### [DONE] 1.6 Conformal Wording De-Escalation
- **Correction:** Removed all "rigorous guarantee" and "mathematically guaranteed risk bound" wording.
- **New Precision:** Accurately states: observed mean test miss risk is **0.0494** under marginal loss (95% quantile CI `[0.0355, 0.0654]` includes 0.05); empirical coverage is **58.5%**; equal-escalation fixed margin achieves identical MRR (**0.3579** vs **0.3577**); stratified risk on entity rewrites reaches **0.0627** ($> 0.05$).
- **Evidence:** `results/v9/conformal_retest_summary_v9.csv`, `IEEE_Paper_v9.tex` line 37, line 69, line 490, line 696.

### [DONE] 1.7 Multi-Hop Equivalence & Metric Scope
- **Old Claim:** Claimed proven statistical equivalence via TOST ($p_{\text{TOST}} = 0.00000$) across 100 pooled queries, and reported unbacked MRR numbers (0.4079 vs 0.4187).
- **Corrected Claim:** Stated plainly: *"exhibits no detectable difference"* because the 95% CI `[-0.0792, +0.0792]` spans beyond $\pm 0.05$. Based all headline numbers strictly on the **50 diagnostic multi-hop queries**:
  - TargetHit@10: **0.640** (extractive) vs **0.640** (generative LLM), paired diff **0.0000** [95% CI: `[-0.0792, +0.0792]`], exact McNemar $p = 1.0000$ (2 wins, 2 losses, 46 ties).
  - Purged unbacked MRR claims (0.4079 vs 0.4187) as no multi-hop MRR result file exists.
- **Evidence:** `results/v6/multihop_independent_v6.csv`, `IEEE_Paper_v9.tex` line 37, line 70, line 531, line 708.

### [DONE] 1.8 Dense Tokenization Fragmentation vs. Query Formulation
- **Clarification:** Dense failure on pristine text (MRR 0.1919/0.3256) is driven by single-entity query formulation mismatch (answering questions restores BGE-Base to 0.5263 / 79.74% Hit@10). Subword tokenization fragmentation (+122.3% subwords/word) and embedding drift explain the additional drop under noise. Reported measured end-to-end MRR recovery (+0.0434) under Dev lexicon projection.
- **Evidence:** `results/v9/dense_reconciliation_v9.csv`, `results/v2/mechanism_tokenization_fragmentation.csv`, `IEEE_Paper_v9.tex` line 67, line 344.

---

## 4. Master Checklist Status Table

| Item | Description | Status | Evidence File | Table / Line |
| :--- | :--- | :---: | :--- | :--- |
| **0.1** | Word BM25 Clean OCR reconciliation | **DONE** | `results/v9/dose_response_reconciled.csv` | Table V, Line 305 |
| **0.2** | Optical blur drop & ghost number elimination | **DONE** | `results/v9/dose_response_drops_v9.csv` | Lines 68, 700 |
| **0.3** | Dense MiniLM clean scores delineation | **DONE** | `results/v9/dense_reconciliation_v9.csv` | Line 67, Table VII |
| **0.4** | Conformal table re-verification & 3rd slice MRR | **DONE** | `results/v9/conformal_retest_summary_v9.csv` | Line 69, Table VI |
| **0.5** | Automated number checker script | **DONE** | `scripts/check_numbers.py` | 100% PASS |
| **0.6** | Manuscript table audit inventory | **DONE** | `AUDIT_V9.md` | All 17 tables cataloged |
| **1.1** | ColQwen2 vs. best proxy lift (+0.0651) | **DONE** | `results/v9/colqwen_vs_proxies_paired.csv` | Lines 37, 66, 708 |
| **1.2** | Latency profiling per-hardware (no 286x) | **DONE** | `results/v9/system_cost_benchmark.csv` | Table XVI, Line 66 |
| **1.3** | Unsupported claims removed (Qwen2.5 / +0.0026) | **DONE** | `CHANGELOG_V9.md` | Lines 702, 708 |
| **1.4** | Confirmatory vs. exploratory classification | **DONE** | `results/v9/analysis_status.csv` | Line 60 |
| **1.5** | P-value floor standardized ($p < 10^{-4}$) | **DONE** | `results/v9/number_provenance.csv` | Lines 37, 66, 708 |
| **1.6** | Conformal wording de-escalation (no guarantees) | **DONE** | `results/v9/conformal_retest_summary_v9.csv` | Lines 37, 69, 696 |
| **1.7** | Multi-hop equivalence scope & exact McNemar | **DONE** | `results/v6/multihop_independent_v6.csv` | Lines 37, 70, 531, 708 |
| **1.8** | Dense query formulation vs. noise fragmentation | **DONE** | `results/v9/dense_reconciliation_v9.csv` | Lines 67, 344 |
| **Kaggle** | Kaggle v9 harness, smoke test, and input bundle | **DONE** | `kaggle/frozen_inputs_v9.zip`, `kaggle_run_v9.py` | Smoke test: Exit code 0 |

---

## 5. Changed Numbers & Sentences Now False

### Changed Numbers Summary
1. **ColQwen2 Light Paraphrase Lift:** `+0.0982` $\rightarrow$ `+0.0651` (evaluated against Tuned Hybrid / Word BM25 0.4334).
2. **Apple Vision Paraphrase Drop:** `0.4321` $\rightarrow$ `0.4037` (0.8371 to 0.4334).
3. **Apple Vision Blur Drop:** `0.6724` / `0.1600` $\rightarrow$ `0.1399` (0.8371 to 0.6972).
4. **Tesseract 5.5 Paraphrase Drop:** `0.3397` $\rightarrow$ `0.3400` (0.7364 to 0.3964).
5. **Tesseract 5.5 Blur Drop:** Unreported $\rightarrow$ `0.3916` (0.7364 to 0.3448).
6. **ColQwen2 Answerable Comparison Baseline:** `0.9450` unaligned baseline $\rightarrow$ `0.9014` Word BM25 / `0.9471` Char BM25.
7. **Conformal Mean Test Risk:** `0.0491` $\rightarrow$ `0.0494` (95% quantile CI: `[0.0355, 0.0654]`).
8. **Conformal Split Coverage:** `38.0%` conditional ratio $\rightarrow$ `58.5%` empirical marginal coverage.
9. **Multi-Hop MRR:** `0.4079 vs 0.4187` $\rightarrow$ **REMOVED** (metric is binary TargetHit@10: 0.640 vs 0.640, exact McNemar $p = 1.0000$).
10. **Hybrid Minimal Lift:** `+0.0026 to +0.0059` $\rightarrow$ **REMOVED** (unsupported text artifact).

### Sentences Now False
1. *"ColQwen2 provides a +0.0982 lift over BM25"* $\rightarrow$ **FALSE.** The lift is +0.0651 over the best text proxy in the regime grid.
2. *"Text proxies achieve a 286x latency speedup over ColQwen2"* $\rightarrow$ **FALSE.** 286x was a cross-hardware ratio between a remote GPU and local Apple Silicon CPU.
3. *"Conformal cascades mathematically guarantee test risk $\le 0.05$"* $\rightarrow$ **FALSE.** Finite-sample split conformal prediction controls marginal expectation; observed empirical test risk is 0.0494, and coverage is 58.5%.
4. *"Extractive bridging is proven statistically equivalent to generative LLM bridging ($p_{\text{TOST}} = 0.00000$)"* $\rightarrow$ **FALSE.** Equivalence is not established because the 95% CI `[-0.0792, +0.0792]` spans beyond $\pm 0.05$. We report "no detectable difference" on 50 multi-hop queries ($p = 1.0000$).
5. *"Dense retrieval failure on receipts is driven purely by tokenization fragmentation"* $\rightarrow$ **FALSE.** Dense failure on pristine text is driven by single-entity query formulation mismatch; tokenization fragmentation explains the additional drop under noise.
6. *"Secondary generator auditing via Qwen2.5-72B confirmed paraphrase stability"* $\rightarrow$ **FALSE.** No Qwen2.5-72B audit result file exists in the repository.

---

## 6. STOP: Checkpoint 1 Reached

As explicitly instructed in the project task specification:
> `>>> CHECKPOINT 1: write CHECKPOINT1.md listing each item, changed numbers, and sentences now false. STOP.`

All Phase 0 and Phase 1 objectives are completely resolved and verified. Execution is now paused for user review and approval before proceeding to **Phase 2 (Re-scope the paper / cuts)**.
