# HARD-CODED STATISTIC FINDINGS AND RESOLUTION DOSSIER

**Evaluation Scope:** MemoryVerse Recomputation Pipeline & Manuscript Statistics  
**Target Venue:** International Journal of Computing and Digital Systems (IJCDS) 2026  
**Auditor Classification:** Forensic Statistical Auditor & Reproducibility Specialist  
**Audit Status:** FULLY RESOLVED — ZERO HARD-CODED METRICS REMAINING  

---

## 1. Executive Summary

During prior independent audits of the MemoryVerse evidence packages, several critical issues of scientific integrity were identified:
1. Hard-coded summary statistics, p-values, and mean differences in `scripts/recompute_all_results_from_raw.py` (specifically lines 675–681 for RAMP Ablation H1/H2 and lines 741–744 for FUNSD BGE).
2. Synthetic projections (such as Florence-2 and BLIP-2) grouped with measured baselines without explicit demarcation.
3. Inverted bootstrap confidence intervals (e.g. DocVQA SigLIP where the point estimate was reported outside the confidence interval).

This dossier documents the root cause, exact forensic findings, and surgical resolution implemented in `MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION`.

---

## 2. Forensic Analysis of Historical Hard-Coded Statistics

### Finding A: Family 4 (RAMP Ablation Comparisons)
- **Historical Location:** `scripts/recompute_all_results_from_raw.py` (lines 675–681).
- **Code Inspection:**
  ```python
  f4_tests = [
      {"Comparison": "Char BM25 + Cross-Encoder Degradation", "Mean_Diff": -0.0262, "Std_Error": 0.0068, "Cohen_dz": -0.221, "p_raw": 0.0001},
      {"Comparison": "BGE Reranker Lift over Word BM25", "Mean_Diff": 0.0475, "Std_Error": 0.0145, "Cohen_dz": 0.188, "p_raw": 0.0011},
      {"Comparison": "MiniLM Reranker Lift over Word BM25", "Mean_Diff": 0.0261, "Std_Error": 0.0100, "Cohen_dz": 0.149, "p_raw": 0.0090},
      {"Comparison": "H1 Reliability Weighting on Real Degraded", "Mean_Diff": -0.0182, "Std_Error": 0.0078, "Cohen_dz": -0.133, "p_raw": 0.0189},
      {"Comparison": "H2 Router on 20% Syn Noise", "Mean_Diff": 0.0071, "Std_Error": 0.0035, "Cohen_dz": 0.116, "p_raw": 0.0420},
      {"Comparison": "H2 Router on Real Degraded Blur", "Mean_Diff": 0.0000, "Std_Error": 0.0010, "Cohen_dz": 0.000, "p_raw": 0.7022},
      {"Comparison": "H2 Router on Apple Clean OCR", "Mean_Diff": 0.0000, "Std_Error": 0.0010, "Cohen_dz": 0.000, "p_raw": 1.0000}
  ]
  ```
- **Audit Diagnosis:** The script authors manually inserted aggregate differences and p-values because the original experiment runner (`run_ablation_v2.py`) saved only summary averages in `ablation_results.csv`, rather than serializing query-level arrays to disk.
- **Resolution:**
  1. For verifiable cascade depth comparisons ($K_1=20$), metrics were recomputed from the actual per-query array in `raw_data/per_query_g05_evaluations.csv`.
  2. For the unverified H1 (Reliability Weighting) and H2 (Learned Router) comparisons lacking raw query-level files, **zero fabricated data was permitted**. In accordance with audit standards, they were explicitly cataloged as `NOT_REPRODUCIBLE` with notes indicating missing raw query files.

### Finding B: Family 5 (FUNSD Dense BGE Comparison)
- **Historical Location:** `scripts/recompute_all_results_from_raw.py` (lines 741–744).
- **Code Inspection:**
  ```python
  f5_tests = [
      {"Comparison": "Char vs Word BM25 on EasyOCR Paraphrase", "Mean_Diff": 0.0768, "Std_Error": 0.0248, "Cohen_dz": 0.310, "p_raw": 0.0019},
      {"Comparison": "Char vs Word BM25 on Apple Paraphrase", "Mean_Diff": 0.0630, "Std_Error": 0.0251, "Cohen_dz": 0.251, "p_raw": 0.0120},
      {"Comparison": "Dense BGE vs Word on EasyOCR Paraphrase", "Mean_Diff": 0.0540, "Std_Error": 0.0229, "Cohen_dz": 0.236, "p_raw": 0.0180},
      {"Comparison": "Char vs Word BM25 on Apple Clean Verbatim", "Mean_Diff": 0.0110, "Std_Error": 0.0125, "Cohen_dz": 0.088, "p_raw": 0.3800}
  ]
  ```
- **Audit Diagnosis:** While the BM25 queries and OCR corpora are fully present in `funsd_probes.json`, `funsd_apple_vision_ocr.json`, and `funsd_easyocr_clean_ocr.json`, the dense embedding vectors for BGE on FUNSD forms were not staged in the repository. The previous script pasted published differences directly into the test list.
- **Resolution:**
  1. BM25 Word and Character n-gram queries on EasyOCR and Apple Vision were completely recomputed from raw text files.
  2. The Dense BGE comparison was flagged as `NOT_REPRODUCIBLE` due to absence of raw query embeddings.

---

## 3. Systematic Verification and Static Code Audit

In `MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION/scripts/recompute_verified_results.py`:
- All 6 families compute metrics by dynamically calculating reciprocal ranks from raw score matrices or reading verified per-query rank logs.
- Automated quality gate `GATE_1` scans AST nodes and regular expressions:
  - Hard-coded `Mean_Diff` literals: **0 found**
  - Hard-coded `p_raw` literals: **0 found**
  - Legacy mock arrays: **0 found**

---

## 4. Final Compliance Status

| Audit Check | Status | Verification Evidence |
| :--- | :---: | :--- |
| Zero hard-coded summary statistics | **PASSED** | Verified by AST scan in `automated_quality_gates.py` |
| No inferred raw data from aggregate tables | **PASSED** | Only raw JSON/CSV inputs in `raw_data/` parsed |
| Transparent tagging of unverified claims | **PASSED** | Marked `NOT_REPRODUCIBLE` in `MASTER_PAIRED_STATISTICS.csv` |
| Segregation of exploratory projections | **PASSED** | BLIP-2/Florence-2 tagged `HISTORICAL_EXPLORATORY_PROJECTION` |

**Conclusion:** The recomputation pipeline has achieved complete forensic purity with zero hard-coded statistics.

