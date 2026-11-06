# EXPERIMENTAL EXECUTION & VERIFICATION LOG
**Manuscript:** *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*  
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Authors:** Anurag Krishna Vennela and G. Sai Balaji Koushik (NIT Warangal)  
**Date:** November 2026  

---

## 1. Audit Environment & System Profile

All audit inspections, dataset integrity scripts, metric recomputations, and statistical verification runs were executed in the repository's local virtual environment:

- **Host Operating System:** macOS Darwin (Apple Silicon M2 / ARM64)
- **Primary Python Runtime:** `.venv/bin/python` (Python 3.14.7 / CPython)
- **Core ML & Math Libraries:**
  - `torch`: 2.14.1
  - `transformers`: 5.15.1
  - `sentence-transformers`: 6.0.0
  - `rank-bm25`: 0.2.2
  - `scipy`: 1.18.1
  - `scikit-learn`: 1.9.0
  - `pandas`: 3.1.0
  - `numpy`: 2.5.2
  - `jiwer`: 4.0.0
  - `easyocr`: 1.7.2
  - `pillow`: 11.3.0
- **Document Compilers:**
  - `pdflatex`: TeX Live 2024 / MacTeX
  - `bibtex`: 0.99d
- **Security Check:** Verified zero active internet downloads, zero model downloads, zero API credential leaks, zero modifications to existing project files.

---

## 2. Experimental Suites & Rerun Policies

| Experiment ID | Experiment Name | Audit Verification Method | Rerun Status | Reason if Not Rerun |
|---|---|---|---|---|
| **EXP-01** | Headline Regime Grid (Table I) | Recomputed metrics from test probes & raw predictions | Re-evaluated | Cached raw probe predictions present in `results/v3/` |
| **EXP-02** | Recall@K Accuracy (Table II) | Direct recomputation from `table_ii_recall_at_k.csv` | Re-evaluated | Full probe-level rank lists available |
| **EXP-03** | Dense Chunking Sensitivity (Table III) | Statistical paired test verification | Re-evaluated | Paired scores verified from `table_iii_chunked_unchunked.csv` |
| **EXP-04** | Formulation Sensitivity (Table IV) | Direct verification against probe formulations | Re-evaluated | Query-level scores in `table_iv_corrected.csv` |
| **EXP-05** | SigLIP vs Gold Equivalence (Table VI) | TOST paired test recalculation from raw ranks | Fully Rerun | Recomputed TOST $p$-value and 90% CI on 1,000 queries |
| **EXP-06** | FUNSD Transferability (Table VII) | Metric verification on OCR outputs | Verified | Results verified against `funsd_easyocr_rerun.csv` |
| **EXP-07** | DocVQA Subsample (Table VIII) | Recalculation of MRR and paired delta | Verified | Query-level predictions in `docvqa_metrics_reconciled.csv` |
| **EXP-08** | ColQwen2 Cascade & Native (Table IX) | Latency model code audit & proxy rank check | Audited | ColQwen2 requires GPU (Tesla T4); latency is analytical |
| **EXP-09** | Relational Multi-Hop (Table X) | Metric recomputation & latency check | Verified | LLM bridging requires OpenAI API key; outputs cached |
| **EXP-10** | Query Overlap Dose-Response (Table XI) | Dataset recount & ANOVA/slope check | Re-evaluated | 1,930 queries analyzed in `generator_query_evaluations.csv` |
| **EXP-11** | Corpus Scaling to 500K (Table XII) | Generator source code audit | Audited | Synthetic distractor generator analyzed in `gapfill_t1_04` |
| **EXP-12** | End-to-End RAG (Table XIII) | Agreement file audit & metric check | Audited | Disentangled 150 QA queries from 110 album queries |
| **EXP-13** | Conformal Calibration (Section 6) | Risk distribution analysis across 200 splits | Fully Rerun | Rerun split verification across 200 calibration splits |

---

## 3. Detailed Command Log & Intermediate Scripts

### Phase 1: Environment & Project Inventory Discovery
```bash
# 1. Inspect directory structure and key scripts
find scripts ai_engine results submission_package_ijcds -maxdepth 3 -type d

# 2. Verify git status to confirm pristine working tree
git status

# 3. Check Python environment dependencies
.venv/bin/python -c "import torch, transformers, sentence_transformers, rank_bm25, scipy, sklearn; print('All core packages loaded successfully')"
```

### Phase 2: Dataset Integrity & Leakage Verification
Script created in scratch directory:
`scratch/audit_datasets.py`
```bash
# Run dataset audit script
.venv/bin/python scratch/audit_datasets.py
```
**Output Log:**
```text
Loaded benchmark with 1626 records.
Dev corpus count: 813 (SROIE: 313, COCO: 500)
Test corpus count: 813 (SROIE: 313, COCO: 500)
Overlap between Dev and Test corpus: set() (Disjoint: True)
Dev probes count: 306
Test probes count: 306
Dev probes leak into test corpus: 0
Test probes leak into dev corpus: 0
Excluded ambiguous receipts: 14 receipts confirmed
```

### Phase 3: Metric Recomputations & CSV Generation
Script created in scratch directory:
`scratch/generate_csvs.py`
```bash
# Generate structured verification CSVs
.venv/bin/python scratch/generate_csvs.py
```
**Output Log:**
```text
Wrote METRIC_VERIFICATION.csv: 81 rows
Wrote REPRODUCIBILITY_MATRIX.csv: 13 rows
Wrote CLAIM_TRACEABILITY.csv: 17 rows
```

### Phase 4: Statistical Testing & Holm Family Verification
Inspection of statistical code and CSV tables in `results/v4/R-02/holm_family_complete_table.csv`:
```python
import pandas as pd
df = pd.read_csv("results/v4/R-02/holm_family_complete_table.csv")
f4 = df[df["family"] == 4]
print(f4[["hypothesis", "p_raw", "p_holm", "rank", "significant"]])
```
**Audit Output:**
- Raw p-values: `[0.0001, 0.0090, 0.0150, 0.0230, 0.0410, 0.1200, 0.0001]`
- Hypothesis 4.7 (`Char BM25 + Cross-Encoder Degradation`, $p=0.0001$) was incorrectly positioned after hypothesis 4.6 ($p=0.1200$).
- Correct step-down ranking places MiniLM at Rank 3, giving $\alpha_3 = 0.05 / 5 = 0.0100 \implies p_{\text{Holm}} = 0.0450$ (Significant).

### Phase 5: Two One-Sided Tests (TOST) Re-execution
Verification of MS-COCO SigLIP vs 4-Reference Gold:
```python
import numpy as np
from scipy import stats

siglip = df_coco["rr_siglip"].values
gold = df_coco["rr_gold_bm25"].values
diff = siglip - gold
delta = 0.05
se = stats.sem(diff)
t1 = (diff.mean() - (-delta)) / se
t2 = (diff.mean() - delta) / se
p1 = 1 - stats.t.cdf(t1, df=len(diff)-1)
p2 = stats.t.cdf(t2, df=len(diff)-1)
p_tost = max(p1, p2)
# Result: diff.mean() = -0.0139, 90% CI = [-0.0379, 0.0101], p_tost = 0.0067
```

### Phase 6: LaTeX Manuscript Compilation & Verification
Verification that active source compiles cleanly to exactly 15 pages:
```bash
cd submission_package_ijcds/source
pdflatex -interaction=batchmode main.tex
bibtex main
pdflatex -interaction=batchmode main.tex
pdflatex -interaction=batchmode main.tex
pdfinfo main.pdf | grep Pages
# Output: Pages: 15
```

---

## 4. Random Seeds & Reproducibility Parameters

The following random seeds were extracted and verified across the experimental codebase:

- **Split Construction:** `seed = 42` in `ai_engine/rag/scripts/split_benchmark.py`
- **Permutation Tests:** `seed = 42` with $B = 10{,}000$ paired permutations in `scripts/run_phase1_fixes_v6.py`
- **Bootstrap Confidence Intervals:** `seed = 42` with $B = 2{,}000$ bootstrap resamples
- **Conformal Calibration Splits:** Seeds `42` to `241` ($200$ consecutive seeds) in `scripts/gapfill_g02_conditional_conformal.py`
- **Synthetic OCR Perturbations:** `seed = 42` in `scripts/generate_synthetic_ocr_degradation.py`
- **500K Corpus Generation:** `seed = 42` in `scripts/gapfill_t1_04_corpus_scaling.py`

---

## 5. Security and Confidentiality Verification

- **API Keys:** Checked `.env`, `.env.example`, and all Python scripts. Zero API keys, Supabase service keys, or tokens are checked into version control or logged into artifacts.
- **Data Protection:** Personal identification numbers and customer names in SROIE and DocVQA datasets are from public academic benchmarks.

---
**Auditor Signature:** Independent Statistical & Reproducibility Auditor  
**Date:** November 4, 2026

