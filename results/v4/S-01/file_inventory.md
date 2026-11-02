# File Inventory (S-01)

## 1. LaTeX Sources
- Working manuscript directory: `paper_v4/`
  - Body: `paper_v4/ijcds_body.tex`
  - Header: `paper_v4/ijcds_header.tex`
  - Main driver: `paper_v4/main.tex`
  - References: `paper_v4/references.bib`
  - Compiled PDF: `paper_v4/main.pdf`
- Reference / Baseline manuscripts:
  - `paper_v3/ijcds_body.tex`
  - `IEEE_Paper_MemoryVerse.tex`

## 2. Evaluation Scripts
- Benchmark replication & evaluation: `scripts/run_benchmarks.py`, `scripts/evaluate_sroie.py`, `scripts/evaluate_funsd.py`, `scripts/evaluate_docvqa.py`
- Statistical tests: `scripts/run_significance_test.py`, `scripts/gapfill_t3_08_tost_equivalence.py`
- Packaging scripts: `scripts/package_g01_results.py` through `package_g13_results.py`
- Figure generation: `scripts/make_regime_figure.py`, `scripts/regenerate_fig_conformal_pareto.py`, `scripts/generate_clean_architecture_figure.py`, `scripts/generate_clean_dose_response_figure.py`, `scripts/generate_clean_mechanism_figure.py`

## 3. Results Files
- `results/v3/headline_regime_grid_v3.csv`
- `results/v4/R-01/table_iv_corrected.csv`
- `results/v4/R-02/` (TOST, Holm families, DocVQA metrics)
- `results/v4/R-03/funsd_easyocr_rerun.csv`
- `results/v4/R-04/claims_correction_diff.csv`
- `results/v4/G-01/` through `G-13/`
- `results/v13_colqwen_robustness/colqwen_robustness_summary.csv`

## 4. Per-Query Ranking Logs
- `results/v4/G-09/query_formulation_results.csv`
- `results/v4/G-10/generator_query_evaluations.csv`
- `results/v4/G-11/e2e_per_query_results.csv`
- `results/v4/G-12/query_noise_results.csv`, `results/v4/G-12/per_query_failure_attribution.csv`
- `results/v4/G-13/per_query_evaluations.csv`
- `results/v10/colqwen_per_query_v10.csv`
