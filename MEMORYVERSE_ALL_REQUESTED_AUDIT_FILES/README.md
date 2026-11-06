# MemoryVerse: Complete Reproducibility and Audit Evidence Package
================================================================================
This archive consolidates all Priority 1, Priority 2, Priority 3, and Particularly
Important Files requested for independent forensic auditing and IJCDS publication.

Contents:
---------
1. particularly_important_files/
   - run_conformal_risk_control.py        : Standalone RCPS recomputation engine
   - recompute_human_relevance.py         : Human relevance & ground truth sensitivity audit
   - compute_human_agreement.py           : Dual-reviewer Cohen's kappa verification
   - conformal_pooled_cache.npz           : 2,754 query instances across 306 document clusters
   - family_4_ramp_architectural_ablations: Full RAMP H1/H2 results and subword ablation CSVs
   - family_5_funsd_form_transfer         : FUNSD Dense-BGE reruns and subset analysis CSVs
   - original_statistical_scripts/        : Verified first-principles testing & quality gates

2. priority1_experiment_evidence/
   - scripts_and_notebooks/               : All Kaggle and local reproduction notebooks
   - raw_per_query_rankings/              : Per-query CSVs for DocVQA, SROIE, COCO, Multi-Hop
   - original_paired_comparison_csvs/     : Master paired statistics and hypothesis tables
   - ramp_experiment_outputs/             : RAMP JSON benchmarks and multi-seed noise sweeps
   - funsd_bge_outputs/                   : FUNSD OCR corpora and Dense BGE evaluation outputs
   - rcps_calibration_cache/              : Conformal cascade evaluations across 200 splits

3. priority2_statistical_analysis/
   - testing_and_correction_scripts/      : Non-parametric sign-flip permutation & Holm adjustment
   - master_ledgers_and_matrices/         : Complete master recomputation CSV ledgers
   - protocol_and_hypotheses/             : Pre-specified hypothesis definitions and manifests

4. priority3_execution_logs_and_provenance/
   - gpu_and_execution_logs/              : Executed Tesla T4 GPU logs, Apple M2 logs, Kaggle logs
   - provenance_and_audit_reports/        : Forensic audit reports, blocker registers, checks
   - git_and_changelogs/                  : Git commit log (100 commits) and changelogs
   - environment_and_configs/             : Python requirements, pyproject.toml, checksums
