#!/usr/bin/env python3
"""
package_all_requested_audit_files.py

Consolidates Priority 1, Priority 2, Priority 3, and Particularly Important Files
into MEMORYVERSE_ALL_REQUESTED_AUDIT_FILES/ and zips to MEMORYVERSE_ALL_REQUESTED_AUDIT_FILES.zip.
"""

import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "MEMORYVERSE_ALL_REQUESTED_AUDIT_FILES"

if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

# Define subdirectories
P1_NOTEBOOKS = PACKAGE_DIR / "priority1_experiment_evidence" / "scripts_and_notebooks"
P1_RANKINGS = PACKAGE_DIR / "priority1_experiment_evidence" / "raw_per_query_rankings"
P1_PAIRED = PACKAGE_DIR / "priority1_experiment_evidence" / "original_paired_comparison_csvs"
P1_RAMP = PACKAGE_DIR / "priority1_experiment_evidence" / "ramp_experiment_outputs"
P1_FUNSD = PACKAGE_DIR / "priority1_experiment_evidence" / "funsd_bge_outputs"
P1_RCPS = PACKAGE_DIR / "priority1_experiment_evidence" / "rcps_calibration_cache"

P2_SCRIPTS = PACKAGE_DIR / "priority2_statistical_analysis" / "testing_and_correction_scripts"
P2_LEDGERS = PACKAGE_DIR / "priority2_statistical_analysis" / "master_ledgers_and_matrices"
P2_PROTOCOL = PACKAGE_DIR / "priority2_statistical_analysis" / "protocol_and_hypotheses"

P3_LOGS = PACKAGE_DIR / "priority3_execution_logs_and_provenance" / "gpu_and_execution_logs"
P3_AUDIT = PACKAGE_DIR / "priority3_execution_logs_and_provenance" / "provenance_and_audit_reports"
P3_GIT = PACKAGE_DIR / "priority3_execution_logs_and_provenance" / "git_and_changelogs"
P3_ENV = PACKAGE_DIR / "priority3_execution_logs_and_provenance" / "environment_and_configs"

PARTICULAR = PACKAGE_DIR / "particularly_important_files"

for p in [P1_NOTEBOOKS, P1_RANKINGS, P1_PAIRED, P1_RAMP, P1_FUNSD, P1_RCPS,
          P2_SCRIPTS, P2_LEDGERS, P2_PROTOCOL, P3_LOGS, P3_AUDIT, P3_GIT, P3_ENV, PARTICULAR]:
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst: Path):
    if src.exists():
        if src.is_dir():
            shutil.copytree(src, dst / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst / src.name)
            return True
    return False

print("Staging Priority 1 files...")
# Notebooks and experiment scripts
notebook_files = [
    REPO_ROOT / "kaggle" / "run_colqwen_docvqa_kernel" / "kaggle_colqwen2_docvqa.ipynb",
    REPO_ROOT / "kaggle" / "run_colqwen_cascade_latency_kernel" / "kaggle_colqwen2_cascade_benchmark.ipynb",
    REPO_ROOT / "kaggle" / "run_colqwen_robustness_kernel" / "kaggle_run_v13_colqwen_robustness.ipynb",
    REPO_ROOT / "kaggle" / "run_colqwen_degraded_kernel" / "kaggle_run_v11.ipynb",
    REPO_ROOT / "kaggle" / "run_memoryverse_kernel" / "memoryverse.ipynb",
    REPO_ROOT / "kaggle" / "reproduce_family3_florence_blip2.ipynb",
    REPO_ROOT / "kaggle" / "kaggle_run_v10.ipynb",
    REPO_ROOT / "kaggle" / "kaggle_run_v14_reproducibility.ipynb",
    REPO_ROOT / "kaggle" / "kaggle_run_v5.ipynb",
    REPO_ROOT / "kaggle" / "kaggle_run_v6.ipynb",
    REPO_ROOT / "notebooks" / "ramp_rag_clean_stream_eval.ipynb",
    REPO_ROOT / "notebooks" / "reproduce_paper_v10.ipynb",
    REPO_ROOT / "ai_engine" / "rag" / "scripts" / "run_ramp_rag_benchmark.py",
    REPO_ROOT / "ai_engine" / "rag" / "scripts" / "prepare_ramp_rag_probes.py",
    REPO_ROOT / "scripts" / "gapfill_g02_conditional_conformal.py",
    REPO_ROOT / "scripts" / "gapfill_t3_02_generator_robustness.py",
    REPO_ROOT / "results" / "gapfill_v1" / "T3-07" / "run_coco_native_visual_v2.py",
    REPO_ROOT / "results" / "gapfill_v1" / "T3-07" / "gapfill_t3_07_stronger_caption_proxies.py"
]
for f in notebook_files:
    safe_copy(f, P1_NOTEBOOKS)

# Raw per-query rankings
ranking_files = [
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "docvqa_per_query_rankings.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "colqwen_per_query_v10.csv",
    REPO_ROOT / "results" / "v13_colqwen_robustness" / "colqwen_robustness_per_query.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "coco_per_query_rankings_n1000.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "per_query_g05_evaluations.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "multihop_100_cases_v6.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "expanded_multihop_results.json",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "E_HUMAN_EVALUATION" / "bm25_human_rankings_all_n400.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "E_HUMAN_EVALUATION" / "human_queries_adjudicated.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "results" / "MASTER_PER_QUERY_RESULTS.csv"
]
for f in ranking_files:
    safe_copy(f, P1_RANKINGS)

# Paired comparisons
paired_files = [
    REPO_ROOT / "MASTER_PAIRED_STATISTICS.csv",
    REPO_ROOT / "MASTER_PAIRED_TESTS.csv",
    REPO_ROOT / "results" / "v10" / "colqwen_paired_v10.csv",
    REPO_ROOT / "FAMILY3_PAIRED_STATISTICS.csv",
    REPO_ROOT / "HUMAN_QUERY_RELEVANCE_SENSITIVITY.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "E_HUMAN_EVALUATION" / "human_queries_relevance_summary.csv",
    REPO_ROOT / "HUMAN_QUERY_DUPLICATE_TARGET_AUDIT.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "practical_stack_ablation_v7.csv",
    REPO_ROOT / "CASCADE_DEPTH_RESULTS.csv"
]
for f in paired_files:
    safe_copy(f, P1_PAIRED)

# RAMP outputs
ramp_files = [
    REPO_ROOT / "ai_engine" / "rag" / "data" / "ramp_rag_benchmark_results.json",
    REPO_ROOT / "ai_engine" / "rag" / "data" / "ramp_rag_probes_split.json",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "subword_ablation_results.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "subword_ablation_summary.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "practical_stack_ablation.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "practical_stack_ablation_v7.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations" / "gapfill_t2_03_subword_ablation.py",
    REPO_ROOT / "ai_engine" / "rag" / "scripts" / "run_ramp_rag_benchmark.py"
]
for f in ramp_files:
    safe_copy(f, P1_RAMP)

# FUNSD BGE outputs
funsd_files = [
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_5_funsd_form_transfer" / "funsd_easyocr_rerun.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_5_funsd_form_transfer" / "funsd_easyocr_subset_analysis.csv",
    REPO_ROOT / "results" / "v3" / "funsd_standardized_pools_b08.csv",
    REPO_ROOT / "results" / "all_statistical_families_package" / "family_5_funsd_form_transfer" / "rerun_funsd_easyocr_r03.py",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "funsd_probes.json",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "funsd_apple_vision_ocr.json",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "funsd_easyocr_clean_ocr.json",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "raw_data" / "funsd_tesseract_ocr.json"
]
for f in funsd_files:
    safe_copy(f, P1_FUNSD)

# RCPS Calibration Cache
rcps_files = [
    REPO_ROOT / "results" / "v3" / "conformal_pooled_cache.npz",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "conformal_cascade_v6.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "conformal_high_prob_v6.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "conformal_splits_summary_v6.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "conditional_conformal_summary.csv",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "statistical_conformal_tests.csv",
    REPO_ROOT / "scripts" / "run_conformal_risk_control.py",
    REPO_ROOT / "SUBMISSION_BLOCKER_AND_AUDIT_DOSSIER" / "C_RCPS_CONFORMAL_RISK_CONTROL" / "run_conformal_cascade_v2.py"
]
for f in rcps_files:
    safe_copy(f, P1_RCPS)

print("Staging Priority 2 files...")
# Statistical testing & correction scripts
stat_scripts = [
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "scripts" / "recompute_verified_results.py",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "scripts" / "independent_validation.py",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "scripts" / "automated_quality_gates.py",
    REPO_ROOT / "scripts" / "recompute_multihop_tost.py",
    REPO_ROOT / "scripts" / "recompute_rcps_splits.py",
    REPO_ROOT / "scripts" / "recompute_human_agreement.py",
    REPO_ROOT / "scripts" / "compute_human_agreement.py",
    REPO_ROOT / "scripts" / "recompute_human_relevance.py",
    REPO_ROOT / "scripts" / "run_conformal_risk_control.py",
    REPO_ROOT / "scripts" / "independent_family3_pipeline.py",
    REPO_ROOT / "scripts" / "generate_all_master_audit_ledgers.py"
]
for f in stat_scripts:
    safe_copy(f, P2_SCRIPTS)

# Master ledgers & matrices
ledger_files = [
    REPO_ROOT / "MASTER_RECOMPUTED_RESULTS.csv",
    REPO_ROOT / "MASTER_BOOTSTRAP_CI.csv",
    REPO_ROOT / "MASTER_CONFIDENCE_INTERVALS.csv",
    REPO_ROOT / "MASTER_PAIRED_STATISTICS.csv",
    REPO_ROOT / "MASTER_HOLM_CORRECTIONS.csv",
    REPO_ROOT / "MASTER_MULTIPLE_TESTING_AUDIT.csv",
    REPO_ROOT / "MASTER_TOST_RESULTS.csv",
    REPO_ROOT / "FAMILY3_TOST_SENSITIVITY.csv",
    REPO_ROOT / "MASTER_RCPS_RESULTS.csv",
    REPO_ROOT / "MASTER_HUMAN_AGREEMENT.csv",
    REPO_ROOT / "MASTER_LATENCY_RESULTS.csv"
]
for f in ledger_files:
    safe_copy(f, P2_LEDGERS)

# Protocol & hypothesis definitions
protocol_files = [
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "config" / "pipeline_config.json",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "config" / "RAW_DATA_MANIFEST.csv",
    REPO_ROOT / "FAMILY3_EVIDENCE_INVENTORY.csv",
    REPO_ROOT / "EXPERIMENT_INVENTORY.md",
    REPO_ROOT / "EXPERIMENT_REUSE_MATRIX.csv",
    REPO_ROOT / "FINAL_CLAIM_TO_EVIDENCE_TRACEABILITY.csv",
    REPO_ROOT / "STATISTICAL_RESULTS.md"
]
for f in protocol_files:
    safe_copy(f, P2_PROTOCOL)

print("Staging Priority 3 files...")
# Logs
log_files = [
    REPO_ROOT / "results" / "master_gpu_validation" / "colqwen2-robustness-evaluation.log",
    REPO_ROOT / "results" / "master_gpu_validation" / "colqwen2-cascade-latency-benchmark.log",
    REPO_ROOT / "results" / "master_gpu_validation" / "colqwen2-degraded-visual-benchmark.log",
    REPO_ROOT / "results" / "v13_colqwen_robustness" / "kaggle_measured_run_20261105.log",
    REPO_ROOT / "results" / "v5" / "kaggle_run_v5_execution.log",
    REPO_ROOT / "results" / "gapfill_v1" / "T3-07" / "t3_07_execution.log",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "logs" / "recomputation_execution.log",
    REPO_ROOT / "audit" / "family3_independent_reconstruction.log",
    REPO_ROOT / "results" / "master_reproducibility" / "memoryverse.log"
]
for f in log_files:
    safe_copy(f, P3_LOGS)

# Audit reports & dossiers
audit_files = [
    REPO_ROOT / "MASTER_SOURCE_INVENTORY.csv",
    REPO_ROOT / "MASTER_EVIDENCE_PROVENANCE.csv",
    REPO_ROOT / "MASTER_SUPERSEDED_FILES.csv",
    REPO_ROOT / "MASTER_BLOCKER_REGISTER.csv",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "FINAL_BLOCKER_REGISTER.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "HARD_CODED_STATISTIC_FINDINGS.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "MISSING_INPUT_REGISTER.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "STATISTICAL_VALIDATION_REPORT.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "MANUSCRIPT_TO_RESULTS_CONSISTENCY_REPORT.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "INDEPENDENT_VALIDATION_REPORT.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "audit" / "AUTOMATED_QUALITY_GATES_REPORT.md"
]
for f in audit_files:
    safe_copy(f, P3_AUDIT)

# Git & Changelogs
# Export git log
git_log_path = P3_GIT / "git_commit_log.txt"
try:
    res = subprocess.run(["git", "log", "-n", "100", "--stat"], capture_output=True, text=True, check=True)
    git_log_path.write_text(res.stdout, encoding="utf-8")
except Exception as e:
    git_log_path.write_text(f"Could not export git log: {e}", encoding="utf-8")

git_files = [
    REPO_ROOT / "CHANGELOG_V4.md",
    REPO_ROOT / "FINAL_CHANGELOG.md",
    REPO_ROOT / "CAMERA_READY_CHANGELOG.md",
    REPO_ROOT / "TYPESETTING_CHANGELOG.md"
]
for f in git_files:
    safe_copy(f, P3_GIT)

# Environment & Config
env_files = [
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / "README.md",
    REPO_ROOT / "MEMORYVERSE_IJCDS_VERIFIED_RECOMPUTATION" / "SHA256SUMS.txt"
]
for f in env_files:
    safe_copy(f, P3_ENV)

print("Staging Particularly Important Files directly into particularly_important_files/...")
# Specific files called out by the user:
# 1. run_conformal_risk_control.py
safe_copy(REPO_ROOT / "scripts" / "run_conformal_risk_control.py", PARTICULAR)
# 2. recompute_human_relevance.py
safe_copy(REPO_ROOT / "scripts" / "recompute_human_relevance.py", PARTICULAR)
# 3. compute_human_agreement.py
safe_copy(REPO_ROOT / "scripts" / "compute_human_agreement.py", PARTICULAR)
# 4. conformal_pooled_cache.npz
safe_copy(REPO_ROOT / "results" / "v3" / "conformal_pooled_cache.npz", PARTICULAR)
# 5. RAMP H1/H2 result directories
safe_copy(REPO_ROOT / "results" / "all_statistical_families_package" / "family_4_ramp_architectural_ablations", PARTICULAR)
# 6. FUNSD Dense-BGE outputs
safe_copy(REPO_ROOT / "results" / "all_statistical_families_package" / "family_5_funsd_form_transfer", PARTICULAR)
# 7. Original statistical scripts
orig_stat_dir = PARTICULAR / "original_statistical_scripts"
orig_stat_dir.mkdir(parents=True, exist_ok=True)
for f in [REPO_ROOT / "scripts" / "recompute_verified_results.py",
         REPO_ROOT / "scripts" / "independent_validation.py",
         REPO_ROOT / "scripts" / "automated_quality_gates.py",
         REPO_ROOT / "scripts" / "recompute_multihop_tost.py",
         REPO_ROOT / "scripts" / "recompute_rcps_splits.py",
         REPO_ROOT / "scripts" / "generate_all_master_audit_ledgers.py"]:
    safe_copy(f, orig_stat_dir)

# Create Root README in PACKAGE_DIR
pkg_readme = PACKAGE_DIR / "README.md"
pkg_readme.write_text("""# MemoryVerse: Complete Reproducibility and Audit Evidence Package
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
""", encoding="utf-8")

# Generate SHA256SUMS.txt
print("Generating SHA256SUMS.txt across package...")
sha_lines = []
manifest_rows = [["Relative_Path", "Size_Bytes", "SHA256_Checksum"]]

for root, _, files in os.walk(PACKAGE_DIR):
    for f in sorted(files):
        if f in ["SHA256SUMS.txt", "PACKAGE_MANIFEST.csv"]:
            continue
        p = Path(root) / f
        rel_p = p.relative_to(PACKAGE_DIR).as_posix()
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        size = len(data)
        sha_lines.append(f"{sha}  {rel_p}")
        manifest_rows.append([rel_p, str(size), sha])

(PACKAGE_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

import csv
with open(PACKAGE_DIR / "PACKAGE_MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(manifest_rows)

print(f"Cataloged {len(sha_lines)} files in package.")

# Create ZIP archive
zip_path = REPO_ROOT / "MEMORYVERSE_ALL_REQUESTED_AUDIT_FILES.zip"
print(f"Compressing into {zip_path.name}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(PACKAGE_DIR):
        for f in sorted(files):
            p = Path(root) / f
            arcname = p.relative_to(REPO_ROOT).as_posix()
            zf.write(p, arcname=arcname)

zip_size = zip_path.stat().st_size
print(f"Created {zip_path.name} ({zip_size / (1024*1024):.2f} MB).")

# Mirror to ~/Downloads
downloads_dir = Path.home() / "Downloads"
if downloads_dir.exists():
    shutil.copy2(zip_path, downloads_dir / zip_path.name)
    print(f"Mirrored to {downloads_dir / zip_path.name}")
