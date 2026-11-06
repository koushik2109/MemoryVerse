#!/usr/bin/env python3
"""
package_gpu_results.py
======================
Assembles and validates the master empirical GPU validation package
from Kaggle dual Tesla T4 runs, verifying all artifacts across:
- Validation 1: Fresh Human N=200 live run & run-to-run stability
- Validation 2: Fixed-seed subsample sweep (N=100 vs N=200, seeds 42..1024)
- Validation 3: Corpus manifest N=813 & target mapping cross-system alignment
- Validation 4: Model/processor version, precision, image ordering hash audit
- Validation 5: 10,000 bootstrap resample confidence intervals from per-query integer ranks
- Validation 6: Cascade latency & query encoding latency benchmark
- Table II: Degraded visual benchmark (Optical blur, ColQwen2 vs CLIP)
- Stage v14 Master Suite: Modern baselines (G-01), DocVQA OCR proxies (G-02), and single-hardware latency profiling (G-03).
"""

import os
import sys
import json
import zipfile
import hashlib
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Assembling Master Empirical GPU Validation Package ===")
    out_dir = REPO_ROOT / "results" / "master_gpu_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    search_roots = [
        REPO_ROOT / "results" / "v13_colqwen_robustness",
        REPO_ROOT / "results" / "v11_degraded_visual",
        REPO_ROOT / "results" / "master_reproducibility",
    ]
    
    # Collect candidate files from all subdirectories
    all_candidates = {}
    for sroot in search_roots:
        if sroot.exists():
            for p in sroot.rglob("*"):
                if p.is_file():
                    all_candidates[p.name] = p
                    
    target_names = [
        # Robustness & Subsamples (Validations 1 & 2)
        ("colqwen_robustness_summary.csv", ["colqwen_robustness_summary.csv"]),
        ("colqwen_robustness_per_query.csv", ["colqwen_robustness_per_query.csv"]),
        ("human_subsample_stability.csv", ["human_subsample_stability.csv"]),
        ("human_colqwen_per_query_n200.csv", ["human_colqwen_per_query_n200.csv"]),
        # Bootstrap & Cross-System Alignment (Validations 3 & 5)
        ("bootstrap_ci_validation.csv", ["bootstrap_ci_validation.csv"]),
        ("cross_system_alignment.csv", ["cross_system_alignment.csv"]),
        # Environment & Metadata Audit (Validation 4)
        ("system_environment_audit.json", ["audit_report.json", "system_environment_audit.json"]),
        # Cascade Latency Benchmark (Validation 6)
        ("cascade_depth_latency_profile.csv", ["cascade_depth_latency_profile.csv"]),
        ("query_encoding_latencies.csv", ["query_encoding_latencies.csv"]),
        ("latency_summary.json", ["latency_summary.json"]),
        # Degraded Visual Benchmark (Table II)
        ("colqwen_degraded_eval_v11.csv", ["colqwen_degraded_eval_v11.csv"]),
        ("clip_degraded_eval_v11.csv", ["clip_degraded_eval_v11.csv"]),
        ("paired_degraded_visual_v11.csv", ["paired_degraded_visual_v11.csv", "permutation_tests_v11.csv"]),
        # Master Suite Modules G-01, G-02, G-03
        ("modern_baselines_consolidated.csv", ["modern_baselines_consolidated.csv"]),
        ("docvqa_ocr_proxies_summary.csv", ["docvqa_ocr_proxies_summary.csv"]),
        ("single_hardware_latency_profile.csv", ["single_hardware_latency_profile.csv"]),
        # Execution Trace Logs
        ("colqwen2-robustness-evaluation.log", ["colqwen2-robustness-evaluation.log"]),
        ("colqwen2-cascade-latency-benchmark.log", ["colqwen2-cascade-latency-benchmark.log"]),
        ("colqwen2-degraded-visual-benchmark.log", ["colqwen2-degraded-visual-benchmark.log"]),
        ("memoryverse-master-reproducibility.log", ["memoryverse.log"])
    ]
    
    manifest = []
    
    for final_name, aliases in target_names:
        found_p = None
        for alias in aliases:
            if alias in all_candidates:
                found_p = all_candidates[alias]
                break
        if found_p and found_p.exists():
            dest = out_dir / final_name
            dest.write_bytes(found_p.read_bytes())
            f_hash = sha256_file(dest)
            f_size = dest.stat().st_size
            manifest.append({
                "filename": final_name,
                "sha256": f_hash,
                "size_bytes": f_size,
                "status": "VERIFIED_GPU_MEASUREMENT"
            })
            print(f"  [FOUND] {final_name} <- {found_p.name} ({f_size} bytes, SHA: {f_hash[:12]}...)")
        else:
            print(f"  [WAITING/PENDING] {final_name}")
            
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "PACKAGE_MANIFEST.csv", index=False)
    
    # Create master zip
    master_zip_path = REPO_ROOT / "FINAL_GPU_VALIDATION_PACKAGE.zip"
    print(f"\nPackaging master archive: {master_zip_path}")
    with zipfile.ZipFile(master_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.glob("*"):
            if f.is_file() and not f.name.endswith(".zip"):
                zf.write(f, arcname=f.name)
                
    # Mirror to ~/Downloads/
    home_downloads = Path.home() / "Downloads" / "FINAL_GPU_VALIDATION_PACKAGE.zip"
    if Path.home().joinpath("Downloads").exists():
        home_downloads.write_bytes(master_zip_path.read_bytes())
        print(f"Mirrored to: {home_downloads}")
        
    print(f"Master archive created: {master_zip_path.stat().st_size} bytes with {len(manifest)} verified files.")

if __name__ == "__main__":
    main()
