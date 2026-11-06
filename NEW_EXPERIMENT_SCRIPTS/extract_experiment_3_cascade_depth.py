#!/usr/bin/env python3
"""
NEW_EXPERIMENT_SCRIPTS/extract_experiment_3_cascade_depth.py
MemoryVerse — Experiment 3: Cascade Candidate-Depth Sensitivity Synthesis

Merges pre-computed candidate depth sweep (results/gapfill_v1/T2-04/cascade_per_k_sweep.csv)
with live measured hardware benchmarks:
- Apple M2 CPU (FINAL_BLOCKER_RESOLUTION/cascade_measured_latency_cpu.csv)
- Apple Silicon MPS (FINAL_BLOCKER_RESOLUTION/cascade_measured_latency.csv)

Outputs:
- CASCADE_DEPTH_RESULTS.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    repo_root = Path(".")
    
    # 1. Load cascade sweep
    sweep_csv = repo_root / "results/gapfill_v1/T2-04/cascade_per_k_sweep.csv"
    df_sweep = pd.read_csv(sweep_csv)
    print(f"Loaded {len(df_sweep)} rows from {sweep_csv}")
    
    # Filter to primary recommended stage 1 retriever: Word BM25
    word_sweep = df_sweep[df_sweep["stage1_retriever"] == "Word BM25"].copy()

    # 2. Load CPU and MPS measured latencies
    cpu_csv = repo_root / "FINAL_BLOCKER_RESOLUTION/cascade_measured_latency_cpu.csv"
    mps_csv = repo_root / "FINAL_BLOCKER_RESOLUTION/cascade_measured_latency.csv"
    df_cpu = pd.read_csv(cpu_csv)
    df_mps = pd.read_csv(mps_csv)
    
    # Full baseline latencies (K=813)
    full_cpu_lat = df_cpu[df_cpu["K1_cutoff"] == 813]["total_mean_latency_ms"].values[0]
    full_mps_lat = df_mps[df_mps["K1_cutoff"] == 813]["total_mean_latency_ms"].values[0]

    # Map hardware measurements
    cpu_map = {int(r["K1_cutoff"]): r for _, r in df_cpu.iterrows()}
    mps_map = {int(r["K1_cutoff"]): r for _, r in df_mps.iterrows()}

    out_rows = []
    for _, r in word_sweep.iterrows():
        k = int(r["K_candidates"])
        c_info = cpu_map.get(k, {})
        m_info = mps_map.get(k, {})
        
        cpu_mean = c_info.get("total_mean_latency_ms", np.nan)
        cpu_p50 = c_info.get("total_p50_ms", np.nan)
        cpu_p95 = c_info.get("total_p95_ms", np.nan)
        
        mps_mean = m_info.get("total_mean_latency_ms", np.nan)
        mps_p50 = m_info.get("total_p50_ms", np.nan)
        mps_p95 = m_info.get("total_p95_ms", np.nan)

        cpu_speedup = full_cpu_lat / cpu_mean if not np.isnan(cpu_mean) else np.nan
        mps_speedup = full_mps_lat / mps_mean if not np.isnan(mps_mean) else np.nan

        out_rows.append({
            "stage1_retriever": "Word BM25",
            "K_candidates": k,
            "candidate_recall_hit_at_k": r["stage1_hit_at_k"],
            "cascade_mrr": r["cascade_mrr"],
            "cascade_hit1": r["cascade_hit1"],
            "cascade_hit10": r["cascade_hit10"],
            "pct_full_colqwen_retained": r["pct_full_colqwen_retained"],
            "model_queue_latency_ms": r["estimated_latency_ms"],
            "measured_cpu_mean_ms": cpu_mean,
            "measured_cpu_p50_ms": cpu_p50,
            "measured_cpu_p95_ms": cpu_p95,
            "measured_cpu_speedup": cpu_speedup,
            "measured_mps_mean_ms": mps_mean,
            "measured_mps_p50_ms": mps_p50,
            "measured_mps_p95_ms": mps_p95,
            "measured_mps_speedup": mps_speedup,
        })

    df_out = pd.DataFrame(out_rows)
    out_csv = repo_root / "CASCADE_DEPTH_RESULTS.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} cascade depth evaluations to {out_csv}")
    print("\n=== CASCADE CANDIDATE-DEPTH SENSITIVITY TABLE ===")
    print(df_out[["K_candidates", "candidate_recall_hit_at_k", "cascade_mrr", "cascade_hit1", "pct_full_colqwen_retained", "measured_cpu_mean_ms", "measured_mps_mean_ms"]].to_string())

if __name__ == "__main__":
    main()

