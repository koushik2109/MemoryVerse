#!/usr/bin/env python3
"""
run_conformal_risk_control.py: Standalone Conformal Risk Control (RCPS) Recomputation Engine.

Evaluates Split Conformal Risk Control and High-Probability Risk-Controlling Prediction Sets (RCPS)
on the pooled benchmark of N=2,754 query instances across 306 document clusters.

Inputs:
  - conformal_pooled_cache.npz (contains margins, s1_hit10s, s2_hit10s, cluster_ids)
  - conformal_cascade_v6.csv / conformal_high_prob_v6.csv (if available for cross-validation)

Outputs:
  - 200 random cluster-stratified splits (seeds 1000..1199)
  - Marginal calibration: mean tau*, mean test risk, escalation rate, compute savings
  - High-probability Clopper-Pearson inversion bound (delta = 0.10)
  - Realized empirical split coverage
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

def clopper_pearson_upper_bound(k: int, n: int, alpha: float) -> float:
    """Exact upper Clopper-Pearson confidence bound for binomial parameter."""
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))

def run_rcps_recomputation(cache_path: Path):
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing RCPS cache: {cache_path}")
    
    print(f"Loading RCPS observation cache from: {cache_path}")
    data = np.load(cache_path)
    margins = data["margins"]
    s1_hits = data["s1_hit10s"]
    s2_hits = data["s2_hit10s"]
    cluster_ids = data["cluster_ids"]
    
    n_samples = len(margins)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    print(f"Loaded {n_samples} query instances across {n_clusters} clusters.")
    
    seeds = list(range(1000, 1200))
    target_epsilon = 0.05
    delta = 0.10
    tau_grid = np.linspace(0.0, 0.50, 501)
    
    split_records = []
    
    for seed in seeds:
        rng = np.random.RandomState(seed)
        shuffled_clusters = rng.permutation(unique_clusters)
        half = n_clusters // 2
        calib_clusters = set(shuffled_clusters[:half])
        test_clusters = set(shuffled_clusters[half:])
        
        calib_mask = np.isin(cluster_ids, list(calib_clusters))
        test_mask = np.isin(cluster_ids, list(test_clusters))
        
        calib_margins = margins[calib_mask]
        calib_s1 = s1_hits[calib_mask]
        
        test_margins = margins[test_mask]
        test_s1 = s1_hits[test_mask]
        test_s2 = s2_hits[test_mask]
        
        n_calib = len(calib_margins)
        n_test = len(test_margins)
        
        # 1. Marginal Expected Risk Calibration
        # Loss L = 1[margin >= tau and S1 == 0] (i.e. not escalated and miss)
        tau_star_marginal = 0.50
        for tau in tau_grid:
            loss = (calib_margins >= tau) & (calib_s1 == 0)
            emp_risk = (n_calib * np.mean(loss) + 1.0) / (n_calib + 1.0)
            if emp_risk <= target_epsilon:
                tau_star_marginal = float(tau)
                break
                
        # Evaluate on test half
        test_non_esc = test_margins >= tau_star_marginal
        test_misses = (test_s1 == 0) & test_non_esc
        test_risk_marginal = np.mean(test_misses)
        test_escalation_marginal = np.mean(~test_non_esc)
        
        # 2. High-Probability RCPS Calibration (Clopper-Pearson Inversion)
        tau_star_hp = 0.50
        for tau in tau_grid:
            loss = (calib_margins >= tau) & (calib_s1 == 0)
            k_misses = int(np.sum(loss))
            ucb = clopper_pearson_upper_bound(k_misses, n_calib, delta)
            if ucb <= target_epsilon:
                tau_star_hp = float(tau)
                break
                
        test_non_esc_hp = test_margins >= tau_star_hp
        test_misses_hp = (test_s1 == 0) & test_non_esc_hp
        test_risk_hp = np.mean(test_misses_hp)
        test_escalation_hp = np.mean(~test_non_esc_hp)
        
        split_records.append({
            "seed": seed,
            "tau_marginal": tau_star_marginal,
            "test_risk_marginal": test_risk_marginal,
            "cov_marginal": int(test_risk_marginal <= target_epsilon),
            "esc_marginal": test_escalation_marginal,
            "tau_hp": tau_star_hp,
            "test_risk_hp": test_risk_hp,
            "cov_hp": int(test_risk_hp <= target_epsilon),
            "esc_hp": test_escalation_hp,
        })
        
    df_splits = pd.DataFrame(split_records)
    
    mean_tau_m = df_splits["tau_marginal"].mean()
    mean_risk_m = df_splits["test_risk_marginal"].mean()
    cov_rate_m = df_splits["cov_marginal"].mean()
    mean_esc_m = df_splits["esc_marginal"].mean()
    
    mean_tau_hp = df_splits["tau_hp"].mean()
    mean_risk_hp = df_splits["test_risk_hp"].mean()
    cov_rate_hp = df_splits["cov_hp"].mean()
    mean_esc_hp = df_splits["esc_hp"].mean()
    
    print("\n" + "=" * 70)
    print("      CONFORMAL RISK CONTROL (RCPS) RECOMPUTATION RESULTS")
    print("=" * 70)
    print(f"Evaluated across {len(seeds)} random cluster-stratified splits (seeds 1000..1199)")
    print(f"Target Expected Miss Risk epsilon <= {target_epsilon:.2f}\n")
    print("1. Standard Marginal Conformal Calibration:")
    print(f"   - Mean Calibrated Threshold (tau*): {mean_tau_m:.4f}")
    print(f"   - Mean Test Miss Risk:              {mean_risk_m:.4f}  (<= {target_epsilon:.2f} GUARANTEED IN EXPECTATION)")
    print(f"   - Empirical Split Coverage:         {cov_rate_m * 100:.1f}%")
    print(f"   - Mean Escalation Rate:             {mean_esc_m * 100:.1f}%")
    print(f"   - Mean Compute Savings:             {(1.0 - mean_esc_m) * 100:.1f}%\n")
    print(f"2. High-Probability RCPS (Clopper-Pearson Inversion, delta = {delta:.2f}):")
    print(f"   - Mean Calibrated Threshold (tau*): {mean_tau_hp:.4f}")
    print(f"   - Mean Test Miss Risk:              {mean_risk_hp:.4f}")
    print(f"   - Realized Split Coverage:          {cov_rate_hp * 100:.1f}%")
    print(f"   - Mean Escalation Rate:             {mean_esc_hp * 100:.1f}%")
    print(f"   - Mean Compute Savings:             {(1.0 - mean_esc_hp) * 100:.1f}%")
    print("=" * 70 + "\n")
    
    return df_splits

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_cache = repo_root / "results" / "v3" / "conformal_pooled_cache.npz"
    if not default_cache.exists():
        default_cache = Path("conformal_pooled_cache.npz")
    if len(sys.argv) > 1:
        default_cache = Path(sys.argv[1])
    run_rcps_recomputation(default_cache)
