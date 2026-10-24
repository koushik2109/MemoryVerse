"""
Standardized Statistical Evaluation Library (eval_lib)
======================================================
Unified implementation of metrics, cluster-bootstrapped confidence intervals,
paired permutation significance tests, and step-down Holm-Bonferroni correction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.stats import beta


def hit_at_k(retrieved_ids: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Computes Hit@K (1.0 if at least one ground-truth document appears in top-k, 0.0 otherwise)."""
    return float(bool(set(retrieved_ids[:k]) & ground_truth))


def mrr(retrieved_ids: Sequence[str], ground_truth: Set[str]) -> float:
    """Computes Mean Reciprocal Rank (1/rank of first retrieved relevant document, 0.0 if none)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int = 10) -> float:
    """Computes Discounted Cumulative Gain at K."""
    val = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            val += rel / math.log2(i + 2)
    return val


def ndcg_at_k(retrieved_ids: Sequence[str], ground_truth: Set[str], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at K for binary relevance."""
    relevances = [1 if doc_id in ground_truth else 0 for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = [1] * min(len(ground_truth), k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval for proportions."""
    if n <= 0:
        return 0.0, 0.0
    k = max(0, min(k, n))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def cluster_bootstrap_ci(
    scores: Sequence[float],
    cluster_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes confidence interval resampling clusters with replacement.
    If cluster_ids is provided, scores are aggregated by cluster before resampling
    to prevent artificially inflated sample size when multiple seeds evaluate identical probes.
    If scores are completely saturated (all 1.0 or 0.0), falls back to Clopper-Pearson.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n_total = len(scores_arr)
    if n_total == 0:
        return 0.0, 0.0

    mean_val = float(np.mean(scores_arr))
    if mean_val >= 0.9999 or mean_val <= 0.0001:
        k = int(round(np.sum(scores_arr)))
        n_effective = len(set(cluster_ids)) if cluster_ids is not None else n_total
        k_effective = int(round(mean_val * n_effective))
        return clopper_pearson_ci(k_effective, n_effective, alpha=1.0 - ci)

    if cluster_ids is not None:
        cluster_map: Dict[Any, List[float]] = defaultdict(list)
        for c, s in zip(cluster_ids, scores_arr):
            cluster_map[c].append(s)
        unique_clusters = list(cluster_map.keys())
        cluster_means = np.array([np.mean(cluster_map[c]) for c in unique_clusters], dtype=np.float64)
    else:
        cluster_means = scores_arr

    n_clusters = len(cluster_means)
    if n_clusters == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_clusters, size=(n_boot, n_clusters), replace=True)
    boot_estimates = np.mean(cluster_means[boot_indices], axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def cluster_bootstrap_diff_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    cluster_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes cluster-bootstrapped confidence interval on the paired difference (scores_a - scores_b)."""
    diff_arr = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    n_total = len(diff_arr)
    if n_total == 0:
        return 0.0, 0.0

    if cluster_ids is not None:
        cluster_map: Dict[Any, List[float]] = defaultdict(list)
        for c, d in zip(cluster_ids, diff_arr):
            cluster_map[c].append(d)
        unique_clusters = list(cluster_map.keys())
        cluster_diffs = np.array([np.mean(cluster_map[c]) for c in unique_clusters], dtype=np.float64)
    else:
        cluster_diffs = diff_arr

    n_clusters = len(cluster_diffs)
    if n_clusters == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_clusters, size=(n_boot, n_clusters), replace=True)
    boot_estimates = np.mean(cluster_diffs[boot_indices], axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    cluster_ids: Optional[Sequence[Any]] = None,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired sign-flip permutation test across cluster-level paired differences.
    Tests null hypothesis H0: E[scores_a - scores_b] == 0.
    """
    diff_arr = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    if len(diff_arr) == 0:
        return 1.0

    if cluster_ids is not None:
        cluster_map: Dict[Any, List[float]] = defaultdict(list)
        for c, d in zip(cluster_ids, diff_arr):
            cluster_map[c].append(d)
        unique_clusters = list(cluster_map.keys())
        cluster_diffs = np.array([np.mean(cluster_map[c]) for c in unique_clusters], dtype=np.float64)
    else:
        cluster_diffs = diff_arr

    obs_diff = float(np.abs(np.mean(cluster_diffs)))
    if obs_diff < 1e-12:
        return 1.0

    n_clusters = len(cluster_diffs)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_clusters))
    perm_diffs = np.abs(np.mean(signs * cluster_diffs, axis=1))
    p_val = float(np.mean(perm_diffs >= obs_diff))
    return max(p_val, 1.0 / n_permutations)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Step-down Holm-Bonferroni family-wise error rate control.
    Returns adjusted p-values enforcing monotonicity: p_adj[rank] = max_{k <= rank} min(1.0, (m - k + 1) * p_k).
    """
    if not p_values:
        return {}
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p: Dict[str, float] = {}
    cum_max = 0.0
    for rank, (key, p) in enumerate(sorted_items):
        multiplier = m - rank
        raw_adj = p * multiplier
        adj = min(1.0, max(cum_max, raw_adj))
        cum_max = adj
        adj_p[key] = round(adj, 5)
    return adj_p
