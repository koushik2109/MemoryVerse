"""
Unit Tests for eval_lib
=======================
Tests metrics, cluster bootstrap, paired permutation, and Holm-Bonferroni correction.
"""

import math
import numpy as np
import pytest

from ai_engine.rag.eval_lib import (
    clopper_pearson_ci,
    cluster_bootstrap_ci,
    cluster_bootstrap_diff_ci,
    hit_at_k,
    holm_bonferroni,
    mrr,
    ndcg_at_k,
    paired_permutation_test,
)


def test_hit_at_k():
    gt = {"doc_1", "doc_2"}
    assert hit_at_k(["doc_1", "doc_3", "doc_4"], gt, k=1) == 1.0
    assert hit_at_k(["doc_3", "doc_1", "doc_4"], gt, k=1) == 0.0
    assert hit_at_k(["doc_3", "doc_1", "doc_4"], gt, k=2) == 1.0
    assert hit_at_k(["doc_3", "doc_4", "doc_5"], gt, k=10) == 0.0


def test_mrr():
    gt = {"doc_target"}
    assert mrr(["doc_target", "doc_other"], gt) == 1.0
    assert mrr(["doc_a", "doc_target", "doc_b"], gt) == 0.5
    assert mrr(["doc_a", "doc_b", "doc_c", "doc_target"], gt) == 0.25
    assert mrr(["doc_a", "doc_b"], gt) == 0.0


def test_ndcg_at_k():
    gt = {"doc_1"}
    # Rank 1 hit: NDCG should be 1.0
    assert math.isclose(ndcg_at_k(["doc_1", "doc_2"], gt, k=10), 1.0)
    # Rank 2 hit: NDCG should be 1/log2(3) / (1/log2(2)) = 1/log2(3)
    expected_rank2 = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(ndcg_at_k(["doc_2", "doc_1"], gt, k=10), expected_rank2)
    # Miss
    assert ndcg_at_k(["doc_2", "doc_3"], gt, k=10) == 0.0


def test_clopper_pearson_ci():
    low, high = clopper_pearson_ci(55, 55, alpha=0.05)
    assert low > 0.90
    assert math.isclose(high, 1.0)

    low_zero, high_zero = clopper_pearson_ci(0, 55, alpha=0.05)
    assert math.isclose(low_zero, 0.0)
    assert high_zero < 0.10


def test_cluster_bootstrap_ci():
    # Clustered probes: 3 seeds evaluate 10 probe clusters
    cluster_ids = [c for c in range(10) for _ in range(3)]
    # Scores: each cluster has a base score
    rng = np.random.default_rng(123)
    cluster_base = rng.uniform(0.7, 0.9, size=10)
    scores = [float(cluster_base[c] + rng.normal(0, 0.01)) for c in cluster_ids]

    low, high = cluster_bootstrap_ci(scores, cluster_ids=cluster_ids, n_boot=1000, seed=42)
    mean_val = float(np.mean(cluster_base))
    assert low <= mean_val <= high
    assert 0.65 < low < 0.95


def test_cluster_bootstrap_diff_ci():
    cluster_ids = [c for c in range(20) for _ in range(3)]
    scores_a = [0.9 for _ in cluster_ids]
    scores_b = [0.8 for _ in cluster_ids]
    low, high = cluster_bootstrap_diff_ci(scores_a, scores_b, cluster_ids=cluster_ids, n_boot=1000, seed=42)
    assert 0.09 < low <= 0.10 <= high < 0.11


def test_paired_permutation_test():
    cluster_ids = list(range(30))
    # System A clearly superior to System B
    scores_a = [1.0] * 30
    scores_b = [0.2] * 30
    p_val = paired_permutation_test(scores_a, scores_b, cluster_ids=cluster_ids, n_permutations=2000, seed=42)
    assert p_val < 0.001

    # Identical systems: p-val should be 1.0
    p_identical = paired_permutation_test(scores_a, scores_a, cluster_ids=cluster_ids, n_permutations=1000, seed=42)
    assert math.isclose(p_identical, 1.0)


def test_holm_bonferroni():
    raw_p = {
        "m1": 0.001,
        "m2": 0.010,
        "m3": 0.040,
        "m4": 0.150,
    }
    adj_p = holm_bonferroni(raw_p)
    # Monotonicity check
    assert adj_p["m1"] <= adj_p["m2"] <= adj_p["m3"] <= adj_p["m4"]
    # m1 adjusted: 0.001 * 4 = 0.004
    assert math.isclose(adj_p["m1"], 0.004)
    # m2 adjusted: max(0.004, 0.010 * 3) = 0.030
    assert math.isclose(adj_p["m2"], 0.030)
    # m3 adjusted: max(0.030, 0.040 * 2) = 0.080
    assert math.isclose(adj_p["m3"], 0.080)
    # m4 adjusted: max(0.080, 0.150 * 1) = 0.150
    assert math.isclose(adj_p["m4"], 0.150)
