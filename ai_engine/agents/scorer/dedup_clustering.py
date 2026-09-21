"""
Media Scorer - Deduplication Clustering
Performs Union-Find disjoint-set clustering on CLIP/SigLIP vector representations
to identify near-duplicate clusters (cosine similarity >= 0.88) and select the single highest-quality item.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class UnionFind:
    """Disjoint-set data structure with path compression and rank union."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cluster_and_deduplicate(
    items: List[Dict[str, Any]],
    similarity_threshold: float = 0.88,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Groups items using Union-Find clustering if embedding similarity >= similarity_threshold.
    From each cluster, preserves the item with the highest composite score (quality + similarity).
    Returns (deduplicated_items, rejected_duplicates).
    """
    n = len(items)
    if n <= 1:
        return list(items), []

    uf = UnionFind(n)

    # 1. Compare all pairs with embeddings
    for i in range(n):
        emb_i = items[i].get("image_embedding") or items[i].get("embedding") or items[i].get("clip_embedding")
        if not emb_i:
            continue
        for j in range(i + 1, n):
            emb_j = items[j].get("image_embedding") or items[j].get("embedding") or items[j].get("clip_embedding")
            if not emb_j:
                continue
            sim = cosine_similarity(emb_i, emb_j)
            if sim >= similarity_threshold:
                uf.union(i, j)

    # 2. Group items by cluster root
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    # 3. For each cluster, keep the representative with the highest score
    deduplicated: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for root, member_indices in clusters.items():
        # Sort cluster members by priority: visual_quality_score or similarity
        sorted_members = sorted(
            member_indices,
            key=lambda idx: (
                items[idx].get("visual_quality_score", 0.5)
                + items[idx].get("similarity", 0.5)
            ),
            reverse=True,
        )

        # Winner
        best_item = items[sorted_members[0]]
        deduplicated.append(best_item)

        # Discard other duplicates
        for dup_idx in sorted_members[1:]:
            dup_item = dict(items[dup_idx])
            dup_item["reject_reason"] = f"near_duplicate_of_{best_item.get('id', 'cluster_lead')}"
            rejected.append(dup_item)

    return deduplicated, rejected
