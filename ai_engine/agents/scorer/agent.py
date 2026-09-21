"""
Media Scorer Agent
Performs aesthetic evaluation, Laplacian blur filtering, near-duplicate suppression,
and multi-attribute ranking for memory curation.
"""
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
from ai_engine.agents.scorer.quality_metrics import (
    compute_laplacian_variance,
    compute_exposure_and_luminance,
    evaluate_media_quality,
)
from ai_engine.agents.scorer.dedup_clustering import (
    cluster_and_deduplicate,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


class MediaScorerAgent:
    """
    Evaluates visual quality, rejects low-quality/blurry assets, and suppresses duplicates.
    """

    def __init__(self, blur_threshold: float = 80.0, duplicate_similarity_threshold: float = 0.88):
        self.blur_threshold = blur_threshold
        self.duplicate_similarity_threshold = duplicate_similarity_threshold

    def compute_blur_score(self, image_path: str) -> float:
        """
        Computes the Laplacian variance of an image to measure sharpness.
        Higher variance implies sharp edges; low variance (<80) implies blur.
        """
        return compute_laplacian_variance(image_path)

    def calculate_embedding_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Cosine similarity between two vector embeddings."""
        return cosine_similarity(emb1, emb2)

    def deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out redundant near-duplicate frames/photos with cosine similarity > threshold.
        """
        unique_items: List[Dict[str, Any]] = []

        for item in items:
            is_dup = False
            emb = item.get("image_embedding") or item.get("embedding")

            for kept in unique_items:
                kept_emb = kept.get("image_embedding") or kept.get("embedding")
                if emb is not None and kept_emb is not None and len(emb) == len(kept_emb):
                    sim = self.calculate_embedding_similarity(emb, kept_emb)
                    if sim >= self.duplicate_similarity_threshold:
                        is_dup = True
                        break
                else:
                    # Caption-based Jaccard similarity fallback
                    c1 = set((item.get("auto_caption") or "").lower().split())
                    c2 = set((kept.get("auto_caption") or "").lower().split())
                    if c1 and c2:
                        jaccard = len(c1 & c2) / float(len(c1 | c2))
                        if jaccard > 0.75:
                            is_dup = True
                            break

            if not is_dup:
                unique_items.append(item)

        return unique_items

    def score_items(self, items: List[Dict[str, Any]], target_mood: str = "neutral") -> List[Dict[str, Any]]:
        """
        Calculates composite quality and relevance score for each media item.
        """
        scored = []
        for idx, item in enumerate(items):
            file_path = item.get("file_path", "")
            blur_score = self.compute_blur_score(file_path)
            is_blurry = blur_score < self.blur_threshold

            # Visual quality score normalized (0.0 to 1.0)
            norm_quality = min(1.0, blur_score / 200.0)

            # Retrieval similarity (default 0.7 if not provided)
            relevance = float(item.get("similarity", 0.75))

            # Temporal pacing bonus (spread items across temporal index)
            temporal_bonus = 0.1 * (1.0 - (idx / max(1, len(items))))

            # Composite formula: 0.45 * relevance + 0.35 * quality + 0.20 * balance
            composite = (0.45 * relevance) + (0.35 * norm_quality) + (0.20 * temporal_bonus)

            item_copy = dict(item)
            item_copy["blur_score"] = round(blur_score, 1)
            item_copy["is_blurry"] = is_blurry
            item_copy["quality_score"] = round(norm_quality, 3)
            item_copy["composite_score"] = round(composite, 3)
            scored.append(item_copy)

        # Sort descending by composite score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node execution."""
        retrieved = state.get("retrieved_items", [])
        plan = state.get("plan", {})
        mood = plan.get("target_mood", "neutral")

        # 1. Deduplicate
        unique_items = self.deduplicate(retrieved)

        # 2. Score & rank
        scored_items = self.score_items(unique_items, target_mood=mood)

        return {
            "scored_items": scored_items,
        }
