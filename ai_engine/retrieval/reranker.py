"""
Cross-Encoder Reranker Interface
Provides neural passage re-ranking for top-K retrieved multimodal candidates.
"""
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-scores (query, passage) pairs to elevate the most relevant memories.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._backend = None

    def _get_model(self):
        if self._model is None:
            # 1. Try sentence-transformers CrossEncoder
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._backend = "sentence_transformers"
                return self._model
            except Exception as e:
                logger.debug(f"sentence_transformers.CrossEncoder unavailable: {e}")

            # 2. Try FlagEmbedding FlagReranker
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._backend = "flag_embedding"
                return self._model
            except Exception as e:
                logger.debug(f"FlagReranker unavailable: {e}")
        return self._model

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Re-orders items based on cross-attention / semantic relevance.
        """
        if not items:
            return []

        model = self._get_model()
        reranked = []

        if model is not None:
            try:
                pairs = []
                for it in items:
                    caption = it.get("caption") or it.get("auto_caption") or it.get("title") or ""
                    pairs.append([query, caption])

                if self._backend == "sentence_transformers":
                    scores: Any = model.predict(pairs)
                else:
                    scores = model.compute_score(pairs)

                for idx, it in enumerate(items):
                    entry = dict(it)
                    try:
                        entry["rerank_score"] = float(scores[idx]) if hasattr(scores, "__getitem__") else float(scores)
                    except Exception:
                        entry["rerank_score"] = 0.5
                    reranked.append(entry)
                reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
                return reranked[:top_k]
            except Exception as e:
                logger.debug(f"Cross-encoder scoring error, falling back: {e}")

        # Heuristic fallback
        for it in items:
            entry = dict(it)
            sim = float(entry.get("hybrid_score", entry.get("similarity", 0.5)))
            quality = float(entry.get("visual_quality_score", entry.get("quality_score", 0.7)))
            entry["rerank_score"] = round(0.7 * sim + 0.3 * quality, 4)
            reranked.append(entry)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def cascade_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        margin_threshold: float = 0.065,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Confidence-Triggered Cascade Ranking:
        Evaluates top decision margin delta_1,2 = score_1 - score_2 from Stage 1.
        - If delta_1,2 >= margin_threshold: Stage 1 ranking is accepted immediately (0 ms neural overhead).
        - If delta_1,2 < margin_threshold: Candidates (top-k) escalate to the neural cross-encoder.
        """
        if len(candidates) <= 1:
            return candidates[:top_k]

        s1 = float(candidates[0].get("hybrid_score", candidates[0].get("similarity", 0.0)))
        s2 = float(candidates[1].get("hybrid_score", candidates[1].get("similarity", 0.0)))
        margin = s1 - s2

        if margin >= margin_threshold:
            # Stage 1 ranking accepted; high confidence decision
            return candidates[:top_k]

        # Margin is ambiguous; escalate top candidates to neural cross-encoder
        return self.rerank(query, candidates, top_k=top_k)
