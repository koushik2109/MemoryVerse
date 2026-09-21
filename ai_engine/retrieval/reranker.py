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

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
            except Exception as e:
                logger.debug(f"Neural FlagReranker unavailable, using heuristic reranker: {e}")
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
                    caption = it.get("caption") or it.get("auto_caption") or ""
                    pairs.append([query, caption])
                scores = model.compute_score(pairs)
                for idx, it in enumerate(items):
                    entry = dict(it)
                    entry["rerank_score"] = float(scores[idx]) if isinstance(scores, list) else float(scores)
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
