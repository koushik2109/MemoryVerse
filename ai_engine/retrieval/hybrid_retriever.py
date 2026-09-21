"""
Hybrid Retrieval Module
Combines dense semantic vector search (BGE-M3 / SigLIP) with sparse lexical scoring (BM25 / token matching).
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np


class HybridRetriever:
    """
    Executes hybrid retrieval combining dense vector similarity with sparse lexical matching.
    """

    def __init__(self, alpha: float = 0.7):
        """
        alpha: weight for dense similarity (0.0 to 1.0).
        1 - alpha: weight for sparse lexical matching.
        """
        self.alpha = alpha

    def compute_lexical_score(self, query: str, document_text: str) -> float:
        """Simple BM25-inspired term frequency and overlap score."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        if not q_tokens:
            return 0.0
        doc_tokens = re.findall(r"\w+", document_text.lower())
        if not doc_tokens:
            return 0.0

        matches = sum(1 for t in doc_tokens if t in q_tokens)
        tf = matches / max(1, len(doc_tokens))
        coverage = len(set(doc_tokens) & q_tokens) / len(q_tokens)
        return min(1.0, round(0.6 * coverage + 0.4 * tf, 4))

    def retrieve(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Scores candidate items using hybrid linear combination:
            hybrid_score = alpha * dense_score + (1 - alpha) * lexical_score
        """
        scored = []
        for cand in candidates:
            entry = dict(cand)
            dense_score = float(entry.get("similarity", entry.get("dense_score", 0.5)))
            text = f"{entry.get('caption', '')} {entry.get('auto_caption', '')} {entry.get('title', '')}"
            lexical_score = self.compute_lexical_score(query, text)

            hybrid = (self.alpha * dense_score) + ((1.0 - self.alpha) * lexical_score)
            entry["dense_score"] = round(dense_score, 4)
            entry["lexical_score"] = round(lexical_score, 4)
            entry["hybrid_score"] = round(hybrid, 4)
            scored.append(entry)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]
