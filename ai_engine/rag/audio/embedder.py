"""
BGE-M3 Embedder (local, no API key required)
Generates dense text embeddings using FlagEmbedding's BAAI/bge-m3.
"""
from functools import lru_cache
from typing import List

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from ai_engine.rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _load_model() -> BGEM3FlagModel:
    return BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)


def embed_text(text: str) -> List[float]:
    """Embed a single string and return a flat float list."""
    model = _load_model()
    result = model.encode([text], batch_size=1, max_length=512)
    dense: np.ndarray = result["dense_vecs"][0]
    return dense.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings and return a list of float lists."""
    model = _load_model()
    result = model.encode(texts, batch_size=16, max_length=512)
    return result["dense_vecs"].tolist()
