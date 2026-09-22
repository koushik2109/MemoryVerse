"""
BGE-M3 Model Loader
Singleton thread-safe loader for BAAI/bge-m3 dense text embedder.
Output dimension: 1024
Used for:
  - Multilingual caption embedding
  - Video and audio transcript search
  - High-precision dense semantic retrieval
"""
import logging
import threading
from typing import List, Any, cast
import numpy as np

logger = logging.getLogger(__name__)

BGE_MODEL_ID = "BAAI/bge-m3"
EMBED_DIM = 1024

_bge_model = None
_lock = threading.Lock()


def get_bge_model():
    """Thread-safe singleton getter for BGE-M3 model."""
    global _bge_model
    if _bge_model is None:
        with _lock:
            if _bge_model is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    logger.info(f"Loading BGE-M3 model via FlagEmbedding ({BGE_MODEL_ID})...")
                    _bge_model = BGEM3FlagModel(BGE_MODEL_ID, use_fp16=True)
                except Exception:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading BGE-M3 model via SentenceTransformer fallback ({BGE_MODEL_ID})...")
                    _bge_model = SentenceTransformer(BGE_MODEL_ID)
    return _bge_model


def embed_text(text: str) -> List[float]:
    """Embed a single string into a 1024-dim float list."""
    model: Any = get_bge_model()
    if model is None:
        return [0.0] * EMBED_DIM
    try:
        from FlagEmbedding import BGEM3FlagModel
        if isinstance(model, BGEM3FlagModel):
            res: Any = model.encode([text], batch_size=1, max_length=512)
            dense = res["dense_vecs"][0]
            return [float(x) for x in dense.tolist()]
    except Exception:
        pass
    # SentenceTransformer fallback
    try:
        vec: Any = model.encode(text)
        if hasattr(vec, "tolist"):
            return [float(x) for x in vec.tolist()]
        return [float(x) for x in vec]
    except Exception:
        return [0.0] * EMBED_DIM


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings into 1024-dim float lists."""
    model: Any = get_bge_model()
    if model is None:
        return []
    try:
        from FlagEmbedding import BGEM3FlagModel
        if isinstance(model, BGEM3FlagModel):
            res: Any = model.encode(texts, batch_size=16, max_length=512)
            return cast(List[List[float]], res["dense_vecs"].tolist())
    except Exception:
        pass
    # SentenceTransformer fallback
    try:
        vecs: Any = model.encode(texts)
        if hasattr(vecs, "tolist"):
            return cast(List[List[float]], vecs.tolist())
        return [[float(x) for x in v] for v in vecs]
    except Exception:
        return []
