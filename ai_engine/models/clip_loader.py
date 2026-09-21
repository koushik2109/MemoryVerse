"""
CLIP Model Loader
Singleton thread-safe loader for OpenAI CLIP ViT-B/32 via sentence-transformers.
Output dimension: 512
Used for:
  - 512-dim visual embeddings
  - Zero-shot tag classification (scenes, objects, people)
  - Candidate media scoring & near-duplicate burst detection
"""
import logging
import threading
from typing import Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

_clip_model = None
_lock = threading.Lock()
_cached_label_embeddings: Optional[Any] = None

CLIP_MODEL_NAME = "clip-ViT-B-32"
EMBED_DIM = 512


def get_clip_model():
    """Thread-safe singleton getter for the CLIP SentenceTransformer model."""
    global _clip_model
    if _clip_model is None:
        with _lock:
            if _clip_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading lightweight CLIP model ({CLIP_MODEL_NAME})...")
                    _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
                except Exception as e:
                    logger.warning(f"Failed to load CLIP model: {e}")
                    return None
    return _clip_model


def encode_image(image: Any) -> List[float]:
    """
    Encode a PIL Image or image array into a normalized 512-dim vector.
    """
    model = get_clip_model()
    if model is None:
        return [0.0] * EMBED_DIM
    try:
        emb = model.encode(image)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()
    except Exception as e:
        logger.warning(f"Error encoding image with CLIP: {e}")
        return [0.0] * EMBED_DIM


def encode_text(text: str) -> List[float]:
    """
    Encode a text query string into a normalized 512-dim vector.
    """
    model = get_clip_model()
    if model is None:
        return [0.0] * EMBED_DIM
    try:
        emb = model.encode(text)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()
    except Exception as e:
        logger.warning(f"Error encoding text with CLIP: {e}")
        return [0.0] * EMBED_DIM
