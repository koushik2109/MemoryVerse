"""
BGE-M3 Multilingual Text Embedder
Generates 1024-dimensional normalized dense text embeddings via BAAI/bge-m3.
"""
from typing import List, Union, Optional, Any
import numpy as np
from ai_engine.embeddings.base_embedder import BaseEmbedder
from ai_engine.models.bge_loader import embed_text, embed_batch, EMBED_DIM


class BgeEmbedder(BaseEmbedder):
    """1024-dim BAAI BGE-M3 dense text embedder wrapper."""

    def __init__(self):
        self.dim = EMBED_DIM

    def get_dimension(self) -> int:
        return self.dim

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            vec = embed_text(text)
            return np.array(vec, dtype=np.float32)
        vecs = embed_batch(list(text))
        return np.array(vecs, dtype=np.float32)
