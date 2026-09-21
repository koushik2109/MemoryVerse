"""
CLIP Multimodal Embedder
Generates 512-dimensional normalized vision and text embeddings via OpenAI CLIP ViT-B/32.
"""
from typing import List, Union, Optional, Any
import numpy as np
from ai_engine.embeddings.base_embedder import BaseEmbedder
from ai_engine.models.clip_loader import get_clip_model, encode_image, encode_text, EMBED_DIM


class ClipEmbedder(BaseEmbedder):
    """512-dim CLIP ViT-B/32 multimodal embedder wrapper."""

    def __init__(self):
        self.dim = EMBED_DIM

    def get_dimension(self) -> int:
        return self.dim

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            vec = encode_text(text)
            return np.array(vec, dtype=np.float32)
        vectors = [encode_text(t) for t in text]
        return np.array(vectors, dtype=np.float32)

    def encode_image(self, image_or_path: Any) -> Optional[np.ndarray]:
        if isinstance(image_or_path, str):
            from PIL import Image
            img = Image.open(image_or_path).convert("RGB")
        else:
            img = image_or_path
        vec = encode_image(img)
        return np.array(vec, dtype=np.float32)
