"""
SigLIP Multimodal Embedder
Generates 768-dimensional normalized vision and text embeddings via Google SigLIP.
"""
from typing import List, Union, Optional, Any
import numpy as np
from ai_engine.embeddings.base_embedder import BaseEmbedder
from ai_engine.models.siglip_loader import embed_frames, embed_text_siglip, EMBED_DIM


class SiglipEmbedder(BaseEmbedder):
    """768-dim Google SigLIP multimodal embedder wrapper."""

    def __init__(self):
        self.dim = EMBED_DIM

    def get_dimension(self) -> int:
        return self.dim

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        texts = [text] if isinstance(text, str) else list(text)
        vecs = embed_text_siglip(texts)
        return np.array(vecs[0] if isinstance(text, str) else vecs, dtype=np.float32)

    def encode_image(self, image_or_path: Any) -> Optional[np.ndarray]:
        import cv2
        if isinstance(image_or_path, str):
            bgr = cv2.imread(image_or_path)
            if bgr is None:
                return np.zeros((self.dim,), dtype=np.float32)
            frames = [bgr]
        elif isinstance(image_or_path, np.ndarray):
            frames = [image_or_path]
        else:
            # PIL image
            rgb = np.array(image_or_path)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames = [bgr]
        vecs = embed_frames(frames)
        return np.array(vecs[0], dtype=np.float32)
