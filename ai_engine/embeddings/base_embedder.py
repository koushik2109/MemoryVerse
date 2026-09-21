"""
Base Multimodal Embedder Interface
Defines the standard abstract contract for image, text, and audio embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all neural embedding encoders in MemoryVerse."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the output vector dimensionality (e.g. 512, 768, 1024)."""
        pass

    @abstractmethod
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encodes one or more text strings into normalized vector embeddings."""
        pass

    def encode_image(self, image_or_path: Any) -> Optional[np.ndarray]:
        """Encodes an image (PIL Image or filepath) into a normalized vector embedding."""
        raise NotImplementedError("Image encoding not supported by this embedder.")
