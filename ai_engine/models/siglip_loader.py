"""
SigLIP Model Loader
Singleton thread-safe loader for Google SigLIP (google/siglip-base-patch16-224).
Output dimension: 768
Used for:
  - Video keyframe embedding & cross-modal text-image retrieval
"""
import logging
import threading
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM = 768

_siglip_processor = None
_siglip_model = None
_lock = threading.Lock()


def get_siglip_model():
    """Thread-safe singleton getter returning (processor, model)."""
    global _siglip_processor, _siglip_model
    if _siglip_model is None:
        with _lock:
            if _siglip_model is None:
                from transformers import AutoProcessor, AutoModel
                logger.info(f"Loading SigLIP model ({SIGLIP_MODEL_ID})...")
                _siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
                _siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
                _siglip_model.eval()
    return _siglip_processor, _siglip_model


def embed_frames(frames: List[np.ndarray]) -> List[List[float]]:
    """Embed list of BGR numpy frames into normalized 768-dim vectors."""
    processor, model = get_siglip_model()
    if processor is None or model is None or not frames:
        return []
    pil_images = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))

    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.vision_model(**{k: v for k, v in inputs.items() if "pixel" in k})
        embeds = outputs.last_hidden_state.mean(dim=1)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

    return embeds.cpu().numpy().tolist()


def embed_text_siglip(texts: List[str]) -> List[List[float]]:
    """Embed list of text strings using SigLIP text encoder into 768-dim vectors."""
    processor, model = get_siglip_model()
    if processor is None or model is None or not texts:
        return []
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.text_model(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        embeds = outputs.pooler_output
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

    return embeds.cpu().numpy().tolist()
