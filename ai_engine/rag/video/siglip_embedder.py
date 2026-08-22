"""
SigLIP Embedder
Uses google/siglip-base-patch16-224 (local, no API key) to:
  - embed_frames()  → image embeddings from BGR frames
  - embed_text()    → text embeddings (for cross-modal retrieval)

Average multiple frame embeddings to get a single video embedding.
Output dim: 768 (SigLIP base patch-16)
"""
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM = 768  # SigLIP base patch-16 output dim


@lru_cache(maxsize=1)
def _load_siglip():
    """Load SigLIP model + processor once and cache."""
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
    model.eval()
    return processor, model


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def embed_frames(frames: List[np.ndarray]) -> List[List[float]]:
    """
    Embed a list of BGR numpy frames.
    Returns a list of 768-dim float vectors.
    """
    processor, model = _load_siglip()
    pil_images = [_bgr_to_pil(f) for f in frames]

    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.vision_model(**{k: v for k, v in inputs.items() if "pixel" in k})
        # Pool: mean over patch tokens
        embeds = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

    return embeds.cpu().numpy().tolist()


def embed_text_siglip(texts: List[str]) -> List[List[float]]:
    """
    Embed text strings using SigLIP text encoder.
    Returns a list of 768-dim float vectors.
    """
    processor, model = _load_siglip()

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.text_model(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        embeds = outputs.pooler_output  # (B, 768)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

    return embeds.cpu().numpy().tolist()


def average_frame_embeddings(frame_embeds: List[List[float]]) -> List[float]:
    """
    Average a list of 768-dim frame embeddings into a single video embedding.
    L2-normalised after averaging.
    """
    arr = np.array(frame_embeds, dtype=np.float32)
    avg = arr.mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.tolist()


def select_most_distinct_keyframes(
    frame_embeds: List[List[float]],
    top_k: int = 3,
) -> List[int]:
    """
    Greedily pick the top_k most distinct frames using max-marginal-relevance
    (maximise minimum cosine distance to already-selected frames).

    Returns indices into the frame_embeds list.
    """
    if len(frame_embeds) <= top_k:
        return list(range(len(frame_embeds)))

    arr = np.array(frame_embeds, dtype=np.float32)
    selected: List[int] = []

    # Start with the frame farthest from the mean
    mean = arr.mean(axis=0)
    dists = 1 - (arr @ mean / (np.linalg.norm(arr, axis=1) * np.linalg.norm(mean) + 1e-8))
    selected.append(int(np.argmax(dists)))

    while len(selected) < top_k:
        selected_arr = arr[selected]  # (k, D)
        sims = arr @ selected_arr.T   # (N, k)
        # For each candidate, its max similarity to any already-selected frame
        max_sim_to_selected = sims.max(axis=1)
        # Penalise already-selected frames
        max_sim_to_selected[selected] = 1.0
        next_idx = int(np.argmin(max_sim_to_selected))
        selected.append(next_idx)

    return selected
