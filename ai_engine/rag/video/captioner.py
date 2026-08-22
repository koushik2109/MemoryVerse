"""
Qwen3-VL Video Captioner
Generates natural-language captions for the top-3 most distinct keyframes.
Uses Ollama to run qwen2.5vl:7b locally (vision-language model).

Ollama supports multimodal models via base64-encoded image payloads.
No API key required - runs fully locally.
"""
import base64
import io
import re
from typing import List

import cv2
import httpx
import numpy as np
from PIL import Image

from ai_engine.rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Qwen3-VL model tag on Ollama (pull with: ollama pull qwen2.5vl:7b)
QWEN_VL_MODEL = "qwen2.5vl:7b"

_CAPTION_SYSTEM = (
    "You are a helpful AI for captioning memory photos. "
    "Given a keyframe from a personal video, write ONE concise sentence (max 20 words) "
    "describing what is happening. Focus on people, activities, setting, and mood. "
    "Do NOT mention video quality or technical details."
)


def _frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert BGR frame to base64-encoded JPEG string."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    # Resize to keep within max_size while preserving aspect ratio
    pil.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _caption_single_frame(b64_img: str, frame_index: int) -> str:
    """Ask Ollama (qwen2.5vl) to caption one frame."""
    payload = {
        "model": QWEN_VL_MODEL,
        "messages": [
            {"role": "system", "content": _CAPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                    {
                        "type": "text",
                        "text": f"Caption this keyframe (frame #{frame_index + 1}):",
                    },
                ],
            },
        ],
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        # Strip markdown / quotes if model wraps output
        raw = re.sub(r'^["\']|["\']$', "", raw)
        return raw
    except Exception as e:
        return f"Keyframe {frame_index + 1} from personal video."


def caption_keyframes(
    frames: List[np.ndarray],
    distinct_indices: List[int],
) -> str:
    """
    Generate captions for the top-3 most distinct keyframes
    and concatenate them into a single descriptive passage.

    Args:
        frames:           list of BGR frames (all keyframes)
        distinct_indices: indices of the top-3 most distinct frames

    Returns:
        A single concatenated caption string for embedding.
    """
    captions: List[str] = []
    for idx, fi in enumerate(distinct_indices):
        if fi >= len(frames):
            continue
        b64 = _frame_to_base64(frames[fi])
        caption = _caption_single_frame(b64, idx)
        captions.append(caption)

    return " ".join(captions)
