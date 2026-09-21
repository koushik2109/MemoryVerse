"""
Video Transitions Engine
Computes cross-dissolve opacity blending, dip-to-black luminance curves,
and slide / whip-pan motion parameters.
"""
from typing import Tuple, Dict, Any
import numpy as np


def blend_crossfade(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,  # 0.0 (all frame_a) to 1.0 (all frame_b)
) -> np.ndarray:
    """Blends two video frames linearly using alpha crossfade."""
    alpha = max(0.0, min(1.0, progress))
    blended = (1.0 - alpha) * frame_a.astype(np.float32) + alpha * frame_b.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_dip_to_black(
    frame: np.ndarray,
    progress: float,  # 0.0 to 1.0
) -> np.ndarray:
    """
    Fades frame to black at progress=0.5, then emerges into next image at progress=1.0.
    """
    p = max(0.0, min(1.0, progress))
    factor = abs(2.0 * p - 1.0)  # 1.0 -> 0.0 -> 1.0
    faded = frame.astype(np.float32) * factor
    return np.clip(faded, 0, 255).astype(np.uint8)
