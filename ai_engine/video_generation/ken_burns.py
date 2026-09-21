"""
Ken Burns Camera Interpolator
Computes continuous 30fps pan-and-zoom crop windows over static images.
"""
from typing import Tuple, Dict, Any


def get_crop_window(
    width: int,
    height: int,
    start_scale: float,
    end_scale: float,
    start_center: Tuple[float, float],
    end_center: Tuple[float, float],
    progress: float,  # 0.0 to 1.0
) -> Tuple[int, int, int, int]:
    """
    Computes (x1, y1, x2, y2) bounding box on the original image at progress t.
    """
    t = max(0.0, min(1.0, progress))
    # Smooth cubic easing
    eased_t = t * t * (3.0 - 2.0 * t)

    scale = start_scale + (end_scale - start_scale) * eased_t
    cx = start_center[0] + (end_center[0] - start_center[0]) * eased_t
    cy = start_center[1] + (end_center[1] - start_center[1]) * eased_t

    crop_w = width / scale
    crop_h = height / scale

    pixel_cx = cx * width
    pixel_cy = cy * height

    x1 = max(0, int(pixel_cx - crop_w / 2.0))
    y1 = max(0, int(pixel_cy - crop_h / 2.0))
    x2 = min(width, int(x1 + crop_w))
    y2 = min(height, int(y1 + crop_h))

    return x1, y1, x2, y2
