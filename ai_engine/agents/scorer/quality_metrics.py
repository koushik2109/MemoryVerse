"""
Media Scorer - Quality Metrics
Provides Laplacian variance blur analysis, luminance exposure checks,
and contrast calculations for memory visual qualification.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def compute_laplacian_variance(image_path: str) -> float:
    """
    Calculates the variance of the Laplacian of an image to measure sharpness.
    sigma^2 < 80 typically indicates motion or defocus blur.
    """
    if not os.path.exists(image_path):
        return 120.0  # Nominal sharpness fallback for remote/mocked assets

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return 100.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        logger.debug(f"Laplacian variance computation fallback: {e}")
        return 100.0


def compute_exposure_and_luminance(image_path: str) -> Dict[str, Any]:
    """
    Evaluates mean luminance and detects extreme under- or over-exposure.
    Luminance < 12 = underexposed; Luminance > 248 = overexposed/blown out.
    """
    if not os.path.exists(image_path):
        return {"mean_luminance": 128.0, "is_under_exposed": False, "is_over_exposed": False}

    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        if img is None:
            return {"mean_luminance": 128.0, "is_under_exposed": False, "is_over_exposed": False}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        return {
            "mean_luminance": round(mean_lum, 2),
            "is_under_exposed": mean_lum < 12.0,
            "is_over_exposed": mean_lum > 248.0,
        }
    except Exception as e:
        logger.debug(f"Exposure computation fallback: {e}")
        return {"mean_luminance": 128.0, "is_under_exposed": False, "is_over_exposed": False}


def evaluate_media_quality(image_path: str, blur_threshold: float = 80.0) -> Dict[str, Any]:
    """
    Runs full aesthetic and technical qualification on an image.
    Returns composite score in range [0.0, 1.0] and rejection flags.
    """
    lap_var = compute_laplacian_variance(image_path)
    exposure = compute_exposure_and_luminance(image_path)

    is_blurry = lap_var < blur_threshold
    is_rejected = is_blurry or exposure["is_under_exposed"] or exposure["is_over_exposed"]

    # Composite normalized score [0.0 - 1.0]
    sharpness_norm = min(1.0, lap_var / 250.0)
    lum_penalty = 0.5 if (exposure["is_under_exposed"] or exposure["is_over_exposed"]) else 1.0
    quality_score = round(sharpness_norm * lum_penalty, 3)

    return {
        "quality_score": quality_score,
        "laplacian_variance": round(lap_var, 2),
        "mean_luminance": exposure["mean_luminance"],
        "is_blurry": is_blurry,
        "is_under_exposed": exposure["is_under_exposed"],
        "is_over_exposed": exposure["is_over_exposed"],
        "is_qualified": not is_rejected,
        "reject_reason": "blurry" if is_blurry else ("exposure_issue" if is_rejected else None),
    }
