"""
ESC-50 Ambient Classifier
Uses the ESC-50 metadata to classify ambient audio using librosa MFCC
features + cosine similarity against a pre-built centroid lookup.

Supports all 50 ESC-50 categories, mapped to higher-level environments.
"""
import csv
import os
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
from sklearn.preprocessing import normalize

from ai_engine.rag.config import ESC50_META_PATH, RAG_DIR


# Higher-level environment groupings for ESC-50 categories
ESC50_ENV_MAP: Dict[str, str] = {
    # Animals
    "dog": "nature",
    "rooster": "nature",
    "pig": "nature",
    "cow": "nature",
    "frog": "nature",
    "cat": "nature",
    "hen": "nature",
    "insects": "nature",
    "sheep": "nature",
    "crow": "nature",
    # Natural
    "rain": "rain",
    "sea_waves": "nature",
    "crackling_fire": "indoor",
    "crickets": "nature",
    "chirping_birds": "nature",
    "water_drops": "nature",
    "wind": "nature",
    "pouring_water": "nature",
    "toilet_flush": "indoor",
    "thunderstorm": "rain",
    # Urban / Human
    "crying_baby": "crowd",
    "sneezing": "crowd",
    "clapping": "crowd",
    "breathing": "crowd",
    "coughing": "crowd",
    "footsteps": "crowd",
    "laughing": "crowd",
    "brushing_teeth": "indoor",
    "snoring": "indoor",
    "drinking_sipping": "indoor",
    # Domestic
    "door_wood_knock": "indoor",
    "mouse_click": "indoor",
    "keyboard_typing": "indoor",
    "door_wood_creaks": "indoor",
    "can_opening": "indoor",
    "washing_machine": "indoor",
    "vacuum_cleaner": "indoor",
    "clock_alarm": "indoor",
    "clock_tick": "indoor",
    "glass_breaking": "indoor",
    # Exterior / transport
    "helicopter": "transport",
    "chainsaw": "outdoor",
    "siren": "transport",
    "car_horn": "transport",
    "engine": "transport",
    "train": "transport",
    "church_bells": "outdoor",
    "airplane": "transport",
    "fireworks": "outdoor",
    "hand_saw": "outdoor",
}

_MFCC_N = 40
_CENTROID_CACHE = RAG_DIR / ".esc50_centroids.pkl"


def _extract_mfcc(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=5.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_MFCC_N)
    return np.mean(mfcc, axis=1)


def _build_centroids() -> Dict[str, np.ndarray]:
    """
    Build per-category MFCC centroids from the ESC-50 audio folder.
    Cached to disk so it only runs once.
    """
    audio_dir = ESC50_META_PATH.parent.parent / "audio"
    centroids: Dict[str, list] = defaultdict(list)

    with open(ESC50_META_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = audio_dir / row["filename"]
            if not path.exists():
                continue
            try:
                feat = _extract_mfcc(str(path))
                centroids[row["category"]].append(feat)
            except Exception:
                pass

    result: Dict[str, np.ndarray] = {}
    for cat, feats in centroids.items():
        result[cat] = np.mean(feats, axis=0)
    return result


@lru_cache(maxsize=1)
def _get_centroids() -> Dict[str, np.ndarray]:
    if _CENTROID_CACHE.exists():
        with open(_CENTROID_CACHE, "rb") as f:
            return pickle.load(f)
    centroids = _build_centroids()
    with open(_CENTROID_CACHE, "wb") as f:
        pickle.dump(centroids, f)
    return centroids


def classify_ambient(file_path: str) -> Tuple[str, str]:
    """
    Returns (esc50_category, environment_label) for the given audio file.
    Uses cosine similarity against precomputed ESC-50 category centroids.
    """
    centroids = _get_centroids()
    if not centroids:
        return "unknown", "unknown"

    query_feat = _extract_mfcc(file_path).reshape(1, -1)
    query_feat = normalize(query_feat)[0]

    best_cat, best_sim = "unknown", -1.0
    for cat, centroid in centroids.items():
        c = normalize(centroid.reshape(1, -1))[0]
        sim = float(np.dot(query_feat, c))
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    env_label = ESC50_ENV_MAP.get(best_cat, "outdoor")
    return best_cat, env_label
