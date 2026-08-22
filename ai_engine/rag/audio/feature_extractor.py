"""
Audio Feature Extractor
Uses librosa to extract acoustic features and classify audio type.
"""
import os
from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np


AudioType = Literal["speech", "music", "ambient", "mixed"]

MOOD_MAP = {
    "high_energy_fast": "energetic",
    "high_energy_slow": "intense",
    "low_energy_fast": "light",
    "low_energy_slow": "calm",
}


@dataclass
class AudioFeatures:
    duration: float
    tempo: float
    energy: float                    # RMS energy (normalised 0-1)
    zero_crossing_rate: float        # mean ZCR
    spectral_centroid: float         # mean spectral centroid (Hz)
    audio_type: AudioType
    mood_label: str


def _normalise_energy(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def _classify_audio_type(
    zcr: float,
    energy: float,
    spectral_centroid: float,
    tempo: float,
) -> AudioType:
    """
    Rule-based classifier:
    - Speech:   moderate ZCR, moderate-low energy, mid spectral centroid
    - Music:    clear tempo, higher energy, wider spectral centroid
    - Ambient:  low ZCR, low energy, variable spectral centroid
    - Mixed:    falls between speech and music thresholds
    """
    is_high_zcr = zcr > 0.08
    is_high_energy = energy > 0.05
    is_high_centroid = spectral_centroid > 3000
    has_clear_tempo = tempo > 60

    speech_score = (not is_high_energy) + is_high_zcr + (not is_high_centroid)
    music_score = has_clear_tempo + is_high_energy + is_high_centroid
    ambient_score = (not is_high_zcr) + (not is_high_energy)

    scores = {
        "speech": speech_score,
        "music": music_score,
        "ambient": ambient_score,
    }
    best = max(scores, key=lambda k: scores[k])

    # If top two scores are equal → mixed
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] == sorted_scores[1] and sorted_scores[0] > 0:
        return "mixed"

    return best  # type: ignore[return-value]


def _mood_label(energy: float, tempo: float) -> str:
    energy_tag = "high_energy" if energy > 0.05 else "low_energy"
    tempo_tag = "fast" if tempo > 100 else "slow"
    return MOOD_MAP.get(f"{energy_tag}_{tempo_tag}", "neutral")


def extract_features(file_path: str) -> AudioFeatures:
    """
    Load audio file and extract all features.
    Returns an AudioFeatures dataclass.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])

    # Energy (RMS)
    energy = _normalise_energy(y)

    # Zero-crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))

    # Spectral centroid
    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    )

    audio_type = _classify_audio_type(zcr, energy, centroid, tempo)
    mood = _mood_label(energy, tempo)

    return AudioFeatures(
        duration=duration,
        tempo=tempo,
        energy=energy,
        zero_crossing_rate=zcr,
        spectral_centroid=centroid,
        audio_type=audio_type,
        mood_label=mood,
    )
