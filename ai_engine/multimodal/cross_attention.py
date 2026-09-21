"""
Cross-Modal Attention & Alignment
Aligns visual keyframe timeline timestamps with audio speech transcription timestamps
and acoustic energy envelopes.
"""
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_cross_modal_alignment_matrix(
    visual_embeddings: List[List[float]],
    audio_embeddings: List[List[float]],
) -> np.ndarray:
    """
    Computes cosine affinity / attention matrix between visual frames and audio events:
        Attention(V, A) = softmax((V * A^T) / sqrt(d))
    """
    if not visual_embeddings or not audio_embeddings:
        return np.zeros((len(visual_embeddings), len(audio_embeddings)), dtype=np.float32)

    v = np.array(visual_embeddings, dtype=np.float32)
    a = np.array(audio_embeddings, dtype=np.float32)

    # Normalize vectors
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm[v_norm == 0] = 1.0
    v = v / v_norm

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    a = a / a_norm

    # Dot product similarity
    affinity = np.dot(v, a.T)
    d = v.shape[-1]
    scaled_affinity = affinity / np.sqrt(max(1, d))

    # Softmax along audio dimension
    exp_aff = np.exp(scaled_affinity - np.max(scaled_affinity, axis=-1, keepdims=True))
    attention = exp_aff / np.sum(exp_aff, axis=-1, keepdims=True)
    return attention


def align_audio_segments_to_keyframes(
    keyframes: List[Dict[str, Any]],
    audio_segments: List[Dict[str, Any]],
    tolerance_seconds: float = 2.5,
) -> List[Dict[str, Any]]:
    """
    Temporally links each keyframe to the nearest co-occurring audio transcript or acoustic cue.
    """
    aligned_keyframes = []

    for kf in keyframes:
        kf_time = float(kf.get("timestamp_sec", kf.get("timestamp", 0.0)) or 0.0)
        best_segment = None
        min_distance = float("inf")

        for seg in audio_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 2.0))
            # Distance from keyframe to segment window
            if seg_start <= kf_time <= seg_end:
                dist = 0.0
            else:
                dist = min(abs(kf_time - seg_start), abs(kf_time - seg_end))

            if dist < min_distance:
                min_distance = dist
                best_segment = seg

        kf_entry = dict(kf)
        if best_segment and min_distance <= tolerance_seconds:
            kf_entry["aligned_audio_segment"] = best_segment
            kf_entry["audio_alignment_offset"] = round(min_distance, 2)
        else:
            kf_entry["aligned_audio_segment"] = None
            kf_entry["audio_alignment_offset"] = None

        aligned_keyframes.append(kf_entry)

    return aligned_keyframes
