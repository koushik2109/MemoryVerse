"""
Multimodal Processing Package
Exports Reciprocal Rank Fusion (RRF) and Cross-Modal Attention Aligners.
"""
from ai_engine.multimodal.fusion import reciprocal_rank_fusion, weighted_multimodal_fusion
from ai_engine.multimodal.cross_attention import (
    compute_cross_modal_alignment_matrix,
    align_audio_segments_to_keyframes,
)

__all__ = [
    "reciprocal_rank_fusion",
    "weighted_multimodal_fusion",
    "compute_cross_modal_alignment_matrix",
    "align_audio_segments_to_keyframes",
]
