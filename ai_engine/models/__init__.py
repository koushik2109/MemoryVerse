"""
MemoryVerse AI Engine - Models Package
Centralized singleton model loaders for multimodal embeddings and inference.
"""
from ai_engine.models.clip_loader import get_clip_model, encode_image as clip_encode_image, encode_text as clip_encode_text
from ai_engine.models.siglip_loader import get_siglip_model, embed_frames as siglip_embed_frames, embed_text_siglip
from ai_engine.models.bge_loader import get_bge_model, embed_text as bge_embed_text, embed_batch as bge_embed_batch
from ai_engine.models.whisper_loader import get_whisper_model, transcribe_audio as whisper_transcribe

__all__ = [
    "get_clip_model",
    "clip_encode_image",
    "clip_encode_text",
    "get_siglip_model",
    "siglip_embed_frames",
    "embed_text_siglip",
    "get_bge_model",
    "bge_embed_text",
    "bge_embed_batch",
    "get_whisper_model",
    "whisper_transcribe",
]
