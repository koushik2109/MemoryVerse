"""
Unified Multimodal Ingestion Pipeline
Processes photos, videos, and audio files through their respective feature extractors and vectorizers.
"""
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MultimodalIngestPipeline:
    """
    Orchestrates the ingestion, feature extraction, and vectorization of user media assets.
    """

    def __init__(self):
        pass

    def ingest_image(self, image_path: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Processes a single image file through CLIP ViT-B (512-dim) and quality metrics."""
        from ai_engine.embeddings.clip_embedder import ClipEmbedder
        from ai_engine.agents.scorer.quality_metrics import evaluate_media_quality

        clip = ClipEmbedder()
        emb = clip.encode_image(image_path)
        quality = evaluate_media_quality(image_path)

        return {
            "user_id": user_id,
            "file_path": image_path,
            "modality": "image",
            "clip_embedding": emb.tolist() if emb is not None else [],
            "quality_metrics": quality,
            "status": "ingested",
        }

    def ingest_video(self, video_path: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Processes a video file through keyframe extraction and SigLIP (768-dim) embeddings."""
        from ai_engine.rag.video.keyframe_extractor import extract_keyframes
        from ai_engine.rag.video.siglip_embedder import embed_frames, select_distinct_keyframes

        keyframes, duration, fps, total_count = extract_keyframes(video_path, interval_sec=2.0)
        frames = [f for _, f in keyframes]
        embedded_frames = embed_frames(frames) if frames else []
        selected_frames = select_distinct_keyframes(embedded_frames, top_k=3) if embedded_frames else []

        return {
            "user_id": user_id,
            "file_path": video_path,
            "modality": "video",
            "total_keyframes": total_count,
            "selected_keyframes": len(selected_frames),
            "status": "ingested",
        }

    def ingest_audio(self, audio_path: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Processes an audio track through Librosa acoustics, Whisper STT, and BGE-M3 (1024-dim)."""
        from ai_engine.rag.audio.feature_extractor import extract_features
        from ai_engine.rag.audio.transcriber import transcribe
        from ai_engine.rag.audio.embedder import embed_text

        features = extract_features(audio_path)
        transcript = transcribe(audio_path)
        emb = embed_text(transcript) if transcript else []

        return {
            "user_id": user_id,
            "file_path": audio_path,
            "modality": "audio",
            "features": features.__dict__ if hasattr(features, "__dict__") else {},
            "transcript_embedding": emb,
            "status": "ingested",
        }
