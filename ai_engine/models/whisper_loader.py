"""
Whisper Model Loader
Singleton thread-safe loader for OpenAI Whisper / faster-whisper.
Used for:
  - Speech-to-text audio and video transcription
  - Voice Activity Detection (VAD)
  - Word-level timestamp extraction
"""
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WHISPER_MODEL_NAME = "base"
_whisper_model = None
_lock = threading.Lock()


def get_whisper_model():
    """Thread-safe singleton getter for Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        with _lock:
            if _whisper_model is None:
                try:
                    import whisper
                    logger.info(f"Loading Whisper model ({WHISPER_MODEL_NAME})...")
                    _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
                except ImportError:
                    try:
                        from faster_whisper import WhisperModel
                        logger.info(f"Loading faster-whisper model ({WHISPER_MODEL_NAME})...")
                        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
                    except Exception as e:
                        logger.warning(f"Failed to load Whisper/faster-whisper: {e}")
                        return None
    return _whisper_model


def transcribe_audio(file_path: str) -> str:
    """Transcribe speech in an audio or video file to text."""
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        if hasattr(model, "transcribe"):
            res = model.transcribe(file_path)
            if isinstance(res, dict):
                return str(res.get("text", "")).strip()
            # faster-whisper returns (segments, info)
            segments, _ = res
            return " ".join(s.text for s in segments).strip()
    except Exception as e:
        logger.warning(f"Whisper transcription failed for {file_path}: {e}")
    return ""
