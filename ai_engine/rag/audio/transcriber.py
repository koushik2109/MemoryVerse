"""
Whisper Transcriber
Uses openai/whisper-base (local) to transcribe speech audio to text.
"""
import whisper
import numpy as np
from functools import lru_cache
from ai_engine.rag.config import WHISPER_MODEL


@lru_cache(maxsize=1)
def _load_model():
    """Load Whisper model once and cache it in memory."""
    return whisper.load_model(WHISPER_MODEL)


def transcribe(file_path: str) -> str:
    """
    Transcribe an audio file using Whisper.
    Returns the transcribed text (empty string if nothing detected).
    """
    model = _load_model()
    result = model.transcribe(file_path, fp16=False, language=None)
    text = result.get("text", "").strip()
    return text
