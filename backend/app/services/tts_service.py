"""
TTS Service Layer for MemoryVerse Backend.
Provides high-level APIs for synthesizing speech from text, auto-narrating memories,
and streaming audio responses for the frontend.
"""

import io
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

from ai_engine.video_generation.tts_engine import EmotionTTSEngine, EMOTION_PROFILES
from app.core.db import get_supabase_client

logger = logging.getLogger(__name__)

# Global singleton TTS engine
_tts_engine: Optional[EmotionTTSEngine] = None


def get_tts_engine() -> EmotionTTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = EmotionTTSEngine()
    return _tts_engine


class TTSService:
    @staticmethod
    async def synthesize_speech(
        text: str,
        emotion: str = "auto",
        voice: Optional[str] = None,
    ) -> Tuple[bytes, str, float]:
        """
        Synthesizes speech audio bytes and returns (audio_bytes, mime_type, duration).
        """
        engine = get_tts_engine()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            audio_arr, duration, sr = await engine.synthesize(
                text=text,
                emotion=emotion,
                voice=voice,
                output_path=tmp_path,
            )

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return data, "audio/mpeg", duration
            else:
                # Encode fallback wav bytes
                import wave
                buf = io.BytesIO()
                int16_data = (audio_arr[:, 0] * 32767).astype("int16")
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(int16_data.tobytes())
                return buf.getvalue(), "audio/wav", duration

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    @staticmethod
    async def narrate_memory(
        memory_id: str,
        user_id: str,
        mood: Optional[str] = None,
    ) -> Tuple[bytes, str, str, float]:
        """
        Builds a comprehensive emotional spoken narration for a memory and returns audio bytes.
        Returns (audio_bytes, mime_type, narration_script, duration).
        """
        supabase = get_supabase_client()

        # 1. Fetch memory
        mem_res = supabase.table("memories").select("*").eq("id", memory_id).execute()
        if not mem_res.data:
            raise ValueError("Memory not found")

        mem = mem_res.data[0]
        title = mem.get("title") or "Our Memory"
        desc = mem.get("description") or ""
        date_str = mem.get("memory_date") or mem.get("event_date") or ""
        location = mem.get("location_name") or ""

        # 2. Fetch media items for context
        media_res = supabase.table("media")\
            .select("filename, description, scenes, objects")\
            .eq("memory_id", memory_id)\
            .limit(10)\
            .execute()

        media_items = media_res.data or []

        # 3. Construct rich narrative script
        script_parts = [f"This is the memory of {title}."]
        if date_str:
            script_parts.append(f"Captured on {date_str[:10]}.")
        if location:
            script_parts.append(f"At {location}.")
        if desc:
            script_parts.append(desc)

        # Add visual context if available
        features = []
        for m in media_items[:4]:
            if m.get("description"):
                features.append(m["description"])
        if features:
            script_parts.append(" ".join(features))

        full_script = " ".join(script_parts)

        # 4. Synthesize with emotion
        engine = get_tts_engine()
        detected_emotion = mood or engine.detect_emotion(full_script)

        audio_bytes, mime, dur = await TTSService.synthesize_speech(
            text=full_script,
            emotion=detected_emotion,
        )

        return audio_bytes, mime, full_script, dur
