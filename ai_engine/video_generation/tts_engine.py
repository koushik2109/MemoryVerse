"""
Emotion-Aware Neural Text-to-Speech (TTS) Engine for MemoryVerse.
Synthesizes expressive, emotive speech narration for video scenes, memory recaps,
and interactive AI assistant voiceovers using Microsoft Neural Voices (edge-tts)
with parametric voice modulation and procedural harmonic acoustic fallbacks.
"""

import asyncio
import io
import logging
import math
import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing edge-tts; graceful degradation if not installed
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("edge-tts not installed; EmotionTTSEngine will use procedural acoustic fallback.")

# ── Emotion Configuration Presets ───────────────────────────────────────────
# Maps emotional mood to voice identity, pitch modulation, cadence rate, and volume.
EMOTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "calm": {
        "voice": "en-US-JennyNeural",
        "rate": "-6%",
        "pitch": "-2Hz",
        "volume": "+0%",
        "description": "Warm, measured, relaxed personal journal tone",
    },
    "serene": {
        "voice": "en-US-JennyNeural",
        "rate": "-8%",
        "pitch": "-3Hz",
        "volume": "+0%",
        "description": "Peaceful, meditative, gentle cadence",
    },
    "nostalgic": {
        "voice": "en-US-GuyNeural",
        "rate": "-8%",
        "pitch": "-4Hz",
        "volume": "+0%",
        "description": "Reflective, sentimental, evocative storytelling",
    },
    "reflective": {
        "voice": "en-US-GuyNeural",
        "rate": "-7%",
        "pitch": "-3Hz",
        "volume": "+0%",
        "description": "Contemplative, warm memory reflection",
    },
    "energetic": {
        "voice": "en-US-AriaNeural",
        "rate": "+10%",
        "pitch": "+6Hz",
        "volume": "+0%",
        "description": "Vibrant, upbeat, dynamic rhythm",
    },
    "excited": {
        "voice": "en-US-AriaNeural",
        "rate": "+12%",
        "pitch": "+7Hz",
        "volume": "+0%",
        "description": "Enthusiastic, rapid, high-energy cadence",
    },
    "joyful": {
        "voice": "en-US-AriaNeural",
        "rate": "+5%",
        "pitch": "+4Hz",
        "volume": "+0%",
        "description": "Bright, happy, uplifting tone",
    },
    "happy": {
        "voice": "en-US-AriaNeural",
        "rate": "+5%",
        "pitch": "+4Hz",
        "volume": "+0%",
        "description": "Cheerful, warm personal joy",
    },
    "celebratory": {
        "voice": "en-US-AriaNeural",
        "rate": "+8%",
        "pitch": "+5Hz",
        "volume": "+0%",
        "description": "Festive, triumphant party celebration",
    },
    "dramatic": {
        "voice": "en-US-ChristopherNeural",
        "rate": "-5%",
        "pitch": "-6Hz",
        "volume": "+0%",
        "description": "Deep, resonant, cinematic impact",
    },
    "cinematic": {
        "voice": "en-US-ChristopherNeural",
        "rate": "-6%",
        "pitch": "-5Hz",
        "volume": "+0%",
        "description": "Broad, theatrical documentary narration",
    },
    "neutral": {
        "voice": "en-US-AriaNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Clear, balanced narrative voice",
    },
}


class EmotionTTSEngine:
    """
    Emotion-Aware Text-to-Speech Engine.
    Synthesizes speech with pitch, rate, and voice modulation tailored to emotional context.
    """

    def __init__(self, default_emotion: str = "calm", sample_rate: int = 44100):
        self.default_emotion = default_emotion.lower()
        self.sample_rate = sample_rate

    @staticmethod
    def detect_emotion(text: str) -> str:
        """
        Analyzes narrative text to infer the dominant emotional tone.
        """
        if not text or not text.strip():
            return "calm"

        lower = text.lower()

        # Keywords for celebratory / energetic
        if any(w in lower for w in [
            "party", "dance", "dancing", "celebrat", "cheer", "excit", "fun",
            "jump", "win", "festival", "carnival", "firework", "shout", "wild"
        ]):
            return "celebratory"

        # Keywords for joyful / happy
        if any(w in lower for w in [
            "laugh", "laughing", "smile", "smiling", "happy", "joy", "delight",
            "love", "warmth", "giggle", "sunshine", "bliss", "proud"
        ]):
            return "joyful"

        # Keywords for nostalgic / reflective
        if any(w in lower for w in [
            "remember", "years ago", "childhood", "past", "memory", "memories",
            "used to", "look back", "nostalg", "forever", "sweet old", "reminisce"
        ]):
            return "nostalgic"

        # Keywords for dramatic / cinematic
        if any(w in lower for w in [
            "journey", "climb", "summit", "mountain", "challenge", "storm",
            "monumental", "breath", "intense", "epic", "battle", "endure"
        ]):
            return "dramatic"

        # Keywords for calm / serene
        if any(w in lower for w in [
            "quiet", "sunset", "sunrise", "peace", "peaceful", "gentle", "calm",
            "serene", "nature", "beach", "ocean", "sea", "breeze", "morning", "lake"
        ]):
            return "serene"

        return "calm"

    def get_profile(self, emotion: str) -> Dict[str, Any]:
        """Returns the profile config for the requested emotion."""
        e = emotion.lower().strip()
        if e in EMOTION_PROFILES:
            return EMOTION_PROFILES[e]
        # Partial match
        for key in EMOTION_PROFILES:
            if key in e or e in key:
                return EMOTION_PROFILES[key]
        return EMOTION_PROFILES["calm"]

    async def synthesize(
        self,
        text: str,
        emotion: str = "auto",
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Asynchronously synthesizes speech for the given text.

        Returns:
            (audio_array: np.ndarray [shape (N, 2), float32], duration_seconds: float, sample_rate: int)
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros((1, 2), dtype=np.float32), 0.0, self.sample_rate

        # Resolve emotion
        if emotion == "auto" or not emotion:
            resolved_emotion = self.detect_emotion(clean_text)
        else:
            resolved_emotion = emotion.lower()

        profile = self.get_profile(resolved_emotion)
        selected_voice = voice or profile["voice"]
        rate = profile["rate"]
        pitch = profile["pitch"]
        volume = profile["volume"]

        logger.info(
            f"Synthesizing speech: emotion='{resolved_emotion}', voice='{selected_voice}', "
            f"rate='{rate}', pitch='{pitch}'"
        )

        # Temporary MP3 file for edge-tts output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_path = tmp_mp3.name

        try:
            if HAS_EDGE_TTS:
                try:
                    communicate = edge_tts.Communicate(
                        text=clean_text,
                        voice=selected_voice,
                        rate=rate,
                        pitch=pitch,
                        volume=volume,
                    )
                    await communicate.save(tmp_path)

                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 500:
                        # Decode MP3 to numpy array using moviepy / scipy
                        audio_arr, dur = self._decode_mp3(tmp_path)
                        if output_path:
                            # Save to desired output path
                            with open(tmp_path, "rb") as src, open(output_path, "wb") as dst:
                                dst.write(src.read())
                        return audio_arr, dur, self.sample_rate
                except Exception as net_err:
                    logger.warning(f"edge-tts network synthesis failed: {net_err}. Using procedural acoustic fallback.")

            # Fallback to harmonic procedural speech tone
            audio_arr, dur = self._synthesize_fallback(clean_text, resolved_emotion)
            if output_path:
                from ai_engine.video_generation.audio_synth import write_wav_file
                write_wav_file(output_path, audio_arr[:, 0], self.sample_rate)
            return audio_arr, dur, self.sample_rate

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def synthesize_sync(
        self,
        text: str,
        emotion: str = "auto",
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Synchronous wrapper for synthesize.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In an already running event loop (e.g. FastAPI / Jupyter)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.synthesize(text, emotion=emotion, voice=voice, output_path=output_path)
                    ).result()
            else:
                return loop.run_until_complete(
                    self.synthesize(text, emotion=emotion, voice=voice, output_path=output_path)
                )
        except Exception:
            return asyncio.run(
                self.synthesize(text, emotion=emotion, voice=voice, output_path=output_path)
            )

    def _decode_mp3(self, mp3_path: str) -> Tuple[np.ndarray, float]:
        """
        Decodes an MP3 file into a stereo float32 NumPy array at self.sample_rate.
        """
        try:
            from moviepy import AudioFileClip
            with AudioFileClip(mp3_path) as clip:
                dur = float(clip.duration or 0.0)
                sound = clip.to_soundarray(fps=self.sample_rate)
                if sound.ndim == 1:
                    sound = np.column_stack([sound, sound])
                elif sound.shape[1] == 1:
                    sound = np.column_stack([sound[:, 0], sound[:, 0]])
                return sound.astype(np.float32), dur
        except Exception as e:
            logger.warning(f"AudioFileClip decode failed: {e}. Trying ffmpeg pipe.")
            return self._decode_via_ffmpeg(mp3_path)

    def _decode_via_ffmpeg(self, mp3_path: str) -> Tuple[np.ndarray, float]:
        """FFmpeg pipe fallback for MP3 decoding."""
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-i", mp3_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "2",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        stereo = raw.reshape((-1, 2))
        dur = len(stereo) / self.sample_rate
        return stereo, dur

    def _synthesize_fallback(self, text: str, emotion: str) -> Tuple[np.ndarray, float]:
        """
        Procedural acoustic narration fallback:
        Synthesizes speech-like melodic formant cadence with warm harmonic envelopes.
        Used if network access to Edge-TTS is restricted.
        """
        words = [w for w in re.split(r"\s+", text) if w]
        word_count = max(1, len(words))
        # Average reading rate: ~2.8 words per second
        est_duration = max(1.8, word_count / 2.8)

        total_samples = int(est_duration * self.sample_rate)
        t = np.linspace(0, est_duration, total_samples, endpoint=False)
        audio = np.zeros((total_samples, 2), dtype=np.float32)

        # Base fundamental pitch according to emotion
        base_freq = 196.0 if "dramatic" in emotion else (220.0 if "nostalgic" in emotion else 246.94)
        if "energetic" in emotion or "joyful" in emotion:
            base_freq = 277.18

        # Generate syllables cadence
        syllable_dur = est_duration / (word_count * 1.5)
        for i, time_val in enumerate(t):
            syllable_phase = (time_val % syllable_dur) / syllable_dur
            env = math.sin(math.pi * syllable_phase) if syllable_phase < 0.9 else 0.0

            # Pitch variation
            pitch_mod = 1.0 + 0.08 * math.sin(2 * math.pi * 3.5 * time_val)
            f = base_freq * pitch_mod

            sig = 0.5 * math.sin(2 * math.pi * f * time_val)
            sig += 0.25 * math.sin(2 * math.pi * (f * 2) * time_val)
            sig += 0.15 * math.sin(2 * math.pi * (f * 3) * time_val)

            val = sig * env * 0.22
            audio[i, 0] = val
            audio[i, 1] = val

        return audio, est_duration
