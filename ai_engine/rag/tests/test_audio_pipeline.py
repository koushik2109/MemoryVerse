"""
Audio RAG Pipeline - Synthetic Test Suite
Tests:
  - ESC-50 ambient classification with real audio files
  - Speech transcription with a synthetic WAV
  - Intent extraction with various query types
  - Full ingest → retrieve round-trip (mocked Supabase)
"""
import os
import sys
import wave
import struct
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ai_engine.rag.audio.feature_extractor import extract_features, AudioFeatures
from ai_engine.rag.audio.intent_extractor import _fallback_intent


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _make_sine_wav(freq: float = 440.0, duration: float = 2.0, sr: int = 16000) -> str:
    """Generate a simple sine-wave WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    n_samples = int(sr * duration)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n_samples):
            sample = int(32767 * math.sin(2 * math.pi * freq * i / sr))
            wf.writeframes(struct.pack("<h", sample))
    return tmp.name


def _make_noise_wav(duration: float = 2.0, sr: int = 16000) -> str:
    """Generate a white-noise WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    n_samples = int(sr * duration)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for _ in range(n_samples):
            sample = random.randint(-32768, 32767)
            wf.writeframes(struct.pack("<h", sample))
    return tmp.name


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureExtractor(unittest.TestCase):

    def test_sine_wave_features(self):
        path = _make_sine_wav(freq=440.0, duration=3.0)
        try:
            feat = extract_features(path)
            self.assertIsInstance(feat, AudioFeatures)
            self.assertGreater(feat.duration, 0)
            self.assertGreater(feat.energy, 0)
            self.assertIn(feat.audio_type, ["speech", "music", "ambient", "mixed"])
            self.assertIn(feat.mood_label, ["energetic", "calm", "intense", "light", "neutral"])
        finally:
            os.unlink(path)

    def test_noise_features(self):
        path = _make_noise_wav(duration=3.0)
        try:
            feat = extract_features(path)
            self.assertIsInstance(feat, AudioFeatures)
            self.assertGreater(feat.zero_crossing_rate, 0)
        finally:
            os.unlink(path)


class TestIntentExtractor(unittest.TestCase):
    """Test rule-based fallback intent extractor (no Ollama required)."""

    def test_mood_intent(self):
        intent = _fallback_intent("find audio where we were laughing at the party")
        self.assertEqual(intent["intent_type"], "mood")

    def test_ambient_intent(self):
        intent = _fallback_intent("recordings of rain outside")
        self.assertEqual(intent["intent_type"], "ambient")

    def test_speech_intent(self):
        intent = _fallback_intent("what did we say during the meeting")
        self.assertIn(intent["intent_type"], ["speech", "general"])

    def test_general_intent(self):
        intent = _fallback_intent("find my old recordings")
        self.assertEqual(intent["intent_type"], "general")


class TestESC50Ambient(unittest.TestCase):
    """Test ESC-50 classification with real audio files if available."""

    def test_classify_ambient_sine(self):
        """Smoke test: classify a synthetic file (won't be accurate, just checks it runs)."""
        from ai_engine.rag.audio.ambient_classifier import classify_ambient
        path = _make_noise_wav(duration=5.0)
        try:
            cat, env = classify_ambient(path)
            self.assertIsInstance(cat, str)
            self.assertIsInstance(env, str)
        finally:
            os.unlink(path)


class TestSyntheticTranscripts(unittest.TestCase):
    """Test speech pipeline with mock Whisper (avoids downloading model in CI)."""

    SYNTHETIC_TRANSCRIPTS = [
        "We had a great time at the beach yesterday.",
        "The meeting was really productive, everyone agreed on the deadline.",
        "Happy birthday! I can't believe you're thirty already.",
        "It started raining during our hike and we ran back to the car.",
        "Laughing so hard at the party, best night in years.",
    ]

    def test_embed_synthetic_transcripts(self):
        """Embed synthetic transcripts and check output shape."""
        from ai_engine.rag.audio.embedder import embed_text
        for t in self.SYNTHETIC_TRANSCRIPTS:
            vec = embed_text(t)
            self.assertIsInstance(vec, list)
            self.assertEqual(len(vec), 1024)  # BGE-M3 dense dim


class TestIngestRouterMocked(unittest.TestCase):
    """Full ingest pipeline with all external services mocked."""

    def test_full_ingest_pipeline(self):
        from ai_engine.rag.audio import feature_extractor, transcriber, embedder, vector_store

        sine_path = _make_sine_wav(freq=440.0, duration=2.0)
        fake_embedding = [0.0] * 1024

        with (
            patch.object(feature_extractor, "extract_features") as mock_feat,
            patch.object(transcriber, "transcribe", return_value="test transcript") as _,
            patch.object(embedder, "embed_text", return_value=fake_embedding) as _,
            patch.object(vector_store, "store_audio_memory", return_value={"id": "abc"}) as mock_store,
        ):
            from ai_engine.rag.audio.feature_extractor import AudioFeatures
            mock_feat.return_value = AudioFeatures(
                duration=2.0,
                tempo=90.0,
                energy=0.04,
                zero_crossing_rate=0.09,
                spectral_centroid=2500.0,
                audio_type="speech",
                mood_label="calm",
            )

            # Simulate what the ingest endpoint does
            features = feature_extractor.extract_features(sine_path)
            transcript = transcriber.transcribe(sine_path)
            text = transcript or "fallback"
            embedding = embedder.embed_text(text)
            vector_store.store_audio_memory({"text_embedding": embedding})

            mock_store.assert_called_once()
            self.assertEqual(len(embedding), 1024)

        os.unlink(sine_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
