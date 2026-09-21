"""
Unit Tests for Multimodal RAG Components
Tests keyframe extraction, SigLIP embeddings, Librosa audio features, and intent parsing.
"""
import os
import unittest
import numpy as np
from ai_engine.rag.video.siglip_embedder import select_distinct_keyframes
from ai_engine.rag.audio.intent_extractor import extract_intent
from ai_engine.rag.audio.feature_extractor import _normalise_energy, _classify_audio_type
from ai_engine.multimodal.fusion import reciprocal_rank_fusion, weighted_multimodal_fusion
from ai_engine.retrieval.hybrid_retriever import HybridRetriever


class TestRagComponents(unittest.TestCase):
    def test_select_distinct_keyframes_mmr(self):
        # 3 frames, 2 identical and 1 different
        frames = [
            {"frame_index": 0, "embedding": [1.0, 0.0, 0.0]},
            {"frame_index": 1, "embedding": [0.99, 0.01, 0.0]},
            {"frame_index": 2, "embedding": [0.0, 1.0, 0.0]},
        ]
        selected = select_distinct_keyframes(frames, top_k=2)
        self.assertEqual(len(selected), 2)
        indices = [f["frame_index"] for f in selected]
        self.assertIn(0, indices)
        self.assertIn(2, indices)  # Most distinct frame selected

    def test_audio_intent_extractor(self):
        intent = extract_intent("Find energetic dance songs from the party")
        self.assertEqual(intent.get("intent_type"), "mood")

        intent_speech = extract_intent("What did John say during the speech?")
        self.assertEqual(intent_speech.get("intent_type"), "speech")

    def test_acoustic_feature_computation(self):
        # Generate 1-second synthetic 440Hz sine wave
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        energy = _normalise_energy(audio)
        self.assertGreater(energy, 0.0)

        audio_type = _classify_audio_type(zcr=0.04, energy=energy, spectral_centroid=1800.0, tempo=110.0)
        self.assertIn(audio_type, ["speech", "music", "ambient", "mixed"])

    def test_reciprocal_rank_fusion(self):
        list_a = [{"id": "item1"}, {"id": "item2"}]
        list_b = [{"id": "item2"}, {"id": "item3"}]
        fused = reciprocal_rank_fusion([list_a, list_b])
        # item2 appears in both lists, so its accumulated RRF score should be highest
        self.assertEqual(fused[0]["id"], "item2")

    def test_hybrid_retriever_dense_and_sparse(self):
        retriever = HybridRetriever(alpha=0.6)
        candidates = [
            {"id": "c1", "similarity": 0.5, "caption": "Hiking in Yosemite mountains"},
            {"id": "c2", "similarity": 0.9, "caption": "Dining in downtown cafe"},
        ]
        results = retriever.retrieve("hiking mountains", candidates, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("hybrid_score", results[0])


if __name__ == "__main__":
    unittest.main()
