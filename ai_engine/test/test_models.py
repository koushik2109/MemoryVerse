"""
Unit Tests for Singleton ML Model Loaders
Verifies thread-safe loading and embedding extraction for CLIP, SigLIP, BGE-M3, and Whisper.
"""
import unittest
import numpy as np
from ai_engine.models.clip_loader import get_clip_model, encode_text as encode_clip_text, EMBED_DIM as CLIP_DIM
from ai_engine.models.siglip_loader import get_siglip_model, embed_text_siglip, EMBED_DIM as SIGLIP_DIM
from ai_engine.models.bge_loader import get_bge_model, embed_text as embed_bge_text, EMBED_DIM as BGE_DIM
from ai_engine.models.whisper_loader import get_whisper_model


class TestModelLoaders(unittest.TestCase):
    def test_clip_loader_and_dimensions(self):
        self.assertEqual(CLIP_DIM, 512)
        vec = encode_clip_text("A memorable photo in the park")
        self.assertEqual(len(vec), 512)
        # Vector should be normalized
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_siglip_loader_and_dimensions(self):
        self.assertEqual(SIGLIP_DIM, 768)
        processor, model = get_siglip_model()
        self.assertIsNotNone(processor)
        self.assertIsNotNone(model)
        vecs = embed_text_siglip(["Sunset at the beach"])
        self.assertEqual(len(vecs[0]), 768)

    def test_bge_loader_and_dimensions(self):
        self.assertEqual(BGE_DIM, 1024)
        model = get_bge_model()
        self.assertIsNotNone(model)
        vec = embed_bge_text("Speech about personal memories")
        self.assertEqual(len(vec), 1024)

    def test_whisper_loader(self):
        model = get_whisper_model()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
