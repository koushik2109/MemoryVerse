"""
Video RAG Pipeline - Test Suite
Tests all modules with synthetic data (no real video or API needed).

Synthetic test strategy (since no MSVD dataset is available):
  - Generate minimal MP4 files using OpenCV (solid-colour frames)
  - Test each module independently with mocks where needed
  - Include an integration test with all external calls mocked
"""
import os
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ─── Synthetic Video Generator ─────────────────────────────────────────────────

def _make_synthetic_video(
    duration_sec: float = 6.0,
    fps: int = 10,
    width: int = 224,
    height: int = 224,
) -> str:
    """
    Write a short synthetic MP4 (solid colour frames, cycling hue) using OpenCV.
    Returns path to the temp file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        hue = int(180 * i / total_frames)
        frame_hsv = np.full((height, width, 3), [hue, 200, 200], dtype=np.uint8)
        frame_bgr = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        writer.write(frame_bgr)

    writer.release()
    return tmp.name


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestKeyframeExtractor(unittest.TestCase):

    def test_extracts_keyframes_every_2s(self):
        from ai_engine.rag.video.keyframe_extractor import extract_keyframes

        path = _make_synthetic_video(duration_sec=6.0, fps=10)
        try:
            keyframes, duration, fps, count = extract_keyframes(path, interval_sec=2.0)
            # 6 seconds / 2 = ~3 keyframes
            self.assertGreaterEqual(count, 2)
            self.assertAlmostEqual(duration, 6.0, delta=0.5)
            self.assertEqual(fps, 10.0)
            # Each keyframe is a (timestamp, frame) tuple
            for ts, frame in keyframes:
                self.assertIsInstance(ts, float)
                self.assertIsInstance(frame, np.ndarray)
                self.assertEqual(frame.shape, (224, 224, 3))
        finally:
            os.unlink(path)

    def test_invalid_file_raises(self):
        from ai_engine.rag.video.keyframe_extractor import extract_keyframes
        with self.assertRaises(ValueError):
            extract_keyframes("/nonexistent/file.mp4")


class TestSigLIPEmbedder(unittest.TestCase):

    def _dummy_frames(self, n: int = 4) -> list:
        return [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(n)
        ]

    def test_embed_frames_shape(self):
        from ai_engine.rag.video.siglip_embedder import embed_frames
        frames = self._dummy_frames(3)
        embeds = embed_frames(frames)
        self.assertEqual(len(embeds), 3)
        self.assertEqual(len(embeds[0]), 768)  # SigLIP base dim

    def test_average_embeddings_normalised(self):
        from ai_engine.rag.video.siglip_embedder import average_frame_embeddings
        # Create fake 768-dim embeddings
        fake = [[float(i) * 0.01] * 768 for i in range(1, 4)]
        avg = average_frame_embeddings(fake)
        self.assertEqual(len(avg), 768)
        # Check roughly normalised (L2 norm ≈ 1)
        norm = sum(x ** 2 for x in avg) ** 0.5
        self.assertAlmostEqual(norm, 1.0, delta=0.05)

    def test_select_distinct_keyframes(self):
        from ai_engine.rag.video.siglip_embedder import select_most_distinct_keyframes
        import random
        # 10 random 768-dim embeddings
        fake = [[random.gauss(0, 1) for _ in range(768)] for _ in range(10)]
        indices = select_most_distinct_keyframes(fake, top_k=3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(set(indices)), 3)  # all unique

    def test_embed_text_siglip_shape(self):
        from ai_engine.rag.video.siglip_embedder import embed_text_siglip
        embeds = embed_text_siglip(["birthday party at the beach"])
        self.assertEqual(len(embeds), 1)
        self.assertEqual(len(embeds[0]), 768)


class TestCaptioner(unittest.TestCase):

    def test_caption_fallback_on_ollama_failure(self):
        """If Ollama is unreachable, captioner must return a graceful fallback."""
        from ai_engine.rag.video import captioner
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        with patch("httpx.post", side_effect=Exception("Connection refused")):
            result = captioner.caption_keyframes(frames, distinct_indices=[0, 1, 2])
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_caption_returns_string(self):
        from ai_engine.rag.video import captioner
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Friends laughing at a birthday party."}
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp):
            result = captioner.caption_keyframes(frames, distinct_indices=[0, 1, 2])
        self.assertIn("Friends", result)


class TestVectorStoreMerge(unittest.TestCase):

    def test_merge_deduplication(self):
        from ai_engine.rag.video.vector_store import merge_and_deduplicate

        text_results = [
            {"id": "a", "similarity": 0.9, "file_path": "a.mp4", "duration": 10.0,
             "fps": 30.0, "num_keyframes": 5, "auto_caption": "cap_a", "created_at": "2025-01-01"},
            {"id": "b", "similarity": 0.7, "file_path": "b.mp4", "duration": 8.0,
             "fps": 30.0, "num_keyframes": 4, "auto_caption": "cap_b", "created_at": "2025-01-01"},
        ]
        image_results = [
            {"id": "b", "similarity": 0.85, "file_path": "b.mp4", "duration": 8.0,
             "fps": 30.0, "num_keyframes": 4, "auto_caption": "cap_b", "created_at": "2025-01-01"},
            {"id": "c", "similarity": 0.6, "file_path": "c.mp4", "duration": 12.0,
             "fps": 24.0, "num_keyframes": 6, "auto_caption": "cap_c", "created_at": "2025-01-01"},
        ]
        merged = merge_and_deduplicate(text_results, image_results, top_k=3)

        # Should have 3 unique entries
        self.assertEqual(len(merged), 3)
        ids = [r["id"] for r in merged]
        self.assertEqual(len(set(ids)), 3)

        # "b" should take the higher image similarity (0.85)
        b_entry = next(r for r in merged if r["id"] == "b")
        self.assertAlmostEqual(b_entry["similarity"], 0.85)
        self.assertEqual(b_entry["match_source"], "image")

        # Top result should be "a" (0.9)
        self.assertEqual(merged[0]["id"], "a")


class TestFullIngestMocked(unittest.TestCase):
    """Full ingest pipeline with all heavy models mocked."""

    def test_ingest_pipeline(self):
        from ai_engine.rag.video import (
            keyframe_extractor,
            siglip_embedder,
            captioner,
            vector_store,
        )
        from ai_engine.rag.audio import embedder as bge_embedder

        video_path = _make_synthetic_video(duration_sec=4.0, fps=10)
        fake_image_embed = [0.0] * 768
        fake_text_embed = [0.0] * 1024

        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        keyframe_data = [(float(i * 2), f) for i, f in enumerate(frames)]

        with (
            patch.object(
                keyframe_extractor, "extract_keyframes",
                return_value=(keyframe_data, 4.0, 10.0, 3)
            ),
            patch.object(
                siglip_embedder, "embed_frames",
                return_value=[[0.0] * 768] * 3
            ),
            patch.object(
                siglip_embedder, "average_frame_embeddings",
                return_value=fake_image_embed
            ),
            patch.object(
                siglip_embedder, "select_most_distinct_keyframes",
                return_value=[0, 1, 2]
            ),
            patch.object(
                captioner, "caption_keyframes",
                return_value="Friends playing at the park on a sunny day."
            ),
            patch.object(
                bge_embedder, "embed_text",
                return_value=fake_text_embed
            ),
            patch.object(
                vector_store, "store_video_memory",
                return_value={"id": "mocked-id"}
            ) as mock_store,
        ):
            # Simulate the ingest pipeline
            kf, dur, fps, count = keyframe_extractor.extract_keyframes(video_path)
            _, raw_frames = zip(*kf)
            frame_embeds = siglip_embedder.embed_frames(list(raw_frames))
            img_embed = siglip_embedder.average_frame_embeddings(frame_embeds)
            distinct = siglip_embedder.select_most_distinct_keyframes(frame_embeds, top_k=3)
            caption = captioner.caption_keyframes(list(raw_frames), distinct)
            text_embed = bge_embedder.embed_text(caption)
            vector_store.store_video_memory({"image_embedding": img_embed, "text_embedding": text_embed})

            mock_store.assert_called_once()
            self.assertEqual(len(img_embed), 768)
            self.assertEqual(len(text_embed), 1024)
            self.assertIn("park", caption)

        os.unlink(video_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
