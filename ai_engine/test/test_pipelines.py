"""
Unit Tests for End-to-End Processing Pipelines
Tests IngestPipeline, QueryPipeline, EventAggregator, and procedural audio synthesis.
"""
import unittest
import numpy as np
from ai_engine.pipelines.query_pipeline import MultimodalQueryPipeline
from ai_engine.story_generation.event_aggregator import EventAggregator, haversine_distance
from ai_engine.story_generation.arc_generator import NarrativeArcGenerator
from ai_engine.video_generation.audio_synth import synthesize_ambient_soundtrack, generate_chord_tone
from ai_engine.video_generation.ken_burns import get_crop_window


class TestPipelines(unittest.TestCase):
    def test_query_pipeline_execution(self):
        pipeline = MultimodalQueryPipeline()
        results = pipeline.execute_query("Sunset walk along the beach", user_id="user_pipe_1", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("id", results[0])

    def test_haversine_distance_calculation(self):
        # San Francisco to Oakland is ~13 km
        dist = haversine_distance(37.7749, -122.4194, 37.8044, -122.2712)
        self.assertGreater(dist, 10.0)
        self.assertLess(dist, 20.0)

    def test_event_aggregator_clustering(self):
        aggregator = EventAggregator(max_gap_hours=3.0)
        items = [
            {"id": "1", "timestamp": "2026-07-01T10:00:00Z"},
            {"id": "2", "timestamp": "2026-07-01T11:30:00Z"},  # same event (< 3h)
            {"id": "3", "timestamp": "2026-07-01T18:00:00Z"},  # new event (> 3h gap)
        ]
        events = aggregator.aggregate_into_events(items)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["item_count"], 2)
        self.assertEqual(events[1]["item_count"], 1)

    def test_procedural_audio_soundtrack_synthesis(self):
        audio = synthesize_ambient_soundtrack(mood="calm", duration_seconds=2.0, sample_rate=22050)
        self.assertEqual(len(audio), 2 * 22050)
        self.assertTrue(np.all(np.abs(audio) <= 1.0))
        self.assertGreater(np.max(np.abs(audio)), 0.1)

    def test_ken_burns_crop_window(self):
        x1, y1, x2, y2 = get_crop_window(
            width=1920,
            height=1080,
            start_scale=1.0,
            end_scale=1.2,
            start_center=(0.5, 0.5),
            end_center=(0.5, 0.5),
            progress=0.5,
        )
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 1920)
        self.assertLessEqual(y2, 1080)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)


if __name__ == "__main__":
    unittest.main()
