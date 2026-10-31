"""
Unit Tests for End-to-End Processing Pipelines
Tests query/story pipelines and the no-generated-music video policy.
"""
import unittest
from ai_engine.pipelines.query_pipeline import MultimodalQueryPipeline
from ai_engine.story_generation.event_aggregator import EventAggregator, haversine_distance
from ai_engine.story_generation.arc_generator import NarrativeArcGenerator
from ai_engine.video_generation.renderer import VideoRenderer


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

    def test_video_renderer_reports_background_music_unavailable(self):
        spec = VideoRenderer().render_spec([], target_duration=12.0)
        self.assertEqual(spec["background_music"], "not_added_no_licensed_source")


if __name__ == "__main__":
    unittest.main()
