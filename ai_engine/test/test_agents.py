"""
Unit Tests for Specialized Cognitive Agents
Tests PlanningAgent, MediaScorerAgent, StorytellerAgent, FactAuditorAgent, and VideoDirectorAgent.
"""
import unittest
from ai_engine.agents.planner.agent import PlanningAgent
from ai_engine.agents.scorer.agent import MediaScorerAgent
from ai_engine.agents.scorer.quality_metrics import evaluate_media_quality, compute_laplacian_variance
from ai_engine.agents.scorer.dedup_clustering import cluster_and_deduplicate
from ai_engine.agents.storyteller.agent import StorytellerAgent
from ai_engine.agents.storyteller.narrative_arc import segment_assets_into_phases
from ai_engine.agents.narrator.agent import FactAuditorAgent
from ai_engine.agents.narrator.evidence_manifest import verify_entities_against_manifest, sanitize_statement
from ai_engine.agents.video.agent import VideoDirectorAgent
from ai_engine.agents.video.motion_planner import get_motion_for_shot, interpolate_motion_frame


class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PlanningAgent()

    def test_decompose_query_temporal_and_mood(self):
        plan = self.agent.decompose_query("Trip to Paris last summer 2024 celebration")
        self.assertEqual(plan["temporal_hint"]["year"], 2024)
        self.assertEqual(plan["target_mood"], "energetic")
        self.assertIn("visual", plan["weights"])
        self.assertIn("audio", plan["weights"])
        self.assertIn("text", plan["weights"])

    def test_decompose_query_audio_intent(self):
        plan = self.agent.decompose_query("What did mom say at the campfire?")
        self.assertGreaterEqual(plan["weights"]["audio"], 0.35)

    def test_execute_retrieval_synthetic_fallback(self):
        plan = self.agent.decompose_query("Hiking in Yosemite")
        items = self.agent.execute_retrieval(plan, user_id="test_user", limit=4)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("file_path", items[0])


class TestMediaScorerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MediaScorerAgent(blur_threshold=80.0, duplicate_similarity_threshold=0.88)

    def test_quality_metrics_fallback(self):
        res = evaluate_media_quality("/non/existent/path.jpg")
        self.assertTrue(res["is_qualified"])
        self.assertGreaterEqual(res["quality_score"], 0.0)

    def test_deduplication_clustering(self):
        # Create 3 items, two with identical embeddings
        items = [
            {"id": "img1", "embedding": [1.0, 0.0, 0.0], "visual_quality_score": 0.9},
            {"id": "img2", "embedding": [0.99, 0.05, 0.0], "visual_quality_score": 0.6},  # duplicate of img1
            {"id": "img3", "embedding": [0.0, 1.0, 0.0], "visual_quality_score": 0.8},  # distinct
        ]
        deduped, rejected = cluster_and_deduplicate(items, similarity_threshold=0.88)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(deduped[0]["id"], "img1")  # Higher quality score won

    def test_score_items_ranking(self):
        items = [
            {"id": "a", "file_path": "a.jpg", "similarity": 0.6},
            {"id": "b", "file_path": "b.jpg", "similarity": 0.95},
        ]
        scored = self.agent.score_items(items)
        self.assertGreater(scored[0]["composite_score"], scored[1]["composite_score"])
        self.assertEqual(scored[0]["id"], "b")


class TestStorytellerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = StorytellerAgent()

    def test_build_evidence_manifest(self):
        items = [
            {"id": "1", "auto_caption": "Friends smiling at sunset", "source_modality": "image"},
            {"id": "2", "auto_caption": "Cheering on the trail", "source_modality": "video"},
        ]
        manifest = self.agent.build_evidence_manifest(items)
        self.assertEqual(manifest["total_assets"], 2)
        self.assertEqual(len(manifest["assets"]), 2)

    def test_generate_narrative_arc_phases(self):
        items = [
            {"id": "1", "auto_caption": "Morning arrival at camp", "source_modality": "image"},
            {"id": "2", "auto_caption": "Hiking along the ridgeline", "source_modality": "image"},
            {"id": "3", "auto_caption": "Reaching the mountain peak", "source_modality": "image"},
            {"id": "4", "auto_caption": "Campfire memories at night", "source_modality": "image"},
        ]
        manifest = self.agent.build_evidence_manifest(items)
        script = self.agent.generate_narrative_arc("Mountain Trek", manifest, target_duration=30.0)
        self.assertEqual(len(script["phases"]), 4)
        phase_names = [p["phase"] for p in script["phases"]]
        self.assertEqual(phase_names, ["hook", "build_up", "climax", "resolution"])


class TestFactAuditorAgent(unittest.TestCase):
    def setUp(self):
        self.agent = FactAuditorAgent()

    def test_audit_grounded_script_passes(self):
        manifest = {
            "assets": [
                {"id": "1", "caption": "Hiking through the forest on a sunny morning"},
                {"id": "2", "caption": "Reaching the mountain cabin"},
            ]
        }
        script = {
            "phases": [
                {
                    "phase": "hook",
                    "narration": "We started our hike through the forest under the morning sun.",
                    "referenced_media_ids": ["1"],
                },
                {
                    "phase": "resolution",
                    "narration": "Finally we arrived safely at the mountain cabin.",
                    "referenced_media_ids": ["2"],
                },
            ]
        }
        report = self.agent.audit_script(script, manifest)
        self.assertTrue(report["audit_passed"])
        self.assertEqual(report["hallucination_rate"], 0.0)

    def test_audit_hallucinated_location_sanitization(self):
        manifest = {
            "assets": [{"id": "1", "caption": "Walking on a local park trail"}]
        }
        script = {
            "phases": [
                {
                    "phase": "hook",
                    "narration": "We spent an unforgettable holiday in Paris exploring cafes.",
                    "referenced_media_ids": ["1"],
                }
            ]
        }
        report = self.agent.audit_script(script, manifest)
        # Paris should be flagged and sanitized
        self.assertNotIn("Paris", report["sanitized_script"]["full_narration"])


class TestVideoDirectorAgent(unittest.TestCase):
    def setUp(self):
        self.agent = VideoDirectorAgent(fps=30)

    def test_ken_burns_motion_generation(self):
        motion = get_motion_for_shot(0)
        self.assertIn("type", motion)
        self.assertIn("start_scale", motion)

        scale, cx, cy = interpolate_motion_frame(motion, 0.5)
        self.assertGreaterEqual(scale, 1.0)
        self.assertAlmostEqual(cx, 0.5, delta=0.2)

    def test_compile_shot_list(self):
        script = {
            "phases": [
                {"phase": "hook", "target_seconds": 6.0, "referenced_media_ids": ["m1"]},
                {"phase": "climax", "target_seconds": 8.0, "referenced_media_ids": ["m2"]},
            ]
        }
        scored = [
            {"id": "m1", "file_path": "/media/m1.jpg"},
            {"id": "m2", "file_path": "/media/m2.jpg"},
        ]
        shots = self.agent.compile_shot_list(script, scored, target_duration=14.0)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["shot_index"], 1)
        self.assertEqual(shots[0]["duration"], 6.0)
        self.assertEqual(shots[1]["duration"], 8.0)


if __name__ == "__main__":
    unittest.main()
