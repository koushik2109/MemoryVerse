import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_engine.rag.scripts.compute_human_agreement import cohen_kappa
from ai_engine.rag.scripts.prepare_human_audit import _write_simple_form, stratified_sample
from ai_engine.rag.scripts.run_expanded_multihop_eval import build_answer_prompt


class HumanAuditHelpersTest(unittest.TestCase):
    def test_perfect_agreement_has_unit_kappa(self):
        labels = ["supported", "mixed", "unsupported", "supported"]
        self.assertAlmostEqual(
            cohen_kappa(labels, labels, ("supported", "mixed", "unsupported")),
            1.0,
        )

    def test_stratified_sampling_is_reproducible(self):
        cases = [
            {"id": f"sh_{index:02d}", "is_multihop": False}
            for index in range(10)
        ] + [
            {"id": f"mh_{index:02d}", "is_multihop": True}
            for index in range(10)
        ]
        first = stratified_sample(cases, sample_size=10, seed=42)
        second = stratified_sample(cases, sample_size=10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(not case["is_multihop"] for case in first), 5)
        self.assertEqual(sum(case["is_multihop"] for case in first), 5)

    def test_simple_form_contains_case_data_and_csv_contract(self):
        case = {
            "audit_id": "AUDIT-001",
            "query": "Find the answer.",
            "target_keyword": "target",
            "split": "single_hop",
            "retrieved_context": ["Evidence one", "Evidence two"],
            "retrieved_media": [
                {
                    "item_id": "IMG-1",
                    "file_path": "images/example.jpg",
                    "resolved_media_path": "/tmp/example.jpg",
                    "raw_media_available": True,
                }
            ],
            "answer": "The answer is supported.",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviewer_1_form.html"
            _write_simple_form(path, [case], "Reviewer 1", "reviewer_1")
            form = path.read_text(encoding="utf-8")

        self.assertIn('"audit_id": "AUDIT-001"', form)
        self.assertIn("Download completed CSV", form)
        self.assertIn('const DOWNLOAD_STEM = "reviewer_1";', form)
        self.assertIn("faithfulness_label", form)
        self.assertIn("relevance_label", form)
        self.assertIn("correctness_label", form)
        self.assertIn("Retrieved raw media", form)
        self.assertIn('"item_id": "IMG-1"', form)

    def test_answer_prompt_requires_direct_evidence_and_honest_media_limit(self):
        prompt = build_answer_prompt(
            "Find the image and identify its district.",
            ["cyberpunk night bridge in Yinzhou District photos"],
        )
        self.assertIn("direct answer supported by the context", prompt)
        self.assertIn("cite the supporting Document number(s)", prompt)
        self.assertIn("does not establish it", prompt)
        self.assertIn("Do not infer a cross-modal association", prompt)
        self.assertNotIn("Target detail to check", prompt)

    def test_confidence_gated_bridge_validation(self):
        from ai_engine.rag.scripts.run_expanded_multihop_eval import is_valid_bridge_entity
        self.assertFalse(is_valid_bridge_entity("N/A", "any context"))
        self.assertFalse(is_valid_bridge_entity("No specific value found.", "context text"))
        self.assertFalse(is_valid_bridge_entity("layered cocktail", "beverage glass pouring water"))
        self.assertTrue(is_valid_bridge_entity("thunderstorm", "outdoor rainy skies thunderstorm weather"))


if __name__ == "__main__":
    unittest.main()

