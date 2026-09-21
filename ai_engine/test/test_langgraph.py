"""
Unit & Integration Tests for LangGraph Multi-Agent Orchestration
Tests StateGraph compilation, conditional ablation routes, and memory flow execution.
"""
import unittest
from ai_engine.langgraph.state import AgentState
from ai_engine.langgraph.graph import create_memory_graph, run_memory_flow


class TestLangGraphOrchestration(unittest.TestCase):
    def test_create_memory_graph_compilation(self):
        graph = create_memory_graph()
        self.assertIsNotNone(graph)
        self.assertTrue(hasattr(graph, "invoke"))

    def test_run_memory_flow_model_4_full(self):
        result = run_memory_flow(
            query="Family roadtrip to Grand Canyon",
            user_id="test_user_001",
            ablation_mode="model_4_full",
            target_duration=25.0,
        )
        self.assertEqual(result["ablation_mode"], "model_4_full")
        self.assertIn("final_output", result)
        self.assertEqual(result["final_output"]["status"], "ready_for_render")
        self.assertGreaterEqual(result["final_output"]["total_shots"], 1)

    def test_run_memory_flow_model_1_ablation(self):
        result = run_memory_flow(
            query="Summer beach day",
            user_id="test_user_002",
            ablation_mode="model_1_vanilla_rag",
            target_duration=20.0,
        )
        self.assertEqual(result["ablation_mode"], "model_1_vanilla_rag")
        self.assertIn("final_output", result)

    def test_run_memory_flow_model_2_ablation(self):
        result = run_memory_flow(
            query="Birthday dinner party",
            user_id="test_user_003",
            ablation_mode="model_2_single_prompt",
            target_duration=20.0,
        )
        self.assertEqual(result["ablation_mode"], "model_2_single_prompt")
        self.assertIn("final_output", result)

    def test_run_memory_flow_model_3_ablation(self):
        result = run_memory_flow(
            query="Graduation ceremony",
            user_id="test_user_004",
            ablation_mode="model_3_no_auditor",
            target_duration=20.0,
        )
        self.assertEqual(result["ablation_mode"], "model_3_no_auditor")
        self.assertIn("final_output", result)


if __name__ == "__main__":
    unittest.main()
