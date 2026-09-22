"""
MemoryVerse Multi-Agent LangGraph Cognitive Engine
Wires together the Planner, Scorer, Storyteller, Auditor, Video Director,
and Synthesizer into an executable cyclic state graph.
"""
import logging
import time
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END

from ai_engine.langgraph.state import AgentState
from ai_engine.langgraph.edges import (
    route_after_planner,
    route_after_scorer,
    route_after_storyteller,
    route_after_auditor,
)
from ai_engine.agents.planner.agent import PlanningAgent
from ai_engine.agents.scorer.agent import MediaScorerAgent
from ai_engine.agents.storyteller.agent import StorytellerAgent
from ai_engine.agents.narrator.agent import FactAuditorAgent
from ai_engine.agents.video.agent import VideoDirectorAgent

logger = logging.getLogger(__name__)


# ─── Node Implementations ──────────────────────────────────────────────────────

def planner_node(state: AgentState) -> Dict[str, Any]:
    agent = PlanningAgent()
    return agent.run(dict(state))


def scorer_node(state: AgentState) -> Dict[str, Any]:
    agent = MediaScorerAgent()
    return agent.run(dict(state))


def storyteller_node(state: AgentState) -> Dict[str, Any]:
    agent = StorytellerAgent()
    return agent.run(dict(state))


def auditor_node(state: AgentState) -> Dict[str, Any]:
    agent = FactAuditorAgent()
    return agent.run(dict(state))


def video_director_node(state: AgentState) -> Dict[str, Any]:
    agent = VideoDirectorAgent()
    return agent.run(dict(state))


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Direct Q&A synthesis node (used for question answering and Model 1/2 ablations).
    """
    query = state.get("query", "")
    retrieved = state.get("retrieved_items", [])
    mode = state.get("ablation_mode", "model_4_full")

    # Build context from retrieved items
    context_lines = []
    for it in retrieved[:5]:
        caption = it.get("auto_caption", "")
        ts = it.get("timestamp", "")
        context_lines.append(f"- [{ts}] {caption}")

    context_str = "\n".join(context_lines)
    answer_text = (
        f"Based on your memories about '{query}', here is what happened:\n\n"
        f"{context_str}\n\n"
        f"These moments reflect your memorable experiences recorded in MemoryVerse."
    )

    final_output = {
        "title": f"Summary: {query}",
        "response_type": "text_qa",
        "answer": answer_text,
        "ablation_mode": mode,
        "num_sources": len(retrieved),
        "source_memories": [it.get("id", "") for it in retrieved[:5]],
    }

    return {
        "final_output": final_output,
    }


# ─── Graph Builder ─────────────────────────────────────────────────────────────

def create_memory_graph():
    """
    Compiles and returns the LangGraph executable workflow.
    """
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("storyteller", storyteller_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("video_director", video_director_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Edge from START to Planner
    workflow.add_edge(START, "planner")

    # Conditional branching from Planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "scorer": "scorer",
            "synthesizer": "synthesizer",
        },
    )

    # Scorer to Storyteller
    workflow.add_edge("scorer", "storyteller")

    # Conditional branching from Storyteller
    workflow.add_conditional_edges(
        "storyteller",
        route_after_storyteller,
        {
            "auditor": "auditor",
            "video_director": "video_director",
            "synthesizer": "synthesizer",
        },
    )

    # Conditional branching from Auditor
    workflow.add_conditional_edges(
        "auditor",
        route_after_auditor,
        {
            "storyteller": "storyteller",
            "video_director": "video_director",
            "synthesizer": "synthesizer",
        },
    )

    # Terminal edges
    workflow.add_edge("video_director", END)
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ─── Public Runner ─────────────────────────────────────────────────────────────

def run_memory_flow(
    query: str,
    user_id: str = "default_user",
    ablation_mode: str = "model_4_full",
    target_duration: float = 30.0,
    media_type: str = "all",
) -> Dict[str, Any]:
    """
    Executes the multi-agent cognitive pipeline for a user query.
    """
    start_time = time.time()
    graph = create_memory_graph()

    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
        "ablation_mode": ablation_mode,
        "target_duration": target_duration,
        "media_type": media_type,
        "iteration": 0,
        "max_retries": 2,
        "errors": [],
        "metrics": {},
    }

    result = graph.invoke(initial_state)
    elapsed = time.time() - start_time

    # Attach performance telemetry
    if "final_output" in result and isinstance(result["final_output"], dict):
        result["final_output"]["latency_seconds"] = round(elapsed, 3)

    return result
