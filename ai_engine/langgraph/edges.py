"""
LangGraph Edge Routing Logic
Governs transitions between agents, including conditional loopbacks,
ablation mode bypasses, and multi-modal completion branches.
"""
from typing import Any, Dict
from ai_engine.langgraph.state import AgentState


def route_after_planner(state: AgentState) -> str:
    """
    Route based on ablation mode:
    - model_1_vanilla_rag: Bypasses scorer, storyteller, and auditor -> directly synthesizes.
    - default / model_2 / 3 / 4: Proceeds to media scoring and curation.
    """
    mode = state.get("ablation_mode", "model_4_full")
    if mode == "model_1_vanilla_rag":
        return "synthesizer"
    return "scorer"


def route_after_scorer(state: AgentState) -> str:
    """
    Route after media quality scoring.
    """
    return "storyteller"


def route_after_storyteller(state: AgentState) -> str:
    """
    Route based on ablation mode:
    - model_2_single_prompt & model_3_no_auditor: Bypasses the fact verification auditor.
    - default / model_4_full: Proceeds through strict fact auditing.
    """
    mode = state.get("ablation_mode", "model_4_full")
    if mode in ("model_2_single_prompt", "model_3_no_auditor"):
        target_duration = float(state.get("target_duration", 30.0))
        if target_duration > 0:
            return "video_director"
        return "synthesizer"
    return "auditor"


def route_after_auditor(state: AgentState) -> str:
    """
    Evaluates auditor verdict:
    - If audit failed and retry budget remains -> self-correction loop back to storyteller.
    - If audit passed -> proceeds to video director or QA synthesizer.
    """
    audit = state.get("audit_result", {})
    passed = audit.get("audit_passed", True)
    iteration = state.get("iteration", 1)
    max_retries = state.get("max_retries", 2)

    if not passed and iteration < max_retries:
        return "storyteller"

    target_duration = float(state.get("target_duration", 30.0))
    if target_duration > 0:
        return "video_director"
    return "synthesizer"
