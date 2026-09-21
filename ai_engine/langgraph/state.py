"""
MemoryVerse LangGraph Agent State
Defines the shared state passed between agents in the cognitive pipeline.
"""
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Inputs
    query: str
    user_id: str
    ablation_mode: str  # "model_1_vanilla_rag", "model_2_single_prompt", "model_3_no_auditor", "model_4_full"
    media_type: str     # "all", "video", "image", "audio"
    target_duration: float  # In seconds (e.g. 30.0 for video, 0 for pure QA)
    metadata_filter: Optional[Dict[str, Any]]

    # Pipeline Artifacts
    plan: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
    scored_items: List[Dict[str, Any]]
    evidence_manifest: Dict[str, Any]
    script: Dict[str, Any]
    audit_result: Dict[str, Any]
    shot_list: List[Dict[str, Any]]
    final_output: Dict[str, Any]

    # Control Flow & Telemetry
    iteration: int
    max_retries: int
    errors: List[str]
    metrics: Dict[str, Any]
