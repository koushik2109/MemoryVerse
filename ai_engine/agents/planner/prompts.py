"""
Planning Agent Prompt Templates & Schemas
Formats structured query expansion, temporal windowing, and multimodal weight assignments.
"""
from typing import Any, Dict

PLANNER_SYSTEM_PROMPT = """You are the Retrieval Planning Agent for MemoryVerse.
Your task is to decompose a user's natural language memory request into a structured retrieval plan.
Extract:
1. Temporal boundaries (exact year, season, date range, or relative markers)
2. Spatial / GPS hints (cities, landmarks, indoor vs outdoor)
3. Target emotional mood (energetic, calm, nostalgic, celebratory, intense)
4. Modality weights (visual, audio, text) based on query focus
5. Dispatched search query variants for dual-embedding vector search
"""

PLANNER_QUERY_EXPANSION_PROMPT = """User Query: "{query}"

Analyze the user's memory request and produce a structured JSON plan matching:
{{
  "temporal_hint": {{ "year": Optional[int], "season": Optional[str], "relative": Optional[str] }},
  "target_mood": "energetic" | "calm" | "nostalgic" | "intense" | "light",
  "weights": {{ "visual": float, "text": float, "audio": float }},
  "visual_queries": [str],
  "text_queries": [str],
  "audio_queries": [str]
}}
"""

def format_planner_prompt(query: str) -> str:
    """Formats the query into a standardized prompt for LLM-based planning."""
    return f"{PLANNER_SYSTEM_PROMPT}\n\n{PLANNER_QUERY_EXPANSION_PROMPT.format(query=query)}"
