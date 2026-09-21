"""
Prompts Package
Exports prompt templates for Planner, Storyteller, Auditor, and Video Director agents.
"""
from ai_engine.prompts.planner_prompts import PLANNER_SYSTEM_TEMPLATE, PLANNER_QUERY_PROMPT
from ai_engine.prompts.story_prompts import STORYTELLER_SYSTEM_TEMPLATE, STORY_ARC_GENERATION_PROMPT
from ai_engine.prompts.auditor_prompts import AUDITOR_SYSTEM_TEMPLATE, AUDIT_VERIFICATION_PROMPT
from ai_engine.prompts.director_prompts import DIRECTOR_SYSTEM_TEMPLATE, DIRECTOR_SHOT_LIST_PROMPT

__all__ = [
    "PLANNER_SYSTEM_TEMPLATE",
    "PLANNER_QUERY_PROMPT",
    "STORYTELLER_SYSTEM_TEMPLATE",
    "STORY_ARC_GENERATION_PROMPT",
    "AUDITOR_SYSTEM_TEMPLATE",
    "AUDIT_VERIFICATION_PROMPT",
    "DIRECTOR_SYSTEM_TEMPLATE",
    "DIRECTOR_SHOT_LIST_PROMPT",
]
