"""
Query Intent Extractor
Uses Ollama (local LLM) to extract search intent from a natural-language query.
Returns a structured dict with:
  - intent_type: "speech" | "mood" | "ambient" | "general"
  - mood_label:  (optional) energetic | calm | intense | light | neutral
  - environment: (optional) rain | crowd | nature | indoor | transport etc
  - keywords:    list of key phrases for embedding
"""
import json
import re
from typing import Any, Dict

import httpx

from ai_engine.rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL


_SYSTEM_PROMPT = """You are a search intent extractor for a personal memory app that stores audio files.

Given a user's search query, extract and return ONLY valid JSON with these keys:
- "intent_type": one of ["speech", "mood", "ambient", "general"]
  * speech → user is looking for spoken content (conversations, narration)
  * mood   → user is looking by feeling/emotion/energy (laughing, crying, party)
  * ambient → user is looking by environment (rain, coffee shop, nature)
  * general → unclear or mixed intent
- "mood_label": one of [null, "energetic", "calm", "intense", "light", "neutral"]
- "environment": one of [null, "rain", "crowd", "nature", "indoor", "transport", "outdoor"]
- "keywords": array of key phrases to use for semantic search (max 5)

Return ONLY the JSON object, no markdown, no explanation."""


def _call_ollama(user_query: str) -> str:
    """Call Ollama chat endpoint and return the raw text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        "stream": False,
        "format": "json",
    }
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _fallback_intent(query: str) -> Dict[str, Any]:
    """
    Simple keyword-based fallback if Ollama is unavailable.
    """
    q = query.lower()
    mood_words = {"laugh", "cry", "party", "happy", "sad", "angry", "excited", "relax", "chill"}
    ambient_words = {"rain", "crowd", "nature", "birds", "wind", "fire", "ocean", "traffic"}

    if any(w in q for w in mood_words):
        intent_type = "mood"
    elif any(w in q for w in ambient_words):
        intent_type = "ambient"
    elif "said" in q or "told" in q or "talking" in q or "speech" in q:
        intent_type = "speech"
    else:
        intent_type = "general"

    return {
        "intent_type": intent_type,
        "mood_label": None,
        "environment": None,
        "keywords": [query],
    }


def extract_intent(query: str) -> Dict[str, Any]:
    """
    Extract search intent from a natural-language query.
    Falls back to rule-based extraction if Ollama is unavailable.
    """
    try:
        raw = _call_ollama(query)
        # Strip any markdown code fences just in case
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        intent = json.loads(raw)
        # Ensure keywords is always a list
        if "keywords" not in intent or not isinstance(intent["keywords"], list):
            intent["keywords"] = [query]
        return intent
    except Exception:
        return _fallback_intent(query)
