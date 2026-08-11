"""
AI Service — LLM abstraction layer with memory-aware context retrieval.

Architecture:
    User message
    ↓
    Retrieve authorized memories (enforce auth BEFORE passing to LLM)
    ↓
    Build context string
    ↓
    Call LLM (provider-agnostic interface)
    ↓
    Persist conversation + messages
    ↓
    Return response
"""

from app.core.db import get_supabase_client
from app.config.settings import settings
from app.schemas.domain import AIChatResponse, AIMessageResponse, AIConversationResponse
from fastapi import HTTPException, status
from datetime import datetime, timezone
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


# ── LLM Provider Abstraction ──────────────────────────────────────────────────

class LLMResponse:
    def __init__(self, text: str, related_memory_ids: list[str] | None = None):
        self.text = text
        self.related_memory_ids = related_memory_ids or []


def _call_openai(prompt: str, context: str, api_key: str, model: str) -> LLMResponse:
    """Call OpenAI Chat API."""
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MemoryVerse AI, a personal memory assistant. "
                        "You help users explore and relive their personal memories. "
                        "Answer based only on the provided memory context. "
                        "Be warm, personal, and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Memory context:\n{context}\n\nUser question: {prompt}",
                },
            ],
            "max_tokens": 800,
            "temperature": 0.7,
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return LLMResponse(text=answer)
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        raise


def _call_gemini(prompt: str, context: str, api_key: str, model: str) -> LLMResponse:
    """Call Google Gemini API."""
    try:
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are MemoryVerse AI, a personal memory assistant. "
                                f"Memory context:\n{context}\n\nUser question: {prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 800, "temperature": 0.7},
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return LLMResponse(text=answer)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        raise


def _fallback_response(prompt: str, context: str) -> LLMResponse:
    """Intelligent fallback when no LLM key is configured."""
    if context.strip():
        return LLMResponse(
            text=(
                "I found some relevant memories in your vault! "
                f"Here's what I can see:\n\n{context}\n\n"
                "To get richer AI-powered answers, add your LLM_API_KEY to the backend .env file."
            )
        )
    return LLMResponse(
        text=(
            "You don't have any memories uploaded yet. "
            "Upload photos and videos to your vaults, and I'll be able to tell you stories about them! "
            "(Tip: add LLM_API_KEY to backend .env for full AI capabilities.)"
        )
    )


def _call_llm(prompt: str, context: str) -> LLMResponse:
    """Route to the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    api_key = settings.LLM_API_KEY
    model = settings.LLM_MODEL

    if not api_key or provider == "none":
        return _fallback_response(prompt, context)

    try:
        if provider == "openai":
            return _call_openai(prompt, context, api_key, model)
        elif provider in ("gemini", "google"):
            return _call_gemini(prompt, context, api_key, model or "gemini-2.0-flash")
        else:
            logger.warning(f"Unknown LLM provider: {provider}. Using fallback.")
            return _fallback_response(prompt, context)
    except Exception as e:
        logger.error(f"LLM call failed: {e}. Using fallback.")
        return _fallback_response(prompt, context)


# ── Memory Context Retrieval ──────────────────────────────────────────────────

def _build_memory_context(user_id: str) -> tuple[str, list[str]]:
    """
    Retrieve authorized memories and build context string for the LLM.
    IMPORTANT: Only retrieves memories owned by the authenticated user.
    """
    supabase = get_supabase_client()
    try:
        res = (
            supabase.table("media")
            .select("id, filename, media_type, created_at, location_name, vault_id")
            .eq("owner_id", user_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        media_list = cast(list[dict[str, Any]], res.data or [])
    except Exception as e:
        logger.warning(f"Failed to retrieve memory context: {e}")
        return "", []

    if not media_list:
        return "", []

    memory_ids = [str(m["id"]) for m in media_list if "id" in m]
    lines = []
    for m in media_list:
        dt_str = str(m.get("created_at", ""))[:10] if m.get("created_at") else "unknown date"
        loc = str(m.get("location_name") or "unknown location")
        mtype = str(m.get("media_type", "file"))
        fname = str(m.get("filename", ""))
        lines.append(f"- {mtype.title()} '{fname}' on {dt_str} at {loc}")

    context = f"The user has {len(media_list)} memories:\n" + "\n".join(lines)
    return context, memory_ids


# ── Conversation Persistence ──────────────────────────────────────────────────

class AIService:

    @staticmethod
    def chat(user_id: str, message: str, conversation_id: str | None) -> AIChatResponse:
        supabase = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()

        # 1. Get or create conversation
        if conversation_id:
            conv_res = supabase.table("ai_conversations") \
                .select("*").eq("id", conversation_id).eq("user_id", user_id).execute()
            if not conv_res.data:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conv = cast(dict[str, Any], conv_res.data[0])
        else:
            # Create new conversation, title from first 60 chars of message
            title = message[:60] + ("…" if len(message) > 60 else "")
            conv_insert = supabase.table("ai_conversations").insert({
                "user_id": user_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
            }).execute()
            if not conv_insert.data:
                raise HTTPException(status_code=500, detail="Failed to create conversation")
            conv = cast(dict[str, Any], conv_insert.data[0])

        conv_id = str(conv["id"])

        # 2. Save user message
        user_msg_insert = supabase.table("ai_messages").insert({
            "conversation_id": conv_id,
            "role": "user",
            "content": message,
            "related_memory_ids": [],
            "created_at": now,
        }).execute()
        if not user_msg_insert.data:
            raise HTTPException(status_code=500, detail="Failed to save user message")
        user_msg_row = cast(dict[str, Any], user_msg_insert.data[0])

        # 3. Build memory context (authorized only)
        context, memory_ids = _build_memory_context(user_id)

        # 4. Call LLM
        llm_resp = _call_llm(message, context)

        # 5. Save assistant message
        asst_msg_insert = supabase.table("ai_messages").insert({
            "conversation_id": conv_id,
            "role": "assistant",
            "content": llm_resp.text,
            "related_memory_ids": memory_ids[:10],  # store refs
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        if not asst_msg_insert.data:
            raise HTTPException(status_code=500, detail="Failed to save assistant message")
        asst_msg_row = cast(dict[str, Any], asst_msg_insert.data[0])

        # 6. Update conversation timestamp + title if new
        supabase.table("ai_conversations") \
            .update({"updated_at": datetime.now(timezone.utc).isoformat()}) \
            .eq("id", conv_id).execute()

        def _to_msg(row: dict[str, Any]) -> AIMessageResponse:
            return AIMessageResponse(
                id=str(row["id"]),
                conversation_id=conv_id,
                role=str(row["role"]),
                content=str(row["content"]),
                related_memory_ids=[str(x) for x in (row.get("related_memory_ids") or [])],
                created_at=datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")),
            )

        return AIChatResponse(
            conversation_id=conv_id,
            user_message=_to_msg(user_msg_row),
            assistant_message=_to_msg(asst_msg_row),
        )

    @staticmethod
    def list_conversations(user_id: str) -> list[AIConversationResponse]:
        supabase = get_supabase_client()
        res = supabase.table("ai_conversations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .limit(50) \
            .execute()
        rows = cast(list[dict[str, Any]], res.data or [])
        result = []
        for r in rows:
            msg_count = supabase.table("ai_messages") \
                .select("id", count=cast(Any, "exact")).eq("conversation_id", r["id"]).execute().count or 0
            result.append(AIConversationResponse(
                id=str(r["id"]),
                title=str(r["title"]),
                created_at=datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(str(r["updated_at"]).replace("Z", "+00:00")),
                message_count=msg_count,
            ))
        return result

    @staticmethod
    def get_messages(user_id: str, conversation_id: str) -> list[AIMessageResponse]:
        supabase = get_supabase_client()
        # Verify ownership
        conv_res = supabase.table("ai_conversations") \
            .select("id").eq("id", conversation_id).eq("user_id", user_id).execute()
        if not conv_res.data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        res = supabase.table("ai_messages") \
            .select("*").eq("conversation_id", conversation_id) \
            .order("created_at", desc=False).execute()
        rows = cast(list[dict[str, Any]], res.data or [])
        return [
            AIMessageResponse(
                id=str(r["id"]),
                conversation_id=conversation_id,
                role=str(r["role"]),
                content=str(r["content"]),
                related_memory_ids=[str(x) for x in (r.get("related_memory_ids") or [])],
                created_at=datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")),
            )
            for r in rows
        ]
