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
from app.schemas.domain import AIChatResponse, AIMessageResponse, AIConversationResponse, MediaResponse
from fastapi import HTTPException, status
from datetime import datetime, timezone
from typing import Any, cast, Optional, List, Dict, Tuple
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


def _downsample_image_for_vlm(image_bytes: bytes, max_dim: int = 512, quality: int = 75) -> bytes:
    """Resize high-res images to max_dim (512px) and compress to ultra-lightweight JPEG (~30KB) to prevent network/timeout choke."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image downsampling failed, using original bytes: {e}")
        return image_bytes


def _call_openrouter(prompt: str, context: str, api_key: str, model: str) -> LLMResponse:
    """Call OpenRouter Chat API with automatic free model fallback on rate limit/404."""
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://memoryverse.app",
        "X-Title": "MemoryVerse AI",
        "Content-Type": "application/json",
    }

    # Candidate free models to try in order (active verified models first)
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend([
        "inclusionai/ling-3.0-flash-fin:free",
        "liquid/lfm-2.5-2.6b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "minimax/minimax-m3:free",
        "openai/gpt-4o-mini",
    ])

    # Deduplicate keeping order
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_exc = None
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=35.0, write=30.0, pool=10.0)) as client:
        for m in models_to_try:
            try:
                sys_content = (
                    "You are MemoryVerse AI, a warm and helpful personal memory assistant. "
                    "You help users explore, search, and relive their personal memories. "
                    "Answer based on the provided memory context. "
                    "Be warm, personal, and conversational. "
                    "When presenting, filtering, or highlighting memories, describe them naturally in sentences or bullet points. "
                    "Do NOT render ASCII or Markdown tables of filenames, because the app automatically displays interactive visual photo and video cards for the user below your message."
                ) if context else "You are an intelligent AI assistant. Follow the prompt instructions precisely."

                user_content = f"Memory context:\n{context}\n\nUser question: {prompt}" if context else prompt

                body = {
                    "model": m,
                    "messages": [
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.4,
                }
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices and "message" in choices[0]:
                        answer = choices[0]["message"]["content"].strip()
                        return LLMResponse(text=answer)
                logger.warning(f"OpenRouter model {m} returned {resp.status_code}: {resp.text[:120]}")
            except Exception as e:
                logger.warning(f"OpenRouter model {m} failed: {e}")
                last_exc = e

    if last_exc:
        raise last_exc
    raise RuntimeError("All OpenRouter candidate models failed.")


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
        # Auto-detect OpenRouter keys starting with sk-or-v1-
        if api_key.startswith("sk-or-v1-") or provider == "openrouter":
            return _call_openrouter(prompt, context, api_key, model)
        elif provider == "openai":
            return _call_openai(prompt, context, api_key, model)
        elif provider in ("gemini", "google"):
            return _call_gemini(prompt, context, api_key, model or "gemini-2.0-flash")
        else:
            logger.warning(f"Unknown LLM provider: {provider}. Using fallback.")
            return _fallback_response(prompt, context)
    except Exception as e:
        logger.error(f"LLM call failed: {e}. Using fallback.")
        return _fallback_response(prompt, context)


def call_llm(prompt: str, preferred_model: Optional[str] = None) -> str:
    """Public helper to call configured LLM for general text generation and validation tasks."""
    res = _call_llm(prompt, context="")
    return res.text




# ── Vision-Language Model (VLM) Abstraction ───────────────────────────────────

def _call_openrouter_vlm(image_bytes: bytes, prompt: str, api_key: str, preferred_model: str = "") -> str:
    """Call OpenRouter multimodal vision models with downsampled image payload and automatic fallback."""
    import httpx
    import base64

    # Crucial: downsample image so base64 string is ~50KB instead of 15MB
    compact_bytes = _downsample_image_for_vlm(image_bytes, max_dim=768, quality=80)
    b64_img = base64.b64encode(compact_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_img}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://memoryverse.app",
        "X-Title": "MemoryVerse VLM",
        "Content-Type": "application/json",
    }

    # Vision-capable candidate models in priority order (verified free models first)
    candidate_vision_models = []
    if preferred_model and any(v in preferred_model.lower() for v in ["vl", "vision", "omni", "m3", "gemma", "gpt-4"]):
        candidate_vision_models.append(preferred_model)
    candidate_vision_models.extend([
        "minimax/minimax-m3:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "openai/gpt-4o-mini",
        "qwen/qwen-2.5-vl-72b-instruct",
    ])

    seen = set()
    models_to_try = [m for m in candidate_vision_models if not (m in seen or seen.add(m))]

    last_exc = None
    with httpx.Client(timeout=httpx.Timeout(connect=15.0, read=45.0, write=60.0, pool=15.0)) as client:
        for m in models_to_try:
            try:
                body = {
                    "model": m,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    "max_tokens": 400,
                    "temperature": 0.2,
                }
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content")
                        if content:
                            return content.strip()
                logger.warning(f"OpenRouter VLM model {m} returned status {resp.status_code}: {resp.text[:120]}")
            except Exception as e:
                logger.warning(f"OpenRouter VLM model {m} failed: {e}")
                last_exc = e

    if last_exc:
        raise last_exc
    raise RuntimeError("All OpenRouter vision models failed.")


def call_vlm(image_bytes: bytes, prompt: str) -> str:
    """
    Public entrypoint for Vision-Language Model analysis.
    Sends image bytes and prompt to configured vision provider.
    """
    provider = settings.LLM_PROVIDER.lower()
    api_key = settings.LLM_API_KEY
    model = settings.LLM_MODEL

    if not api_key or provider == "none":
        return "Image moment (VLM key not configured)"

    if api_key.startswith("sk-or-v1-") or provider == "openrouter":
        return _call_openrouter_vlm(image_bytes, prompt, api_key, model)
    elif provider == "openai":
        # Standard OpenAI vision
        import httpx
        import base64
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model or "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            "max_tokens": 300,
        }
        with httpx.Client(timeout=30) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    else:
        logger.warning(f"Vision not supported directly for provider {provider}. Returning default description.")
        return "Visual scene moment"


def call_structured_vlm(
    image_bytes: bytes,
    prompt: str,
    default_fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Public entrypoint for structured JSON extraction from an image using VLM.
    Extracts and validates JSON matching the prompt instructions.
    """
    import json
    import re

    system_instruction = (
        f"{prompt}\n\n"
        "CRITICAL: Return ONLY valid JSON format. Do NOT wrap with markdown, explanation, or code fences."
    )

    try:
        raw_text = call_vlm(image_bytes, system_instruction)
        # Strip markdown fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        # Extract first JSON object if surrounded by extra text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"call_structured_vlm failed ({e}), returning fallback.")
        return default_fallback or {}



# ── Memory Context Retrieval ──────────────────────────────────────────────────

def _build_memory_context(user_id: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve authorized memories and build context string for the LLM.
    IMPORTANT: Only retrieves memories owned by the authenticated user.
    """
    supabase = get_supabase_client()
    try:
        res = (
            supabase.table("media")
            .select("id, filename, media_type, created_at, location_name, vault_id, memory_id, owner_id, storage_path, url, thumbnail_url, file_size, mime_type")
            .eq("owner_id", user_id)
            .order("created_at", desc=True)
            .limit(60)
            .execute()
        )
        media_list = cast(list[dict[str, Any]], res.data or [])
    except Exception as e:
        logger.warning(f"Failed to retrieve memory context: {e}")
        return "", []

    if not media_list:
        return "", []

    lines = []
    for m in media_list:
        dt_str = str(m.get("created_at", ""))[:10] if m.get("created_at") else "unknown date"
        loc = str(m.get("location_name") or "unknown location")
        mtype = str(m.get("media_type", "file"))
        fname = str(m.get("filename", ""))
        lines.append(f"- {mtype.title()} '{fname}' on {dt_str} at {loc}")

    context = f"The user has {len(media_list)} memories:\n" + "\n".join(lines)
    return context, media_list


def _row_to_media_response(m: dict[str, Any]) -> MediaResponse:
    return MediaResponse(
        id=str(m["id"]),
        vault_id=str(m["vault_id"]) if m.get("vault_id") else None,
        memory_id=str(m["memory_id"]) if m.get("memory_id") else None,
        owner_id=str(m.get("owner_id", "")),
        filename=str(m.get("filename", "")),
        storage_path=str(m.get("storage_path", "")),
        url=str(m.get("url", "")),
        thumbnail_url=str(m.get("thumbnail_url")) if m.get("thumbnail_url") else None,
        media_type=str(m.get("media_type", "image")),
        file_size=int(m.get("file_size", 0)),
        mime_type=str(m.get("mime_type")) if m.get("mime_type") else None,
        created_at=datetime.fromisoformat(str(m.get("created_at", datetime.now(timezone.utc).isoformat())).replace("Z", "+00:00")),
    )


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
        context, media_list = _build_memory_context(user_id)

        # 4. Call LLM
        llm_resp = _call_llm(message, context)
        resp_text = llm_resp.text

        # 5. Extract referenced / matched media items
        matched_media_dict: dict[str, dict[str, Any]] = {}
        resp_lower = resp_text.lower()
        msg_lower = message.lower()

        # 5a. Check direct filename mentions in response or user message
        for m in media_list:
            fname = str(m.get("filename", "")).lower()
            if fname and (fname in resp_lower or fname in msg_lower):
                matched_media_dict[str(m["id"])] = m

        # 5b. If query was a search/filter and few direct mentions, match by location, date, or query keywords
        is_query = any(k in msg_lower for k in ("show", "find", "pictures", "photos", "videos", "memories", "vibe", "party", "trip", "vacation", "beach", "look"))
        if is_query and len(matched_media_dict) < 9:
            for m in media_list:
                loc = str(m.get("location_name") or "").lower()
                dt = str(m.get("created_at") or "")[:10]
                # Match location or date keywords
                if loc and loc != "unknown location" and (loc in msg_lower or loc in resp_lower):
                    matched_media_dict[str(m["id"])] = m
                elif dt in msg_lower or dt in resp_lower:
                    matched_media_dict[str(m["id"])] = m

        # 5c. Limit to top 15 visual items
        matched_media_list = list(matched_media_dict.values())[:15]
        related_ids = [str(m["id"]) for m in matched_media_list]

        # 6. Save assistant message
        asst_msg_insert = supabase.table("ai_messages").insert({
            "conversation_id": conv_id,
            "role": "assistant",
            "content": resp_text,
            "related_memory_ids": related_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        if not asst_msg_insert.data:
            raise HTTPException(status_code=500, detail="Failed to save assistant message")
        asst_msg_row = cast(dict[str, Any], asst_msg_insert.data[0])

        # 7. Update conversation timestamp + title if new
        supabase.table("ai_conversations") \
            .update({"updated_at": datetime.now(timezone.utc).isoformat()}) \
            .eq("id", conv_id).execute()

        related_media_responses = [_row_to_media_response(m) for m in matched_media_list]

        def _to_msg(row: dict[str, Any], media_items: list[MediaResponse]) -> AIMessageResponse:
            return AIMessageResponse(
                id=str(row["id"]),
                conversation_id=conv_id,
                role=str(row["role"]),
                content=str(row["content"]),
                related_memory_ids=[str(x) for x in (row.get("related_memory_ids") or [])],
                related_media=media_items,
                created_at=datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")),
            )

        return AIChatResponse(
            conversation_id=conv_id,
            user_message=_to_msg(user_msg_row, []),
            assistant_message=_to_msg(asst_msg_row, related_media_responses),
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

        # Collect all related_memory_ids across messages
        all_media_ids = set()
        for r in rows:
            for mid in (r.get("related_memory_ids") or []):
                all_media_ids.add(str(mid))

        media_by_id: dict[str, MediaResponse] = {}
        if all_media_ids:
            try:
                m_res = supabase.table("media") \
                    .select("id, filename, media_type, created_at, location_name, vault_id, memory_id, owner_id, storage_path, url, thumbnail_url, file_size, mime_type") \
                    .in_("id", list(all_media_ids)).execute()
                for m_row in (m_res.data or []):
                    media_by_id[str(m_row["id"])] = _row_to_media_response(m_row)
            except Exception as e:
                logger.warning(f"Failed to fetch related media in get_messages: {e}")

        return [
            AIMessageResponse(
                id=str(r["id"]),
                conversation_id=conversation_id,
                role=str(r["role"]),
                content=str(r["content"]),
                related_memory_ids=[str(x) for x in (r.get("related_memory_ids") or [])],
                related_media=[media_by_id[str(x)] for x in (r.get("related_memory_ids") or []) if str(x) in media_by_id],
                created_at=datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")),
            )
            for r in rows
        ]

    @staticmethod
    def _curate_single_item_vlm(
        item: dict[str, Any],
        prompt_intent: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        VLM-First inspection for a single candidate image:
        1. Analyzes what is visibly present in the image.
        2. Identifies whether requested subject, landmark, or event context is present.
        3. Distinguishes direct visual evidence from generic scenery (like open oceans or plain hotel rooms).
        4. Produces structured output with confidence, relevance, and evidence.
        """
        item_id = item.get("id") or item.get("filename") or "unknown"
        filename = item.get("filename") or ""
        file_bytes = item.get("bytes") or b""
        mime_type = item.get("mime_type") or "image/jpeg"
        timestamp = item.get("timestamp")

        if not file_bytes or not mime_type.startswith("image/"):
            return {
                "id": item_id,
                "filename": filename,
                "relevance_score": 0.20,
                "reason": "Video / non-image media (needs manual review)",
                "matched_tags": [],
                "recommended": False,
                "timestamp": timestamp,
                "visual_description": None,
            }

        context_str = f"Title: {title}, Description: {description}" if (title or description) else ""
        vlm_prompt = (
            f"Memory intent:\n\"{prompt_intent}\"\n{context_str}\n\n"
            "Analyze the supplied image specifically in relation to this memory intent.\n"
            "1. Describe what is visibly present in the image.\n"
            "2. Identify whether the requested subject, landmark, person, activity, or event context is visibly present.\n"
            "3. Distinguish direct visual evidence from generic scenery (e.g. open ocean, generic room, sky without requested landmarks).\n"
            "4. If the requested subject is NOT visible, explicitly state that and classify relevance as 'none'.\n"
            "5. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "visual_description": "<concise description of visible content>",\n'
            '  "is_subject_visible": true/false,\n'
            '  "relevance": "high" | "uncertain" | "none",\n'
            '  "relevance_score": <float between 0.0 and 1.0>,\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "evidence": "<explanation of visual evidence or lack thereof>"\n'
            "}"
        )
        # Smart local fallback if VLM fails or is rate-limited
        prompt_words = [w.lower() for w in prompt_intent.replace('_', ' ').replace('-', ' ').split() if len(w) > 2]
        fn_clean = filename.lower()
        matched_words = [w for w in prompt_words if w in fn_clean]
        heuristic_score = 0.70 if matched_words else 0.60

        default_fallback = {
            "visual_description": f"Photo: {filename}",
            "is_subject_visible": bool(matched_words),
            "relevance": "high" if matched_words else "uncertain",
            "relevance_score": heuristic_score,
            "confidence": 0.60,
            "evidence": f"Matched prompt keywords in file ({', '.join(matched_words)})" if matched_words else "Candidate photo for memory collection."
        }

        try:
            vlm_res = call_structured_vlm(file_bytes, vlm_prompt, default_fallback=default_fallback)
        except Exception as e:
            logger.warning(f"VLM curation failed for {filename}: {e}")
            vlm_res = default_fallback

        relevance = str(vlm_res.get("relevance", "uncertain")).lower()
        rel_score = float(vlm_res.get("relevance_score", heuristic_score))
        is_subject_visible = bool(vlm_res.get("is_subject_visible", False))
        evidence = str(vlm_res.get("evidence") or "")
        visual_desc = str(vlm_res.get("visual_description") or "")

        # VLM recommendation logic:
        if relevance == "high" or is_subject_visible or rel_score >= 0.65:
            is_recommended = True
            rel_score = max(rel_score, 0.75)
            reason = evidence or f"Visually matches: {visual_desc}"
        elif relevance == "none" or (not is_subject_visible and rel_score < 0.35):
            is_recommended = False
            rel_score = min(rel_score, 0.15)
            reason = evidence or f"Requested subject not visible in {visual_desc}"
        else:
            is_recommended = bool(rel_score >= 0.50)
            reason = evidence or f"Contextual match: {visual_desc}"

        return {
            "id": item_id,
            "filename": filename,
            "relevance_score": round(rel_score, 2),
            "reason": reason,
            "matched_tags": [],
            "recommended": is_recommended,
            "timestamp": timestamp,
            "visual_description": visual_desc,
        }

    @staticmethod
    def curate_local_media(
        prompt: str,
        files_data: list[dict[str, Any]],  # list of {"id": str, "filename": str, "bytes": bytes, "mime_type": str, "timestamp": Optional[datetime]}
        title: Optional[str] = None,
        description: Optional[str] = None,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """
        VLM-First curation of local media files against a memory prompt/context:
        1. Dispatches actual image bytes to VLM with specific user memory intent.
        2. Evaluates visible evidence vs generic scenery false positives.
        3. Sets recommended=True ONLY when VLM confirms visual relevance.
        """
        import concurrent.futures

        if not files_data:
            return {
                "prompt": prompt,
                "suggested_title": title or (prompt.strip().title() if prompt else "New Memory"),
                "suggested_description": description,
                "suggested_date": None,
                "suggested_location": None,
                "matched_media": [],
                "total_candidates": 0,
                "total_matched": 0,
                "total_recommended": 0,
            }

        prompt_clean = prompt.strip()

        # Concurrent VLM visual curation across candidate items
        scored_items = []
        max_workers = min(len(files_data), 6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    AIService._curate_single_item_vlm,
                    item,
                    prompt_clean,
                    title,
                    description
                )
                for item in files_data
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    scored_items.append(res)
                except Exception as ex:
                    logger.warning(f"Error in curation future: {ex}")

        # If any items failed, fallback to original order
        if len(scored_items) < len(files_data):
            seen_ids = {it["id"] for it in scored_items}
            for item in files_data:
                iid = item.get("id") or item.get("filename") or "unknown"
                if iid not in seen_ids:
                    scored_items.append({
                        "id": iid,
                        "filename": item.get("filename"),
                        "relevance_score": 0.20,
                        "reason": "VLM analysis skipped",
                        "matched_tags": [],
                        "recommended": False,
                        "timestamp": item.get("timestamp"),
                    })

        # Sort with highest relevance first
        scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        recommended_count = sum(1 for it in scored_items if it["recommended"])

        # Determine suggested title & dates
        suggested_title = title or prompt_clean.title()
        suggested_desc = description or f"Curated memory collection for '{prompt_clean}'."
        suggested_date = None
        dates = [it["timestamp"] for it in scored_items if it.get("timestamp")]
        if dates:
            suggested_date = sorted(dates)[len(dates) // 2].isoformat()

        return {
            "prompt": prompt_clean,
            "suggested_title": suggested_title,
            "suggested_description": suggested_desc,
            "suggested_date": suggested_date,
            "suggested_location": None,
            "matched_media": [
                {
                    "id": it["id"],
                    "filename": it["filename"],
                    "relevance_score": it["relevance_score"],
                    "reason": it["reason"],
                    "matched_tags": it.get("matched_tags", []),
                    "recommended": it["recommended"],
                    "visual_description": it.get("visual_description"),
                }
                for it in scored_items
            ],
            "total_candidates": len(files_data),
            "total_matched": len(scored_items),
            "total_recommended": recommended_count,
        }

    @staticmethod
    def filter_media_rag(
        prompt: str,
        candidates: list[Any],
        top_k: int = 30,
        threshold: float = 0.15
    ) -> dict[str, Any]:
        """
        RAG retrieval over candidate media items using multi-factor ranking.
        """
        import re
        import numpy as np

        if not candidates:
            return {
                "prompt": prompt,
                "suggested_title": prompt.strip().title() if prompt else "New Memory",
                "suggested_description": None,
                "suggested_date": None,
                "suggested_location": None,
                "matched_media": [],
                "total_candidates": 0,
                "total_matched": 0,
                "total_recommended": 0,
            }

        prompt_clean = prompt.strip().lower()
        prompt_words = set(re.findall(r'\w+', prompt_clean))

        # 1. Extract temporal cues
        year_matches = re.findall(r'\b(19\d\d|20\d\d)\b', prompt_clean)
        target_years = set(int(y) for y in year_matches)
        
        months_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
            "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        target_months = {v for k, v in months_map.items() if k in prompt_words}

        # 2. Try loading CLIP text embedder
        clip_model = None
        prompt_embedding = None
        try:
            from app.services.ai_extractor import get_clip_model
            clip_model = get_clip_model()
            prompt_embedding = clip_model.encode(prompt)
            prompt_norm = np.linalg.norm(prompt_embedding)
            if prompt_norm > 0:
                prompt_embedding = prompt_embedding / prompt_norm
        except Exception as e:
            logger.warning(f"CLIP embedding unavailable for RAG filter, using lexical fallback: {e}")

        # 3. Score each candidate
        scored_items = []
        for cand in candidates:
            cand_dict = cand if isinstance(cand, dict) else cand.model_dump()
            c_id = cand_dict.get("id", "")
            fname = (cand_dict.get("filename") or "").lower()
            loc_name = (cand_dict.get("location_name") or "").lower()
            tags = [t.lower() for t in (cand_dict.get("tags") or [])]
            ts = cand_dict.get("timestamp")

            # A. Keyword / token matching
            cand_text = f"{fname} {loc_name} {' '.join(tags)}".lower()
            cand_words = set(re.findall(r'\w+', cand_text))
            
            common_words = prompt_words.intersection(cand_words)
            stop_words = {"my", "the", "a", "an", "in", "on", "at", "with", "and", "or", "of", "to", "for", "trip", "photos", "video"}
            meaningful_common = [w for w in common_words if w not in stop_words and len(w) > 2]
            lexical_score = min(len(meaningful_common) * 0.40 + (len(common_words) * 0.1), 0.9)

            # B. Temporal score
            temporal_score = 0.0
            dt_obj = None
            if ts:
                if isinstance(ts, str):
                    try:
                        dt_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        pass
                elif isinstance(ts, datetime):
                    dt_obj = ts

            if dt_obj:
                if target_years:
                    if dt_obj.year in target_years:
                        temporal_score += 0.45
                    else:
                        temporal_score -= 0.2
                if target_months:
                    if dt_obj.month in target_months:
                        temporal_score += 0.35

            # C. Semantic CLIP score
            semantic_score = 0.0
            if clip_model is not None and prompt_embedding is not None:
                try:
                    descriptive_text = f"a photo of {loc_name or fname} {' '.join(tags)}" if tags or loc_name else f"a memory of {fname}"
                    cand_embed = clip_model.encode(descriptive_text)
                    c_norm = np.linalg.norm(cand_embed)
                    if c_norm > 0:
                        cand_embed = cand_embed / c_norm
                        semantic_score = float(np.dot(prompt_embedding, cand_embed))
                except Exception:
                    semantic_score = 0.0

            # Combined relevance score (0.0 to 1.0)
            combined_score = (0.45 * lexical_score) + (0.35 * max(semantic_score, 0.0)) + (0.20 * max(temporal_score, 0.0))

            matched_tags_list = list(meaningful_common)
            if loc_name and loc_name in prompt_clean:
                matched_tags_list.append(loc_name)
            for t in tags:
                if t in prompt_clean and t not in matched_tags_list:
                    matched_tags_list.append(t)

            scored_items.append({
                "id": c_id,
                "filename": cand_dict.get("filename"),
                "relevance_score": round(min(max(combined_score, 0.0), 1.0), 4),
                "reason": f"Matched concepts: {', '.join(matched_tags_list)}" if matched_tags_list else "Chronological & context correlation",
                "matched_tags": matched_tags_list,
                "recommended": bool(combined_score >= 0.45),
                "timestamp": dt_obj,
                "location_name": loc_name or None
            })

        # 4. Sort by relevance
        scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        recommended_count = sum(1 for it in scored_items if it["recommended"])

        # 5. Infer suggested title, date, location
        suggested_title = prompt.strip().title()
        suggested_desc = f"Discovered memories matching prompt '{prompt}'."
        suggested_date = None
        suggested_loc = None

        matched_dates = [it["timestamp"] for it in scored_items if it.get("timestamp")]
        if matched_dates:
            suggested_date = sorted(dates := matched_dates)[len(dates) // 2].isoformat()

        matched_locs = [it["location_name"] for it in scored_items if it.get("location_name")]
        if matched_locs:
            suggested_loc = matched_locs[0].title()

        return {
            "prompt": prompt,
            "suggested_title": suggested_title,
            "suggested_description": suggested_desc,
            "suggested_date": suggested_date,
            "suggested_location": suggested_loc,
            "matched_media": [
                {
                    "id": it["id"],
                    "filename": it["filename"],
                    "relevance_score": it["relevance_score"],
                    "reason": it["reason"],
                    "matched_tags": it["matched_tags"],
                    "recommended": it["recommended"],
                }
                for it in scored_items
            ],
            "total_candidates": len(candidates),
            "total_matched": len(scored_items),
            "total_recommended": recommended_count,
        }

