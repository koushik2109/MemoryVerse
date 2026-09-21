from fastapi import APIRouter, Depends, File, Form, UploadFile, Response, Body
from app.schemas.domain import (
    AIChatRequest,
    AIChatResponse,
    AIConversationResponse,
    AIMessageResponse,
    FilterMediaRequest,
    FilterMediaResponse,
)
from app.services.ai_service import AIService
from app.services.tts_service import TTSService
from ai_engine.video_generation.tts_engine import EMOTION_PROFILES
from app.core.security import get_current_user, CurrentUser
from typing import List, Optional, Dict, Any

router = APIRouter(prefix="/ai", tags=["AI Memory Assistant"])


@router.post(
    "/chat",
    response_model=AIChatResponse,
    summary="Send a message to the AI memory assistant",
    description=(
        "Send a message and get an AI-generated response about your memories. "
        "Pass conversation_id to continue an existing conversation, or omit to start a new one."
    ),
)
async def chat(
    payload: AIChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    return AIService.chat(
        user_id=current_user.id,
        message=payload.message,
        conversation_id=payload.conversation_id,
    )


@router.post(
    "/filter-media",
    response_model=FilterMediaResponse,
    summary="Filter and rank discovered media candidates using RAG",
    description="Takes a natural user memory prompt and candidate media items, returning relevance rankings, reasons, and suggested title/date.",
)
async def filter_media(
    payload: FilterMediaRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    return AIService.filter_media_rag(
        prompt=payload.prompt,
        candidates=payload.candidates,
        top_k=payload.top_k or 30,
        threshold=payload.threshold or 0.15,
    )


@router.post(
    "/curate-local-media",
    response_model=FilterMediaResponse,
    summary="Visually curate local candidate media files against memory prompt",
    description="Accepts actual image/media files, computes visual CLIP embeddings, and determines which files belong to the memory.",
)
async def curate_local_media(
    prompt: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    threshold: Optional[float] = Form(0.50),
    files: List[UploadFile] = File(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append({
            "id": f.filename,
            "filename": f.filename,
            "bytes": content,
            "mime_type": f.content_type or "image/jpeg",
        })

    import asyncio
    return await asyncio.to_thread(
        AIService.curate_local_media,
        prompt=prompt,
        files_data=files_data,
        title=title,
        description=description,
        threshold=threshold or 0.50,
    )


@router.get(
    "/conversations",
    response_model=List[AIConversationResponse],
    summary="List AI conversations",
)
async def list_conversations(current_user: CurrentUser = Depends(get_current_user)):
    return AIService.list_conversations(current_user.id)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[AIMessageResponse],
    summary="Get messages in a conversation",
)
async def get_messages(
    conversation_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    return AIService.get_messages(current_user.id, conversation_id)


@router.post(
    "/multi-agent-flow",
    summary="Execute multi-agent memory synthesis & video direction flow",
    description="Invokes the full LangGraph cognitive architecture (Planner, Scorer, Storyteller, Auditor, Video Director).",
)
async def multi_agent_flow(
    prompt: str = Form(...),
    ablation_mode: Optional[str] = Form("model_4_full"),
    target_duration: Optional[float] = Form(30.0),
    current_user: CurrentUser = Depends(get_current_user),
):
    return AIService.run_multi_agent_flow(
        user_id=current_user.id,
        prompt=prompt,
        ablation_mode=ablation_mode or "model_4_full",
        target_duration=target_duration or 30.0,
    )


# ── Emotion-Aware Neural TTS Endpoints ────────────────────────────────────────

@router.post(
    "/tts",
    summary="Synthesize emotion-aware speech audio from text",
    description="Generates expressive speech with voice, pitch, and cadence tailored to the requested emotional vibe.",
)
async def synthesize_speech(
    payload: Dict[str, Any] = Body(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    text = payload.get("text", "").strip()
    if not text:
        return Response(status_code=400, content="Text is required for TTS synthesis.")

    emotion = payload.get("emotion", "auto")
    voice = payload.get("voice")

    audio_bytes, mime_type, duration = await TTSService.synthesize_speech(
        text=text,
        emotion=emotion,
        voice=voice,
    )

    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Emotion": emotion,
            "Content-Disposition": "inline; filename=speech.mp3",
        },
    )


@router.get(
    "/memory/{memory_id}/narrate",
    summary="Generate an emotion-aware spoken audio story for a memory",
    description="Synthesizes a rich spoken audio narration describing the memory and its captured moments.",
)
async def narrate_memory(
    memory_id: str,
    mood: Optional[str] = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        audio_bytes, mime_type, script, duration = await TTSService.narrate_memory(
            memory_id=memory_id,
            user_id=current_user.id,
            mood=mood,
        )

        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "X-Audio-Duration": f"{duration:.2f}",
                "X-Narration-Length": str(len(script)),
                "Content-Disposition": f"inline; filename=narration_{memory_id[:8]}.mp3",
            },
        )
    except ValueError as e:
        return Response(status_code=404, content=str(e))


@router.get(
    "/emotions",
    summary="Get all available emotion presets and voice descriptions",
)
async def get_emotion_profiles():
    return {
        "success": True,
        "emotions": [
            {
                "emotion": k,
                "voice": v["voice"],
                "rate": v["rate"],
                "pitch": v["pitch"],
                "description": v["description"],
            }
            for k, v in EMOTION_PROFILES.items()
        ],
    }
