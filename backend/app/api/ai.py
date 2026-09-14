from fastapi import APIRouter, Depends, File, Form, UploadFile
from app.schemas.domain import (
    AIChatRequest,
    AIChatResponse,
    AIConversationResponse,
    AIMessageResponse,
    FilterMediaRequest,
    FilterMediaResponse,
)
from app.services.ai_service import AIService
from app.core.security import get_current_user, CurrentUser
from typing import List, Optional

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

    return AIService.curate_local_media(
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


