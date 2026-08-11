from fastapi import APIRouter, Depends
from app.schemas.domain import AIChatRequest, AIChatResponse, AIConversationResponse, AIMessageResponse
from app.services.ai_service import AIService
from app.core.security import get_current_user, CurrentUser
from typing import List

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

