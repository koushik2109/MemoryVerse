"""
MemoryVerse - AI Sessions API Endpoints
Provides endpoints for conversational AI Memory Director sessions, natural language guidance,
explicit approval video generation, and selective revisions.
"""
import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import CurrentUser, get_current_user
from app.schemas.ai_session import (
    AISessionCreateRequest,
    AISessionResponse,
    GenerateVideoRequest,
    RevisionRequest,
    UserMessageRequest,
)
from app.services.ai_director_service import ai_director_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-sessions", tags=["AI Memory Director"])


@router.post(
    "",
    response_model=AISessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Initialize or resume an AI Director session for a memory",
)
async def create_or_get_ai_session(
    payload: AISessionCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        return ai_director_service.create_or_get_session(
            user_id=current_user.id,
            memory_id=payload.memory_id,
            initial_prompt=payload.initial_prompt,
        )
    except Exception as e:
        logger.exception(f"Failed to create AI session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initialize AI Director session: {str(e)}",
        )


@router.get(
    "/{session_id}",
    response_model=AISessionResponse,
    summary="Get active AI Director session state, specification, and messages",
)
async def get_ai_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    session = ai_director_service.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI Director session not found or unauthorized.",
        )
    return session


@router.post(
    "/{session_id}/messages",
    response_model=AISessionResponse,
    summary="Send a message to the AI Memory Director to refine the video specification",
)
async def send_session_message(
    session_id: str,
    payload: UserMessageRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        return ai_director_service.process_user_message(
            session_id=session_id,
            user_id=current_user.id,
            user_text=payload.message,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as e:
        logger.exception(f"Error handling AI message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process director message.",
        )


@router.post(
    "/{session_id}/generate",
    summary="Explicitly approve VideoSpecification and launch durable video generation job",
)
async def approve_and_generate_video(
    session_id: str,
    payload: GenerateVideoRequest = GenerateVideoRequest(),
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        session, job_id = ai_director_service.approve_and_generate(
            session_id=session_id,
            user_id=current_user.id,
            custom_spec=payload.specification,
        )
        return {
            "success": True,
            "job_id": job_id,
            "session": session,
            "status": "queued",
        }
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as e:
        logger.exception(f"Failed to launch video generation from session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate video generation.",
        )


@router.post(
    "/{session_id}/revision",
    summary="Request a natural language revision and compute selective stage invalidation",
)
async def request_video_revision(
    session_id: str,
    payload: RevisionRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        session, revision = ai_director_service.request_revision(
            session_id=session_id,
            user_id=current_user.id,
            revision_prompt=payload.prompt,
        )
        return {
            "success": True,
            "session": session,
            "revision": revision,
        }
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as e:
        logger.exception(f"Failed to compute revision: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process revision request.",
        )
