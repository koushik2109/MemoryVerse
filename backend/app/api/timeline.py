from fastapi import APIRouter, Depends, Query
from app.schemas.domain import TimelineResponse
from app.services.timeline_service import TimelineService
from app.core.security import get_current_user, CurrentUser
from typing import Optional

router = APIRouter(prefix="/timeline", tags=["Timeline"])


@router.get(
    "",
    response_model=TimelineResponse,
    summary="Get user timeline",
    description="Returns user's media grouped chronologically by year → month → week.",
)
async def get_timeline(
    vault_id: Optional[str] = Query(None, description="Filter by specific vault"),
    current_user: CurrentUser = Depends(get_current_user),
):
    return TimelineService.get_timeline(current_user.id, vault_id=vault_id)
