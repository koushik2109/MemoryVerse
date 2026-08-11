from fastapi import APIRouter, Depends, Query
from app.schemas.domain import GlobalSearchResponse
from app.services.search_service import SearchService
from app.core.security import get_current_user, CurrentUser

router = APIRouter(prefix="/search", tags=["Search"])

@router.get("", response_model=GlobalSearchResponse)
async def search(
    q: str = Query("", description="Query string for vaults, media, collaborators"),
    current_user: CurrentUser = Depends(get_current_user)
):
    return SearchService.global_search(q, current_user.id)
