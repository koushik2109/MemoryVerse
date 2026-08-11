from fastapi import APIRouter, Depends
from app.schemas.domain import UserProfile, ProfileUpdate
from app.services.profile_service import ProfileService
from app.core.security import get_current_user, CurrentUser

router = APIRouter(prefix="/profile", tags=["Profile"])

@router.get("", response_model=UserProfile)
async def get_profile(current_user: CurrentUser = Depends(get_current_user)):
    return ProfileService.get_profile(current_user.id, current_user.email)

@router.put("", response_model=UserProfile)
async def update_profile(payload: ProfileUpdate, current_user: CurrentUser = Depends(get_current_user)):
    return ProfileService.update_profile(current_user.id, payload)

@router.delete("/account")
async def delete_account(current_user: CurrentUser = Depends(get_current_user)):
    ProfileService.delete_account(current_user.id)
    return {"message": "Account successfully deleted"}
