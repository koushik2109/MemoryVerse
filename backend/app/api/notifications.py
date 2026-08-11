from fastapi import APIRouter, Depends, Query
from app.schemas.domain import NotificationResponse
from app.services.notification_service import NotificationService
from app.core.security import get_current_user, CurrentUser
from typing import List

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("", response_model=List[NotificationResponse])
async def list_notifications(
    limit: int = Query(50, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user)
):
    return NotificationService.get_notifications(current_user.id, limit=limit)

@router.post("/{notification_id}/read")
async def mark_read(notification_id: str, current_user: CurrentUser = Depends(get_current_user)):
    NotificationService.mark_as_read(notification_id, current_user.id)
    return {"message": "Notification marked as read"}

@router.post("/read-all")
async def mark_all_read(current_user: CurrentUser = Depends(get_current_user)):
    NotificationService.mark_all_as_read(current_user.id)
    return {"message": "All notifications marked as read"}
