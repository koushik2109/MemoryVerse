from app.core.db import get_supabase_client
from app.schemas.domain import NotificationResponse
from typing import cast, Any

class NotificationService:
    @staticmethod
    def get_notifications(user_id: str, limit: int = 50) -> list[NotificationResponse]:
        supabase = get_supabase_client()
        res = supabase.table("notifications").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        items = cast(list[dict[str, Any]], res.data or [])
        return [
            NotificationResponse(
                id=n["id"],
                title=n["title"],
                message=n["message"],
                type=n["type"],
                data=n.get("data", {}),
                is_read=n.get("is_read", False),
                created_at=n["created_at"]
            ) for n in items
        ]

    @staticmethod
    def mark_as_read(notification_id: str, user_id: str) -> bool:
        supabase = get_supabase_client()
        supabase.table("notifications").update({"is_read": True}).eq("id", notification_id).eq("user_id", user_id).execute()
        return True

    @staticmethod
    def mark_all_as_read(user_id: str) -> bool:
        supabase = get_supabase_client()
        supabase.table("notifications").update({"is_read": True}).eq("user_id", user_id).execute()
        return True
