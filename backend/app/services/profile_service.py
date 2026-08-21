from app.core.db import get_supabase_client
from app.schemas.domain import UserProfile, ProfileUpdate
from fastapi import HTTPException, status
from typing import cast, Any
from postgrest.types import CountMethod
from datetime import datetime

class ProfileService:
    @staticmethod
    def get_profile(user_id: str, email: str = "") -> UserProfile:
        supabase = get_supabase_client()
        res = supabase.table("profiles").select("*").eq("id", user_id).execute()
        if not res.data:
            # Auto create profile row if missing
            new_profile = {
                "id": user_id,
                "email": email or f"user_{user_id[:8]}@example.com",
                "full_name": email.split("@")[0] if email else "User",
            }
            supabase.table("profiles").insert(new_profile).execute()
            prof = new_profile
        else:
            prof = cast(list[dict[str, Any]], res.data)[0]

        # Calculate counts
        vault_cnt = supabase.table("vault_members").select("id", count="exact").eq("user_id", user_id).execute().count or 0  # type: ignore
        media_cnt = supabase.table("media").select("id", count="exact").eq("owner_id", user_id).execute().count or 0  # type: ignore
        memory_cnt = supabase.table("memories").select("id", count="exact").eq("owner_id", user_id).execute().count or 0  # type: ignore

        return UserProfile(
            id=prof["id"],
            email=prof.get("email") or email,
            full_name=prof.get("full_name"),
            username=prof.get("username"),
            avatar_url=prof.get("avatar_url"),
            bio=prof.get("bio"),
            vault_count=vault_cnt,
            media_count=media_cnt,
            memory_count=memory_cnt,
            created_at=datetime.fromisoformat(cast(str, prof.get("created_at")).replace("Z", "+00:00")) if prof.get("created_at") else None
        )

    @staticmethod
    def update_profile(user_id: str, payload: ProfileUpdate) -> UserProfile:
        supabase = get_supabase_client()
        update_dict = {}
        if payload.full_name is not None: update_dict["full_name"] = payload.full_name
        if payload.username is not None: update_dict["username"] = payload.username
        if payload.avatar_url is not None: update_dict["avatar_url"] = payload.avatar_url
        if payload.bio is not None: update_dict["bio"] = payload.bio

        if update_dict:
            supabase.table("profiles").update(update_dict).eq("id", user_id).execute()

        return ProfileService.get_profile(user_id)

    @staticmethod
    def delete_account(user_id: str) -> bool:
        supabase = get_supabase_client()
        # Delete user's media, vault memberships, and profile
        supabase.table("media").delete().eq("owner_id", user_id).execute()
        supabase.table("vault_members").delete().eq("user_id", user_id).execute()
        supabase.table("notifications").delete().eq("user_id", user_id).execute()
        supabase.table("profiles").delete().eq("id", user_id).execute()
        try:
            supabase.auth.admin.delete_user(user_id)
        except Exception:
            pass
        return True
