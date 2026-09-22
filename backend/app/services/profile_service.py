from app.core.db import get_supabase_client
from app.schemas.domain import UserProfile, ProfileUpdate
from fastapi import HTTPException, status
from typing import cast, Any
from postgrest.types import CountMethod
from datetime import datetime

class ProfileService:
    @staticmethod
    def get_profile(user_id: str, email: str = "") -> UserProfile:
        from app.core.cache import get_cache, set_cache
        cache_key = f"profile:{user_id}"
        cached = get_cache(cache_key)
        if cached is not None:
            return cached

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

        # Calculate counts concurrently for instant response
        from concurrent.futures import ThreadPoolExecutor

        def _count_vaults():
            try:
                return supabase.table("vault_members").select("id", count=cast(Any, CountMethod.exact)).eq("user_id", user_id).execute().count or 0
            except Exception:
                return 0

        def _count_media():
            try:
                return supabase.table("media").select("id", count=cast(Any, CountMethod.exact)).eq("owner_id", user_id).execute().count or 0
            except Exception:
                return 0

        def _count_memories():
            try:
                return supabase.table("memories").select("id", count=cast(Any, CountMethod.exact)).eq("owner_id", user_id).execute().count or 0
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_v = pool.submit(_count_vaults)
            f_m = pool.submit(_count_media)
            f_mem = pool.submit(_count_memories)
            vault_cnt = f_v.result()
            media_cnt = f_m.result()
            memory_cnt = f_mem.result()

        result = UserProfile(
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
        set_cache(cache_key, result, ttl_seconds=30.0)
        return result

    @staticmethod
    def update_profile(user_id: str, payload: ProfileUpdate) -> UserProfile:
        from app.core.cache import invalidate_user_cache
        invalidate_user_cache(user_id)
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
