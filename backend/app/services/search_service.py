from app.core.db import get_supabase_client
from app.schemas.domain import GlobalSearchResponse, VaultResponse, MediaResponse, UserProfile
from app.services.vault_service import VaultService
from typing import cast, Any

class SearchService:
    @staticmethod
    def global_search(query: str, user_id: str) -> GlobalSearchResponse:
        if not query or len(query.strip()) == 0:
            return GlobalSearchResponse(vaults=[], media=[], collaborators=[])

        q = f"%{query.strip()}%"
        supabase = get_supabase_client()

        # 1. Vaults matching query
        user_vaults = VaultService.get_user_vaults(user_id)
        matching_vaults = [v for v in user_vaults if query.lower() in v.name.lower() or (v.description and query.lower() in v.description.lower())]

        # 2. Media matching filename
        m_res = supabase.table("media").select("*").eq("owner_id", user_id).ilike("filename", q).limit(20).execute()
        matching_media = [
            MediaResponse(
                id=m["id"],
                vault_id=m.get("vault_id"),
                owner_id=m["owner_id"],
                filename=m["filename"],
                storage_path=m["storage_path"],
                url=m["url"],
                thumbnail_url=m.get("thumbnail_url"),
                media_type=m["media_type"],
                file_size=m["file_size"],
                mime_type=m.get("mime_type"),
                created_at=m["created_at"]
            ) for m in (cast(list[dict[str, Any]], m_res.data or []))
        ]

        # 3. Collaborators matching name or email
        p_res = supabase.table("profiles").select("*").or_(f"full_name.ilike.{q},email.ilike.{q}").limit(10).execute()
        collaborators = [
            UserProfile(
                id=p["id"],
                email=p["email"],
                full_name=p.get("full_name"),
                username=p.get("username"),
                avatar_url=p.get("avatar_url"),
                bio=p.get("bio")
            ) for p in (cast(list[dict[str, Any]], p_res.data or [])) if p["id"] != user_id
        ]

        return GlobalSearchResponse(
            vaults=matching_vaults,
            media=matching_media,
            collaborators=collaborators
        )
