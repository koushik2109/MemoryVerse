from app.core.db import get_supabase_client
from app.schemas.domain import MediaCreate, MediaResponse
from fastapi import HTTPException, status
from datetime import datetime, timezone
from typing import cast, Any

class MediaService:
    @staticmethod
    def create_media(user_id: str, payload: MediaCreate) -> MediaResponse:
        supabase = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()
        media_data = {
            "vault_id": payload.vault_id,
            "memory_id": payload.memory_id,
            "owner_id": user_id,
            "filename": payload.filename,
            "storage_path": payload.storage_path,
            "url": payload.url,
            "thumbnail_url": payload.thumbnail_url or payload.url,
            "media_type": payload.media_type,
            "file_size": payload.file_size,
            "mime_type": payload.mime_type,
            "width": payload.width,
            "height": payload.height,
            "duration": payload.duration,
            "metadata": payload.metadata or {},
            "created_at": now
        }
        res = supabase.table("media").insert(media_data).execute()
        if not res.data:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save media metadata")
        
        m = cast(list[dict[str, Any]], res.data)[0]
        
        # If media belongs to a memory, check if memory needs a cover image
        if payload.memory_id:
            try:
                mem_res = supabase.table("memories").select("cover_media_id").eq("id", payload.memory_id).execute()
                if mem_res.data and not mem_res.data[0].get("cover_media_id"):
                    supabase.table("memories").update({"cover_media_id": m["id"]}).eq("id", payload.memory_id).execute()
            except Exception:
                pass

        return MediaResponse(
            id=m["id"],
            vault_id=m.get("vault_id"),
            memory_id=m.get("memory_id"),
            owner_id=m["owner_id"],
            filename=m["filename"],
            storage_path=m["storage_path"],
            url=m["url"],
            thumbnail_url=m.get("thumbnail_url"),
            media_type=m["media_type"],
            file_size=m["file_size"],
            mime_type=m.get("mime_type"),
            created_at=m["created_at"]
        )

    @staticmethod
    def get_user_media(user_id: str, vault_id: str | None = None, limit: int = 50) -> list[MediaResponse]:
        supabase = get_supabase_client()
        query = supabase.table("media").select("*").eq("owner_id", user_id)
        if vault_id:
            query = query.eq("vault_id", vault_id)
        
        res = query.order("created_at", desc=True).limit(limit).execute()
        media_list = cast(list[dict[str, Any]], res.data or [])

        return [
            MediaResponse(
                id=m["id"],
                vault_id=m.get("vault_id"),
                memory_id=m.get("memory_id"),
                owner_id=m["owner_id"],
                filename=m["filename"],
                storage_path=m["storage_path"],
                url=m["url"],
                thumbnail_url=m.get("thumbnail_url"),
                media_type=m["media_type"],
                file_size=m["file_size"],
                mime_type=m.get("mime_type"),
                created_at=m["created_at"]
            ) for m in media_list
        ]

    @staticmethod
    def delete_media(media_id: str, user_id: str) -> bool:
        supabase = get_supabase_client()
        res = supabase.table("media").select("*").eq("id", media_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Media not found")
        m = cast(list[dict[str, Any]], res.data)[0]
        if m["owner_id"] != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Delete from storage if path exists
        if m.get("storage_path"):
            try:
                supabase.storage.from_("memories").remove([m["storage_path"]])
            except Exception:
                pass
        
        supabase.table("media").delete().eq("id", media_id).execute()
        return True
