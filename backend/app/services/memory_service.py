import logging
from typing import List, Optional, cast, Any
from fastapi import HTTPException, status
from postgrest.exceptions import APIError
from app.schemas.domain import MemoryCreate, MemoryUpdate, MemoryResponse

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self, supabase_client):
        self.client = supabase_client

    async def get_memories(self, user_id: str, vault_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        query = self.client.table("memories").select("*, media!media_memory_id_fkey(*)")
        if vault_id:
            query = query.eq("vault_id", vault_id)
        else:
            query = query.eq("owner_id", user_id)
            
        try:
            res = query.order("memory_date", desc=True).limit(limit).execute()
            return cast(list[dict[str, Any]], res.data)
        except APIError as e:
            logger.error(f"Error fetching memories: {e}")
            return []

    async def get_memory(self, user_id: str, memory_id: str) -> dict:
        try:
            res = self.client.table("memories").select("*, media!media_memory_id_fkey(*)").eq("id", memory_id).execute()
            if not res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
            
            memory = cast(list[dict[str, Any]], res.data)[0]
            # Simple check: user must be owner or (todo: vault member check)
            # RLS handles most of this but since service role might bypass, we'll let RLS do it if using user client
            # Or if using service role, we should enforce logic.
            return memory
        except APIError as e:
            logger.error(f"Error fetching memory {memory_id}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch memory")

    async def create_memory(self, user_id: str, memory: MemoryCreate) -> dict:
        data = memory.model_dump(mode="json", exclude_unset=True)
        data["owner_id"] = user_id
        
        try:
            res = self.client.table("memories").insert(data).execute()
            if not res.data:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create memory")
            return cast(list[dict[str, Any]], res.data)[0]
        except APIError as e:
            logger.error(f"Error creating memory: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def update_memory(self, user_id: str, memory_id: str, memory: MemoryUpdate) -> dict:
        data = memory.model_dump(mode="json", exclude_unset=True)
        if not data:
            return await self.get_memory(user_id, memory_id)
            
        try:
            # First check ownership (or let RLS handle it)
            res = self.client.table("memories").update(data).eq("id", memory_id).eq("owner_id", user_id).execute()
            if not res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found or unauthorized")
            return cast(list[dict[str, Any]], res.data)[0]
        except APIError as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def delete_memory(self, user_id: str, memory_id: str) -> None:
        try:
            res = self.client.table("memories").delete().eq("id", memory_id).eq("owner_id", user_id).execute()
            if not res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found or unauthorized")
        except APIError as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete memory")
