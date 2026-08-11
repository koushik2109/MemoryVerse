from fastapi import APIRouter, Depends, Query, status
from typing import List, Optional
import logging

from app.schemas.domain import MemoryCreate, MemoryUpdate, MemoryResponse
from app.services.memory_service import MemoryService
from app.core.security import get_current_user, CurrentUser
from app.core.db import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memories", tags=["Memories"])

def get_memory_service(client = Depends(get_supabase_client)) -> MemoryService:
    return MemoryService(client)

@router.get("", response_model=List[MemoryResponse])
async def list_memories(
    vault_id: Optional[str] = Query(None, description="Filter by vault ID"),
    limit: int = Query(50, ge=1, le=100),
    user: CurrentUser = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service)
):
    """Get all memories for the current user, optionally filtered by vault."""
    memories = await service.get_memories(user.id, vault_id, limit)
    
    # Format response: Supabase joins media, we map it back
    for m in memories:
        m['media_count'] = len(m.get('media', []))
        
    return memories

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory: MemoryCreate,
    user: CurrentUser = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service)
):
    """Create a new memory container."""
    data = await service.create_memory(user.id, memory)
    data['media'] = []
    data['media_count'] = 0
    return data

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    user: CurrentUser = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service)
):
    """Get a specific memory and its media."""
    data = await service.get_memory(user.id, memory_id)
    data['media_count'] = len(data.get('media', []))
    return data

@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    memory: MemoryUpdate,
    user: CurrentUser = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service)
):
    """Update a memory's details."""
    data = await service.update_memory(user.id, memory_id, memory)
    # Re-fetch with media to return full response
    full_data = await service.get_memory(user.id, memory_id)
    full_data['media_count'] = len(full_data.get('media', []))
    return full_data

@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    user: CurrentUser = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service)
):
    """Delete a memory and cascade delete its media (handled by DB)."""
    await service.delete_memory(user.id, memory_id)
