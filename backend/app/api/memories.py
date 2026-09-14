from fastapi import APIRouter, Depends, Query, status
from typing import List, Optional
from datetime import datetime
import logging

from app.schemas.domain import (
    MemoryCreate, MemoryUpdate, MemoryResponse, ClusterPreviewResponse, ClusterExecutionResponse,
    MemoryNarrativeResponse, AnalyzeMemoryRequest
)
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


@router.get("/auto-cluster/preview", response_model=ClusterPreviewResponse, status_code=status.HTTP_200_OK)
@router.post("/auto-cluster/preview", response_model=ClusterPreviewResponse, status_code=status.HTTP_200_OK)
async def preview_auto_cluster_memories(
    vault_id: Optional[str] = Query(None, description="Preview clustering for a specific vault"),
    user: CurrentUser = Depends(get_current_user)
):
    """
    Preview automatic event clustering for unassociated media without creating memories.
    """
    from app.services.event_clustering_service import EventClusteringService
    return await EventClusteringService.preview_clusters_for_user(user.id, vault_id=vault_id)


@router.post("/auto-cluster", response_model=ClusterExecutionResponse, status_code=status.HTTP_200_OK)
async def auto_cluster_memories(
    vault_id: Optional[str] = Query(None, description="Trigger clustering for a specific vault"),
    user: CurrentUser = Depends(get_current_user)
):
    """
    Trigger automatic event detection and clustering for the user's unassociated media.
    Groups photos/videos chronologically and semantically into logical Memories.
    """
    from app.services.event_clustering_service import EventClusteringService
    return await EventClusteringService.cluster_and_create_memories(user.id, vault_id=vault_id)


# ── STEP 8: MEMORY NARRATIVE INTELLIGENCE ENDPOINTS ──────────────────────────

@router.post("/{memory_id}/analyze", response_model=MemoryNarrativeResponse, status_code=status.HTTP_200_OK)
async def analyze_memory_narrative(
    memory_id: str,
    payload: Optional[AnalyzeMemoryRequest] = None,
    user: CurrentUser = Depends(get_current_user)
):
    """
    Generate or regenerate grounded narrative intelligence (timeline, key moments, atmosphere, highlights)
    and persist results in memories.metadata without overwriting user custom titles.
    """
    from app.services.memory_narrative_service import MemoryNarrativeService
    force = payload.force_regenerate if payload else False
    return await MemoryNarrativeService.analyze_and_persist_memory(
        memory_id=memory_id,
        user_id=user.id,
        force_regenerate=force
    )


@router.get("/{memory_id}/narrative", response_model=MemoryNarrativeResponse, status_code=status.HTTP_200_OK)
async def get_memory_narrative(
    memory_id: str,
    user: CurrentUser = Depends(get_current_user)
):
    """
    Retrieve structured narrative intelligence, timeline phases, key moments, and atmosphere for a memory.
    """
    from app.services.memory_narrative_service import MemoryNarrativeService
    return await MemoryNarrativeService.analyze_and_persist_memory(
        memory_id=memory_id,
        user_id=user.id,
        force_regenerate=False
    )


@router.post("/narrative/preview", response_model=MemoryNarrativeResponse, status_code=status.HTTP_200_OK)
async def preview_memory_narrative(
    items_data: List[dict],
    user: CurrentUser = Depends(get_current_user)
):
    """
    Dry-run narrative analysis on a list of media item contexts without mutating database records.
    """
    from app.services.memory_narrative_service import MemoryNarrativeService
    from app.services.event_clustering_service import MediaItemContext

    contexts = []
    for idx, d in enumerate(items_data):
        dt_str = d.get("taken_at") or d.get("timestamp") or d.get("created_at")
        dt = None
        if dt_str:
            try:
                dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
            except Exception:
                pass
        
        contexts.append(MediaItemContext(
            id=d.get("id") or f"preview_{idx}",
            filename=d.get("filename") or f"media_{idx}.jpg",
            timestamp=dt,
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            location_name=d.get("location_name"),
            scenes=d.get("scenes", []),
            objects=d.get("objects", []),
            people_count=d.get("people_count"),
            vlm_description=d.get("vlm_description"),
            quality_score=float(d.get("quality_score", 0.80)),
            media_type=d.get("media_type") or "image"
        ))

    return MemoryNarrativeService.analyze_memory_event(items=contexts, enable_llm=True)


