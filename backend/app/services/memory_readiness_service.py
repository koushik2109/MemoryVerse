"""
MemoryReadinessService - Evaluates whether a memory has sufficient, processed,
and indexed media assets to generate a high-fidelity video reel.
"""

import logging
from typing import List, Optional, Dict, Any, cast, Set
from pydantic import BaseModel
from app.core.db import get_supabase_client
from app.services.job_state_manager import job_state_manager

logger = logging.getLogger(__name__)


class MediaReadinessStats(BaseModel):
    total_media: int = 0
    photos_count: int = 0
    videos_count: int = 0
    processed_count: int = 0
    pending_count: int = 0
    failed_count: int = 0
    indexed_count: int = 0


class MemoryReadinessResponse(BaseModel):
    memory_id: str
    memory_title: str
    stats: MediaReadinessStats
    is_ready_for_video: bool
    status_summary: str
    active_job_id: Optional[str] = None
    active_job_status: Optional[str] = None
    active_job_progress: Optional[int] = None
    blockers: List[str] = []
    recommendations: List[str] = []


class MemoryReadinessService:
    @staticmethod
    def evaluate_memory_readiness(user_id: str, memory_id: str) -> MemoryReadinessResponse:
        supabase = get_supabase_client()

        # 1. Fetch memory record with ownership check
        mem_res = supabase.table("memories").select("id, title, owner_id").eq("id", memory_id).execute()
        raw_mems: list[dict[str, Any]] = cast(list[dict[str, Any]], mem_res.data or [])  # type: ignore
        if not raw_mems:
            raise ValueError(f"Memory with ID '{memory_id}' not found.")
        
        mem: dict[str, Any] = raw_mems[0]
        if mem.get("owner_id") != user_id:
            raise PermissionError("Unauthorized access to memory.")

        memory_title = str(mem.get("title") or "Untitled Memory")

        # 2. Fetch associated media items
        media_res = supabase.table("media").select(
            "id, media_type, storage_path, url, metadata, created_at"
        ).eq("memory_id", memory_id).execute()

        media_items: list[dict[str, Any]] = cast(list[dict[str, Any]], media_res.data or [])  # type: ignore
        total_media = len(media_items)
        photos_count = sum(1 for m in media_items if m.get("media_type") == "image")
        videos_count = sum(1 for m in media_items if m.get("media_type") == "video")

        # 3. Check processing state of media items
        processed_count = 0
        pending_count = 0
        failed_count = 0

        media_ids: list[str] = [str(m.get("id")) for m in media_items if m.get("id")]

        # Check embedding store for indexed media
        indexed_ids: set[str] = set()
        if media_ids:
            try:
                emb_res = supabase.table("media_embeddings").select("media_id").in_("media_id", media_ids).execute()
                raw_embs: list[dict[str, Any]] = cast(list[dict[str, Any]], emb_res.data or [])  # type: ignore
                if raw_embs:
                    indexed_ids = {str(row.get("media_id")) for row in raw_embs if row.get("media_id")}
            except Exception as e:
                logger.warning(f"Could not query media_embeddings for readiness check: {e}")

        for m in media_items:
            meta: dict[str, Any] = cast(dict[str, Any], m.get("metadata") or {})  # type: ignore[redundant-cast]
            mid: str = str(m.get("id") or "")
            is_processed = bool(
                meta.get("processed_at")
                or meta.get("ai_tags")
                or mid in indexed_ids
            )
            is_failed = meta.get("processing_status") == "failed"

            if is_failed:
                failed_count += 1
            elif is_processed:
                processed_count += 1
            else:
                # If it has a valid url/storage_path, it is at least uploaded
                if m.get("storage_path") or m.get("url"):
                    processed_count += 1
                else:
                    pending_count += 1

        indexed_count = len(indexed_ids)

        stats = MediaReadinessStats(
            total_media=total_media,
            photos_count=photos_count,
            videos_count=videos_count,
            processed_count=processed_count,
            pending_count=pending_count,
            failed_count=failed_count,
            indexed_count=indexed_count,
        )

        # 4. Check for active video generation jobs
        active_job = None
        active_job_id = None
        active_job_status = None
        active_job_progress = None

        user_jobs = job_state_manager.get_user_jobs(user_id)
        for job in user_jobs:
            _sv = getattr(job.status, "value", None)
            status_val: str = _sv if isinstance(_sv, str) else str(job.status)
            if job.memory_id == memory_id and status_val in ["queued", "processing", "retrying"]:
                active_job = job
                active_job_id = job.job_id
                active_job_status = status_val
                active_job_progress = job.overall_progress
                break

        # 5. Evaluate Readiness & Blockers
        blockers = []
        recommendations = []

        if total_media == 0:
            blockers.append("Memory has no photos or videos.")
            recommendations.append("Upload at least 3 photos to generate a video reel.")
        elif total_media < 3:
            blockers.append(f"Memory has only {total_media} media item(s). Minimum recommended is 3.")
            recommendations.append("Add more photos or videos for dynamic transitions and pacing.")

        if pending_count > 0:
            recommendations.append(f"{pending_count} media item(s) are still processing in the background.")

        if failed_count > 0:
            recommendations.append(f"{failed_count} item(s) failed processing, but remaining items can still be used.")

        if active_job:
            blockers.append(f"A video generation job ({active_job_id}) is currently in progress ({active_job_progress}%).")

        is_ready = len(blockers) == 0 and total_media >= 3

        if is_ready:
            status_summary = (
                f"Ready for Video: {total_media} moments available "
                f"({photos_count} photos, {videos_count} videos). "
                f"{processed_count}/{total_media} processed."
            )
        elif active_job:
            status_summary = (
                f"Video Generation in Progress: {active_job_progress}% "
                f"({active_job.current_task or active_job.stage})."
            )
        else:
            status_summary = f"Not Ready for Video: {'; '.join(blockers)}"

        return MemoryReadinessResponse(
            memory_id=memory_id,
            memory_title=memory_title,
            stats=stats,
            is_ready_for_video=is_ready,
            status_summary=status_summary,
            active_job_id=active_job_id,
            active_job_status=active_job_status,
            active_job_progress=active_job_progress,
            blockers=blockers,
            recommendations=recommendations,
        )


memory_readiness_service = MemoryReadinessService()
