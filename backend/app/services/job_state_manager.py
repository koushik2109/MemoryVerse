"""
MemoryVerse - Video Job State Machine & Persistence Manager
Manages job lifecycle, stage transitions, progress normalization (0-100%),
real ETA estimation, worker heartbeats, crash recovery, and telemetry metrics.
"""
import copy
import json
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from app.core.cache_service import cache_service
from app.core.db import get_supabase_client

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    ANALYZING_MEDIA = "analyzing_media"
    RETRIEVING_MEMORY = "retrieving_memory"
    DETECTING_EMOTION = "detecting_emotion"
    SELECTING_MEDIA = "selecting_media"
    PLANNING_STORY = "planning_story"
    GENERATING_MOTION = "generating_motion"
    GENERATING_AUDIO = "generating_audio"
    COMPOSING_VIDEO = "composing_video"
    ENCODING_VIDEO = "encoding_video"
    UPLOADING = "uploading"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


PIPELINE_STAGES_ORDER: List[str] = [
    "initializing",
    "analyzing_media",
    "retrieving_memory",
    "detecting_emotion",
    "selecting_media",
    "planning_story",
    "generating_motion",
    "generating_audio",
    "composing_video",
    "encoding_video",
    "uploading",
    "finalizing",
]

# Configurable raw weights (will be normalized automatically to exactly 100.0%)
CONFIGURED_STAGE_WEIGHTS: Dict[str, float] = {
    "initializing": 3.0,
    "analyzing_media": 12.0,
    "retrieving_memory": 8.0,
    "detecting_emotion": 10.0,
    "selecting_media": 7.0,
    "planning_story": 10.0,
    "generating_motion": 12.0,
    "generating_audio": 10.0,
    "composing_video": 15.0,
    "encoding_video": 8.0,
    "uploading": 5.0,
    "finalizing": 2.0,
}

# Default historical duration priors (seconds) for ETA cold-start estimation
DEFAULT_STAGE_DURATIONS: Dict[str, float] = {
    "initializing": 1.5,
    "analyzing_media": 4.5,
    "retrieving_memory": 2.0,
    "detecting_emotion": 3.5,
    "selecting_media": 1.5,
    "planning_story": 3.0,
    "generating_motion": 5.0,
    "generating_audio": 4.0,
    "composing_video": 12.0,
    "encoding_video": 6.0,
    "uploading": 4.0,
    "finalizing": 1.0,
}

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "queued": "Waiting for available background worker...",
    "initializing": "Initializing generation environment & assets...",
    "analyzing_media": "Analyzing image quality, EXIF metadata, and semantic tags...",
    "retrieving_memory": "Retrieving memory context and RAG representations...",
    "detecting_emotion": "Inferring multimodal emotional tone & generating timeline...",
    "selecting_media": "Selecting best story candidates with diversity ranking...",
    "planning_story": "Directing narrative arc and Ken Burns choreography...",
    "generating_motion": "Synthesizing Ken Burns camera pan and zoom vectors...",
    "generating_audio": "Synthesizing emotion-tuned 4-chord soundtrack & narration...",
    "composing_video": "Composing video clips, burning subtitles & ducking audio...",
    "encoding_video": "Encoding high-efficiency H.264 video reel...",
    "uploading": "Uploading generated memory video to secure storage...",
    "finalizing": "Finalizing memory video record & signed links...",
    "completed": "Memory video creation completed!",
    "failed": "Memory video creation paused.",
    "cancelled": "Memory video creation cancelled by user.",
    "retrying": "Recovering and resuming video generation...",
}


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    """Ensures stage weights always sum to exactly 100.0%."""
    total = sum(raw_weights.values())
    if total <= 0:
        count = len(raw_weights)
        return {k: 100.0 / count for k in raw_weights}
    return {k: (v / total) * 100.0 for k, v in raw_weights.items()}


NORMALIZED_STAGE_WEIGHTS = normalize_weights(CONFIGURED_STAGE_WEIGHTS)


class VideoJobState:
    """In-memory and persistent state container for a single video job."""

    def __init__(
        self,
        job_id: str,
        memory_id: str,
        user_id: str,
        dimension: str = "9:16",
        mood: str = "calm",
        selected_media_ids: Optional[List[str]] = None,
        specification: Optional[Dict[str, Any]] = None,
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.id: str = job_id
        self.memory_id: str = memory_id
        self.user_id: str = user_id
        self.dimension: str = dimension
        self.mood: str = mood
        self.selected_media_ids: List[str] = selected_media_ids or []
        self.specification: Optional[Dict[str, Any]] = specification

        self.status: str = JobStatus.QUEUED.value
        self.stage: str = "queued"
        self.stage_index: int = 0
        self.total_stages: int = len(PIPELINE_STAGES_ORDER)
        self.overall_progress: int = 0
        self.stage_progress: int = 0
        self.current_task: str = STAGE_DESCRIPTIONS["queued"]

        self.created_at: str = now_iso
        self.started_at: Optional[str] = None
        self.updated_at: str = now_iso
        self.completed_at: Optional[str] = None
        self.heartbeat_at: Optional[str] = None

        self.worker_id: Optional[str] = None
        self.attempt_number: int = 0
        self.retry_count: int = 0
        self.last_completed_stage: Optional[str] = None

        self.estimated_remaining_seconds: Optional[int] = None
        self.elapsed_seconds: int = 0
        self.upload_speed_mbps: Optional[float] = None
        self.upload_progress: Optional[int] = None

        self.result_media_id: Optional[str] = None
        self.result_url: Optional[str] = None
        self.staged_file_path: Optional[str] = None
        self.error_code: Optional[str] = None
        self.error_message: Optional[str] = None

        # Stage duration timestamps for real ETA and historical telemetry
        self.stage_start_times: Dict[str, float] = {}
        self.stage_durations: Dict[str, float] = {}

        # Telemetry metrics
        self.telemetry: Dict[str, Any] = {
            "queue_wait_ms": 0,
            "total_generation_ms": 0,
            "cache_hit": 0,
            "cache_miss": 0,
            "model_load_count": 0,
            "upload_retry_count": 0,
            "upload_throughput_mbps": 0.0,
            "failure_reason": None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete job state for API responses and persistence."""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "status": self.status,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "overall_progress": self.overall_progress,
            "stage_progress": self.stage_progress,
            "current_task": self.current_task,
            "started_at": self.started_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "heartbeat_at": self.heartbeat_at,
            "worker_id": self.worker_id,
            "attempt_number": self.attempt_number,
            "retry_count": self.retry_count,
            "last_completed_stage": self.last_completed_stage,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "upload_speed_mbps": self.upload_speed_mbps,
            "upload_progress": self.upload_progress,
            "result_media_id": self.result_media_id,
            "result_url": self.result_url,
            "staged_file_path": self.staged_file_path,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "dimension": self.dimension,
            "mood": self.mood,
            "specification": self.specification,
            "telemetry": self.telemetry,
        }


class JobStateManager:
    """
    Central, thread-safe manager for video job states.
    Maintains in-memory state, Redis synchronization, and Supabase database persistence.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, VideoJobState] = {}
        # Historical average stage durations collected across completed jobs
        self._historical_stage_durations: Dict[str, float] = copy.deepcopy(DEFAULT_STAGE_DURATIONS)

    def create_job(
        self,
        job_id: str,
        memory_id: str,
        user_id: str,
        dimension: str = "9:16",
        mood: str = "calm",
        selected_media_ids: Optional[List[str]] = None,
        specification: Optional[Dict[str, Any]] = None,
    ) -> VideoJobState:
        with self._lock:
            state = VideoJobState(
                job_id=job_id,
                memory_id=memory_id,
                user_id=user_id,
                dimension=dimension,
                mood=mood,
                selected_media_ids=selected_media_ids,
                specification=specification,
            )
            self._jobs[job_id] = state

        # Sync to Redis cache
        self._sync_to_cache(state)
        return state

    def get_job(self, job_id: str) -> Optional[VideoJobState]:
        with self._lock:
            state = self._jobs.get(job_id)
            if state:
                return state

        # Try recovering from Redis
        cached = cache_service.get(f"mv:job_state:{job_id}")
        if cached:
            state = self._deserialize_state(cached)
            with self._lock:
                self._jobs[job_id] = state
            return state

        # Try recovering from DB
        try:
            supabase = get_supabase_client()
            res = supabase.table("video_jobs").select("*").eq("id", job_id).execute()
            if res.data and isinstance(res.data, list) and len(res.data) > 0:
                row = cast(Dict[str, Any], res.data[0])
                state = VideoJobState(
                    job_id=str(row.get("id") or job_id),
                    memory_id=str(row.get("memory_id") or ""),
                    user_id=str(row.get("user_id") or ""),
                )
                state.status = str(row.get("status") or "queued")
                err_msg = row.get("error_message")
                state.error_message = str(err_msg) if err_msg is not None else None
                res_media_id = row.get("result_media_id")
                state.result_media_id = str(res_media_id) if res_media_id is not None else None
                with self._lock:
                    self._jobs[job_id] = state
                return state
        except Exception as e:
            logger.debug(f"Could not recover job {job_id} from DB: {e}")

        return None

    def transition_stage(
        self,
        job_id: str,
        stage: str,
        task_description: Optional[str] = None,
        custom_task_desc: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Transitions job to a new stage and computes updated weighted progress."""
        desc = task_description or custom_task_desc
        return self.start_stage(job_id=job_id, stage=stage, custom_task_desc=desc)

    def start_stage(
        self,
        job_id: str,
        stage: str,
        custom_task_desc: Optional[str] = None,
        task_description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Transitions job to a new stage and computes updated weighted progress."""
        desc = custom_task_desc or task_description
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            now = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()

            # Finalize previous stage duration if any
            if job.stage and job.stage in job.stage_start_times:
                prev_duration = now - job.stage_start_times[job.stage]
                job.stage_durations[job.stage] = prev_duration
                # Update telemetry for previous stage
                telemetry_key = f"{job.stage}_ms"
                job.telemetry[telemetry_key] = int(prev_duration * 1000)
                job.last_completed_stage = job.stage

            # Set new stage
            job.stage = stage
            job.status = "processing"
            job.updated_at = now_iso
            job.heartbeat_at = now_iso
            job.stage_start_times[stage] = now
            job.stage_progress = 0

            if stage in PIPELINE_STAGES_ORDER:
                job.stage_index = PIPELINE_STAGES_ORDER.index(stage) + 1
            else:
                job.stage_index = min(job.stage_index + 1, job.total_stages)

            job.current_task = desc or STAGE_DESCRIPTIONS.get(stage, f"Executing {stage}...")

            # Calculate overall progress
            self._update_progress_and_eta(job)

        self._persist_job(job)

    def update_stage_progress(
        self,
        job_id: str,
        stage_progress: int,
        custom_task_desc: Optional[str] = None,
        upload_speed_mbps: Optional[float] = None,
        stage: Optional[str] = None,
        task_description: Optional[str] = None,
        upload_progress: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Updates internal progress for the active stage (e.g. 3 of 10 items analyzed)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            if stage:
                job.stage = stage
            job.stage_progress = max(0, min(100, stage_progress))
            desc = custom_task_desc or task_description
            if desc:
                job.current_task = desc
            if upload_speed_mbps is not None:
                job.upload_speed_mbps = round(upload_speed_mbps, 2)
            if upload_progress is not None:
                job.upload_progress = max(0, min(100, upload_progress))
            elif job.stage == "uploading":
                job.upload_progress = job.stage_progress

            job.heartbeat_at = datetime.now(timezone.utc).isoformat()
            self._update_progress_and_eta(job)

        self._persist_job(job, db_sync=False)

    def record_upload_retry(self, job_id: str) -> None:
        """Increments upload retry counter in telemetry."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.telemetry["upload_retry_count"] = job.telemetry.get("upload_retry_count", 0) + 1

    def set_staged_file(self, job_id: str, file_path: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.staged_file_path = file_path
        if job:
            self._persist_job(job, db_sync=False)

    def update_heartbeat(self, job_id: str, worker_id: str) -> None:
        """Worker liveness pulse."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in ("completed", "failed", "cancelled"):
                return
            job.worker_id = worker_id
            job.heartbeat_at = datetime.now(timezone.utc).isoformat()
            if job.started_at:
                try:
                    st = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                    job.elapsed_seconds = int((datetime.now(timezone.utc) - st).total_seconds())
                except Exception:
                    pass

        self._sync_to_cache(job)

    def complete_job(
        self,
        job_id: str,
        result_media_id: str,
        result_url: str,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            now_iso = datetime.now(timezone.utc).isoformat()
            job.status = JobStatus.COMPLETED.value
            job.stage = "completed"
            job.stage_progress = 100
            job.overall_progress = 100
            job.estimated_remaining_seconds = 0
            job.completed_at = now_iso
            job.updated_at = now_iso
            job.result_media_id = result_media_id
            job.result_url = result_url
            job.current_task = STAGE_DESCRIPTIONS["completed"]

            if job.started_at:
                try:
                    st = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                    dur_ms = int((datetime.now(timezone.utc) - st).total_seconds() * 1000)
                    job.telemetry["total_generation_ms"] = dur_ms
                except Exception:
                    pass

            # Update historical stage durations for future ETA learning
            for s, d in job.stage_durations.items():
                if s in self._historical_stage_durations:
                    # Exponential moving average (alpha = 0.3)
                    self._historical_stage_durations[s] = (
                        0.7 * self._historical_stage_durations[s] + 0.3 * d
                    )

        self._persist_job(job, db_sync=True)
        logger.info(f"Job {job_id} marked COMPLETED. Total time: {job.elapsed_seconds}s")

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = JobStatus.FAILED.value
            job.stage = "failed"
            job.error_code = error_code
            job.error_message = error_message
            job.updated_at = datetime.now(timezone.utc).isoformat()
            job.current_task = f"Failed: {error_message}"
            job.telemetry["failure_reason"] = f"{error_code}: {error_message}"

        self._persist_job(job, db_sync=True)
        logger.error(f"Job {job_id} marked FAILED ({error_code}): {error_message}")

    def cancel_job(self, job_id: str, user_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.user_id != user_id:
                return False
            if job.status in ("completed", "failed", "cancelled"):
                return False

            job.status = JobStatus.CANCELLED.value
            job.stage = "cancelled"
            job.updated_at = datetime.now(timezone.utc).isoformat()
            job.current_task = "Cancelled by user."

        self._persist_job(job, db_sync=True)
        logger.info(f"Job {job_id} CANCELLED by user {user_id}")
        return True

    def record_cache_hit(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.telemetry["cache_hit"] += 1

    def record_cache_miss(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.telemetry["cache_miss"] += 1

    # ── Internal Progress & ETA Computation ───────────────────────────────────

    def _update_progress_and_eta(self, job: VideoJobState) -> None:
        """Calculates normalized overall progress (0-100%) and stage-aware ETA."""
        # 1. Overall Progress
        completed_weight = 0.0
        current_weight = 0.0

        if job.stage in PIPELINE_STAGES_ORDER:
            curr_idx = PIPELINE_STAGES_ORDER.index(job.stage)
            for i, st in enumerate(PIPELINE_STAGES_ORDER):
                w = NORMALIZED_STAGE_WEIGHTS.get(st, 0.0)
                if i < curr_idx:
                    completed_weight += w
                elif i == curr_idx:
                    current_weight = w
                    break

        weighted_progress = completed_weight + (current_weight * (job.stage_progress / 100.0))
        # Ensure progress is non-decreasing and capped at 99% until completed
        new_progress = int(weighted_progress)
        job.overall_progress = min(99, max(job.overall_progress, new_progress))

        # 2. Elapsed Seconds
        now = datetime.now(timezone.utc)
        if job.started_at:
            try:
                st = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                job.elapsed_seconds = max(0, int((now - st).total_seconds()))
            except Exception:
                pass

        # 3. Real ETA Estimator
        if job.stage in PIPELINE_STAGES_ORDER:
            curr_idx = PIPELINE_STAGES_ORDER.index(job.stage)
            remaining_seconds = 0.0

            # Remaining work in current stage
            curr_stage_name = job.stage
            expected_curr_duration = self._historical_stage_durations.get(
                curr_stage_name, DEFAULT_STAGE_DURATIONS.get(curr_stage_name, 5.0)
            )
            stage_fraction_left = max(0.0, 1.0 - (job.stage_progress / 100.0))
            remaining_seconds += expected_curr_duration * stage_fraction_left

            # Remaining upcoming stages
            for upcoming in PIPELINE_STAGES_ORDER[curr_idx + 1:]:
                remaining_seconds += self._historical_stage_durations.get(
                    upcoming, DEFAULT_STAGE_DURATIONS.get(upcoming, 5.0)
                )

            # Low confidence guard: If in first stage and elapsed < 2s, show "Estimating..."
            if curr_idx == 0 and job.elapsed_seconds < 2:
                job.estimated_remaining_seconds = None
            else:
                job.estimated_remaining_seconds = max(2, int(remaining_seconds))
        else:
            job.estimated_remaining_seconds = None

    # ── Persistence & Synchronization ─────────────────────────────────────────

    def _sync_to_cache(self, job: VideoJobState) -> None:
        """Saves current state snapshot to Redis with 24-hour TTL."""
        try:
            cache_service.set(f"mv:job_state:{job.id}", job.to_dict(), ttl=86400)
        except Exception as e:
            logger.debug(f"Failed to sync job {job.id} to cache: {e}")

    def _persist_job(self, job: VideoJobState, db_sync: bool = False) -> None:
        """Persists state to Redis and optionally updates Supabase video_jobs table."""
        self._sync_to_cache(job)

        if db_sync or job.status in ("completed", "failed", "cancelled", "processing"):
            try:
                supabase = get_supabase_client()
                update_payload: dict[str, Any] = {
                    "status": job.status,
                    "updated_at": job.updated_at,
                }
                if job.error_message:
                    update_payload["error_message"] = f"[{job.error_code or 'ERR'}] {job.error_message}"
                if job.result_media_id:
                    update_payload["result_media_id"] = job.result_media_id

                update_payload["script_metadata"] = {
                    "stage": job.stage,
                    "current_task": job.current_task,
                    "overall_progress": job.overall_progress,
                    "stage_progress": job.stage_progress,
                    "dimension": job.dimension,
                    "mood": job.mood,
                    "specification": job.specification,
                }

                supabase.table("video_jobs").update(update_payload).eq("id", job.id).execute()
            except Exception as db_err:
                # Resilient fallback: Database connection drop does NOT crash worker!
                logger.warning(
                    f"Supabase DB update failed for job {job.id} (network/DNS drop). "
                    f"State safely retained in local memory & cache fallback: {db_err}"
                )

    def _deserialize_state(self, d: Dict[str, Any]) -> VideoJobState:
        state = VideoJobState(
            job_id=d["id"],
            memory_id=d["memory_id"],
            user_id=d["user_id"],
            dimension=d.get("dimension", "9:16"),
            mood=d.get("mood", "calm"),
            selected_media_ids=d.get("selected_media_ids"),
            specification=d.get("specification"),
        )
        for k, v in d.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# Global singleton instance
job_state_manager = JobStateManager()
