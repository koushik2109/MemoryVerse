"""
MemoryVerse - Durable Video Job Queue & Background Worker
Provides persistent queueing (Redis List + disk fallback), worker concurrency bounds,
heartbeat monitoring, stale job recovery, and idempotent stage resumption.
"""
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.core.cache_service import cache_service
from app.services.job_state_manager import job_state_manager, JobStatus

logger = logging.getLogger(__name__)

QUEUE_KEY = "mv:video_job_queue"
MAX_CONCURRENT_JOBS = 2
MAX_JOB_ATTEMPTS = 3
HEARTBEAT_INTERVAL_SEC = 5.0
STALE_THRESHOLD_SEC = 45.0
LOCAL_QUEUE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "storage", "queue_journal.json"
)


class DurableVideoQueue:
    """
    Durable queue supporting Upstash Redis with local disk journal fallback.
    Guarantees job persistence across server restarts and process lifecycles.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._local_queue: list[str] = []
        self._init_local_journal()

    def _init_local_journal(self):
        os.makedirs(os.path.dirname(os.path.abspath(LOCAL_QUEUE_FILE)), exist_ok=True)
        if os.path.exists(LOCAL_QUEUE_FILE):
            try:
                with open(LOCAL_QUEUE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._local_queue = data
            except Exception as e:
                logger.warning(f"Could not load local queue journal: {e}")
                self._local_queue = []

    def _save_local_journal(self):
        try:
            with open(LOCAL_QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._local_queue, f)
        except Exception as e:
            logger.warning(f"Could not save local queue journal: {e}")

    def enqueue(self, job_id: str) -> None:
        """Pushes job_id to the durable queue."""
        # 1. Redis Queue if available
        if cache_service.redis:
            try:
                cache_service.redis.rpush(QUEUE_KEY, job_id)
                logger.info(f"Job {job_id} enqueued to Upstash Redis.")
                return
            except Exception as e:
                logger.warning(f"Failed to enqueue to Redis, falling back to local journal: {e}")

        # 2. Local Disk Journal Fallback
        with self._lock:
            if job_id not in self._local_queue:
                self._local_queue.append(job_id)
                self._save_local_journal()
        logger.info(f"Job {job_id} enqueued to local disk journal.")

    def dequeue(self) -> Optional[str]:
        """Pulls next job_id from queue."""
        # 1. Redis Queue
        if cache_service.redis:
            try:
                job_id = cache_service.redis.lpop(QUEUE_KEY)
                if job_id:
                    return str(job_id)
            except Exception as e:
                logger.debug(f"Redis dequeue failed: {e}")

        # 2. Local Journal
        with self._lock:
            if self._local_queue:
                job_id = self._local_queue.pop(0)
                self._save_local_journal()
                return job_id

        return None

    def length(self) -> int:
        if cache_service.redis:
            try:
                return int(cache_service.redis.llen(QUEUE_KEY) or 0)
            except Exception:
                pass
        with self._lock:
            return len(self._local_queue)


class VideoWorkerPool:
    """
    Background worker pool managing durable job processing,
    heartbeats, bounded concurrency, and crash recovery.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_JOBS):
        self.queue = DurableVideoQueue()
        self.max_concurrent = max_concurrent
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        self._running = False
        self._active_job_ids: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._pipeline_runner: Optional[Callable[..., None]] = None

    def register_pipeline_runner(self, runner_func: Callable[..., None]) -> None:
        """Registers the main video processing function (VideoService.execute_job)."""
        self._pipeline_runner = runner_func

    def start(self) -> None:
        """Starts background worker threads and watchdog."""
        if self._running:
            return
        self._running = True
        logger.info(f"Starting VideoWorkerPool [{self.worker_id}] with {self.max_concurrent} workers...")

        for i in range(self.max_concurrent):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"video_worker_{i+1}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # Watchdog thread for stale jobs and recovery
        watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="video_worker_watchdog",
            daemon=True,
        )
        watchdog.start()
        self._threads.append(watchdog)

    def stop(self) -> None:
        self._running = False
        logger.info(f"VideoWorkerPool [{self.worker_id}] shutting down.")

    def submit_job(self, job_id: str) -> None:
        self.queue.enqueue(job_id)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                job_id = self.queue.dequeue()
                if not job_id:
                    time.sleep(1.0)
                    continue

                job = job_state_manager.get_job(job_id)
                if not job or job.status in ("completed", "cancelled"):
                    continue

                self._process_single_job(job_id)

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(1.0)

    def _process_single_job(self, job_id: str) -> None:
        job = job_state_manager.get_job(job_id)
        if not job:
            return

        with threading.Lock():
            self._active_job_ids.add(job_id)

        now_iso = datetime.now(timezone.utc).isoformat()
        job.worker_id = self.worker_id
        job.attempt_number += 1
        job.started_at = job.started_at or now_iso
        job.updated_at = now_iso
        job.heartbeat_at = now_iso
        job.status = JobStatus.INITIALIZING.value

        # Calculate queue wait time
        try:
            ct = datetime.fromisoformat(job.created_at.replace("Z", "+00:00"))
            wait_ms = int((datetime.now(timezone.utc) - ct).total_seconds() * 1000)
            job.telemetry["queue_wait_ms"] = max(0, wait_ms)
        except Exception:
            pass

        job_state_manager._persist_job(job, db_sync=True)

        # Start dedicated heartbeat thread for this active job
        stop_heartbeat = threading.Event()
        hb_thread = threading.Thread(
            target=self._job_heartbeat_loop,
            args=(job_id, stop_heartbeat),
            daemon=True,
        )
        hb_thread.start()

        try:
            if self._pipeline_runner:
                self._pipeline_runner(job_id)
            else:
                logger.error(f"No pipeline runner registered to execute job {job_id}")
                job_state_manager.fail_job(job_id, "NO_PIPELINE_RUNNER", "Pipeline runner not configured")
        except Exception as e:
            logger.exception(f"Unhandled error in video pipeline for job {job_id}: {e}")
            job_state_manager.fail_job(job_id, "UNHANDLED_EXCEPTION", str(e))
        finally:
            stop_heartbeat.set()
            with threading.Lock():
                self._active_job_ids.discard(job_id)

    def _job_heartbeat_loop(self, job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            job_state_manager.update_heartbeat(job_id, self.worker_id)
            stop_event.wait(timeout=HEARTBEAT_INTERVAL_SEC)

    def _watchdog_loop(self) -> None:
        """
        Periodically detects stale jobs whose heartbeat expired (> 45s),
        recovering or requeueing them up to MAX_JOB_ATTEMPTS.
        """
        while self._running:
            try:
                time.sleep(30.0)
                now = datetime.now(timezone.utc)

                with job_state_manager._lock:
                    all_jobs = list(job_state_manager._jobs.values())

                for job in all_jobs:
                    if job.status not in ("processing", "initializing"):
                        continue
                    if job.id in self._active_job_ids:
                        continue  # Currently active in this process

                    # Check heartbeat age
                    if job.heartbeat_at:
                        try:
                            hb_time = datetime.fromisoformat(job.heartbeat_at.replace("Z", "+00:00"))
                            delta_sec = (now - hb_time).total_seconds()
                            if delta_sec > STALE_THRESHOLD_SEC:
                                logger.warning(
                                    f"Watchdog detected STALE job {job.id} (last heartbeat {delta_sec:.1f}s ago). "
                                    f"Attempt {job.attempt_number}/{MAX_JOB_ATTEMPTS}."
                                )
                                if job.attempt_number < MAX_JOB_ATTEMPTS:
                                    logger.info(f"Re-queueing stale job {job.id} for recovery...")
                                    job.status = JobStatus.RETRYING.value
                                    job.retry_count += 1
                                    job_state_manager._persist_job(job, db_sync=True)
                                    self.queue.enqueue(job.id)
                                else:
                                    logger.error(f"Job {job.id} exceeded max attempts. Marking FAILED.")
                                    job_state_manager.fail_job(
                                        job.id,
                                        "WORKER_CRASH_LIMIT_EXCEEDED",
                                        f"Worker heartbeat died {delta_sec:.0f}s ago and exceeded retry limits.",
                                    )
                        except Exception as hb_err:
                            logger.debug(f"Error parsing heartbeat for job {job.id}: {hb_err}")

            except Exception as w_err:
                logger.debug(f"Watchdog loop iteration error: {w_err}")


# Global worker pool instance
video_worker_pool = VideoWorkerPool(max_concurrent=MAX_CONCURRENT_JOBS)
