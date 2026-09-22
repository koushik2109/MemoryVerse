"""
MemoryVerse - Resilient Staging & Storage Service
Handles local disk staging, streaming multipart uploads to Supabase Storage,
bandwidth tracking, error classification, exponential backoff retries with jitter,
and disk lifecycle management for generated videos.
"""
import io
import logging
import os
import random
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

import httpx

from app.core.db import get_supabase_client
from app.services.job_state_manager import job_state_manager

logger = logging.getLogger(__name__)

# Base storage staging and ratio-partitioned videos root
STAGING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "storage", "staging", "reels")
)
VIDEOS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "storage", "videos")
)

# Retention policies
SUCCESSFUL_CLEANUP_IMMEDIATE = False  # Retain offline local copy in ratio folder
FAILED_JOB_RETENTION_HOURS = 24
MAX_UPLOAD_ATTEMPTS = 5
BASE_RETRY_DELAY_SEC = 1.0


class ProgressFileReader(io.BufferedReader):
    """
    BufferedReader wrapper that tracks bytes read and invokes progress callback
    with elapsed time, upload speed (MB/s), and progress percentage (0-100%).
    Inherits from io.BufferedReader so Supabase storage3 and httpx recognize it as a valid byte stream.
    """

    def __init__(
        self,
        filepath: str,
        total_size: int,
        progress_callback: Optional[Callable[[int, float, int], None]] = None,
    ):
        raw = io.FileIO(filepath, "rb")
        super().__init__(raw)
        self.total_size = total_size
        self.progress_callback = progress_callback
        self.bytes_read = 0
        self.start_time = time.time()
        self._last_callback_time = 0.0

    def read(self, size: Optional[int] = -1) -> bytes:
        chunk = super().read(size if size is not None else -1)
        if chunk:
            self.bytes_read += len(chunk)
            now = time.time()
            # Throttle callback to at most once every 100ms
            if self.progress_callback and (now - self._last_callback_time >= 0.1 or self.bytes_read >= self.total_size):
                self._last_callback_time = now
                elapsed = max(0.001, now - self.start_time)
                speed_mbps = (self.bytes_read / (1024 * 1024)) / elapsed
                percent = int(min(100, (self.bytes_read / max(1, self.total_size)) * 100))
                try:
                    self.progress_callback(percent, round(speed_mbps, 2), self.bytes_read)
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")
        return chunk


class StorageService:
    """
    Manages local staging of generated reels, streaming upload to Supabase,
    error classification, and automated retry recovery.
    """

    def __init__(self):
        os.makedirs(STAGING_DIR, exist_ok=True)

    @staticmethod
    def get_staging_path(job_id: str, filename: str) -> str:
        """Returns the deterministic local staged filepath for a video job."""
        os.makedirs(STAGING_DIR, exist_ok=True)
        safe_filename = os.path.basename(filename)
        return os.path.join(STAGING_DIR, f"{job_id}_{safe_filename}")

    @staticmethod
    def get_ratio_video_path(ratio: str, filename: str) -> str:
        """
        Returns the persistent local filepath partitioned by aspect ratio.
        e.g., storage/videos/16_9/filename.mp4, storage/videos/9_16/filename.mp4
        """
        safe_ratio = ratio.replace(":", "_").replace("/", "_").strip() or "16_9"
        ratio_dir = os.path.join(VIDEOS_DIR, safe_ratio)
        os.makedirs(ratio_dir, exist_ok=True)
        safe_filename = os.path.basename(filename)
        return os.path.join(ratio_dir, safe_filename)

    @staticmethod
    def save_to_ratio_storage(src_path: str, ratio: str, filename: str) -> str:
        """
        Copies/saves the rendered video file into the ratio-partitioned persistent videos folder.
        """
        dst_path = StorageService.get_ratio_video_path(ratio, filename)
        try:
            shutil.copy2(src_path, dst_path)
            logger.info(f"Persisted video to ratio storage: {dst_path}")
            return dst_path
        except Exception as e:
            logger.warning(f"Could not copy to ratio storage {dst_path}: {e}")
            return src_path

    @staticmethod
    def classify_error(exc: Exception) -> Tuple[str, str]:
        """
        Classifies network and storage exceptions into structured error codes.
        Returns: (error_code, human_friendly_message)
        """
        err_str = str(exc).lower()

        if isinstance(exc, httpx.ConnectError) or "no address associated with hostname" in err_str or "errno -5" in err_str:
            return (
                "NETWORK_DNS_FAILURE",
                "DNS lookup failed when connecting to cloud storage. Your generated video is safely saved locally."
            )
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timeout" in err_str or "timed out" in err_str:
            return (
                "NETWORK_TIMEOUT",
                "Connection timed out during cloud upload. The video artifact is preserved."
            )
        if "401" in err_str or "unauthorized" in err_str or "invalid jwt" in err_str:
            return (
                "STORAGE_AUTH_FAILURE",
                "Authentication failed with storage service. Please re-authenticate."
            )
        if "403" in err_str or "permission denied" in err_str or "access denied" in err_str:
            return (
                "STORAGE_FORBIDDEN",
                "Insufficient permissions to upload to target storage bucket."
            )
        if any(code in err_str for code in ["500", "502", "503", "504"]):
            return (
                "STORAGE_SERVER_ERROR",
                "Cloud storage service encountered a server error. Automatic retry available."
            )

        return ("STORAGE_UPLOAD_ERROR", f"Storage error: {str(exc)[:200]}")

    def upload_video_with_retry(
        self,
        job_id: str,
        user_id: str,
        staged_filepath: str,
        output_filename: str,
        max_attempts: int = MAX_UPLOAD_ATTEMPTS,
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Streams staged video from local disk to Supabase Storage with exponential backoff & jitter.
        Returns: (success, signed_url, storage_path, error_code)
        """
        if not os.path.exists(staged_filepath):
            logger.error(f"Staged video not found: {staged_filepath}")
            return False, None, None, "STAGED_FILE_NOT_FOUND"

        file_size = os.path.getsize(staged_filepath)
        reel_storage_path = f"{user_id}/reels/{output_filename}"
        supabase = get_supabase_client()

        def on_progress(percent: int, speed_mbps: float, bytes_sent: int):
            job_state_manager.update_stage_progress(
                job_id=job_id,
                stage_progress=percent,
                custom_task_desc=f"Uploading video reel ({speed_mbps:.1f} MB/s)...",
                upload_speed_mbps=speed_mbps,
                stage="uploading",
                upload_progress=percent,
            )

        attempt = 1
        last_error_code = "UNKNOWN_ERROR"
        last_error_msg = ""

        while attempt <= max_attempts:
            try:
                logger.info(
                    f"Job {job_id}: Upload attempt {attempt}/{max_attempts} for {output_filename} "
                    f"({file_size // 1024} KB) -> {reel_storage_path}"
                )

                # Use ProgressFileReader to stream without loading 100MB into memory
                try:
                    with ProgressFileReader(staged_filepath, file_size, on_progress) as reader:
                        supabase.storage.from_("memories").upload(
                            file=cast(Any, reader),
                            path=reel_storage_path,
                            file_options={"content-type": "video/mp4", "upsert": "true"},
                        )
                except Exception as stream_err:
                    logger.warning(
                        f"Job {job_id}: Progress reader stream upload failed ({stream_err}), falling back to direct file upload..."
                    )
                    supabase.storage.from_("memories").upload(
                        file=staged_filepath,
                        path=reel_storage_path,
                        file_options={"content-type": "video/mp4", "upsert": "true"},
                    )

                # Generate long-lived signed URL
                signed_res = supabase.storage.from_("memories").create_signed_url(reel_storage_path, 31536000)
                raw_url = signed_res.get("signedURL") or signed_res.get("signed_url") or ""
                public_url: str = str(raw_url)

                logger.info(f"Job {job_id}: Successfully uploaded reel to Supabase Storage.")
                return True, public_url, reel_storage_path, None

            except Exception as e:
                error_code, friendly_msg = self.classify_error(e)
                last_error_code = error_code
                last_error_msg = friendly_msg
                logger.warning(
                    f"Job {job_id}: Upload attempt {attempt}/{max_attempts} failed with "
                    f"[{error_code}]: {e}"
                )

                # Record upload retry count in telemetry
                job_state_manager.record_upload_retry(job_id)

                if attempt < max_attempts:
                    # Exponential backoff with random jitter: 2^(attempt-1) * base + jitter
                    jitter = random.uniform(0.1, 0.9)
                    delay = (BASE_RETRY_DELAY_SEC * (2 ** (attempt - 1))) + jitter
                    logger.info(f"Job {job_id}: Sleeping {delay:.2f}s before next upload attempt...")
                    time.sleep(delay)

                attempt += 1

        # All attempts failed: Preserve the artifact on disk!
        logger.error(
            f"Job {job_id}: All {max_attempts} upload attempts failed. "
            f"Artifact PRESERVED at {staged_filepath}."
        )
        return False, None, reel_storage_path, last_error_code

    def cleanup_staged_file(self, staged_filepath: str) -> None:
        """Safely removes staged file after verified database insertion."""
        try:
            if os.path.exists(staged_filepath):
                os.remove(staged_filepath)
                logger.info(f"Cleaned up staged artifact: {staged_filepath}")
        except Exception as e:
            logger.warning(f"Could not remove staged file {staged_filepath}: {e}")

    def cleanup_expired_staging(self, retention_hours: int = FAILED_JOB_RETENTION_HOURS) -> int:
        """
        Deletes orphaned staging files older than retention_hours.
        Protects recently failed jobs so users can still retry them.
        """
        if not os.path.exists(STAGING_DIR):
            return 0

        now = time.time()
        max_age_sec = retention_hours * 3600
        removed_count = 0

        try:
            for fname in os.listdir(STAGING_DIR):
                fpath = os.path.join(STAGING_DIR, fname)
                if os.path.isfile(fpath):
                    mtime = os.path.getmtime(fpath)
                    if (now - mtime) > max_age_sec:
                        try:
                            os.remove(fpath)
                            removed_count += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Error during staging cleanup: {e}")

        return removed_count


storage_service = StorageService()
