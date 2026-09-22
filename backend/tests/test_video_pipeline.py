"""
Unit and Integration Tests for MemoryVerse Advanced Video Pipeline
Tests:
  - Multi-level Cache Service & Job Fingerprinting
  - JobStateManager 12-stage normalization & ETA computation
  - Durable Video Queue & concurrency bounds
  - Storage Service staging & error categorization
  - Emotion Analyzer 5 ablation modes & timeline fusion
"""
import os
import tempfile
import pytest
from app.core.cache_service import CacheService
from app.services.job_state_manager import (
    JobStateManager,
    JobStatus,
    NORMALIZED_STAGE_WEIGHTS,
    PIPELINE_STAGES_ORDER,
)
import io
from app.services.storage_service import StorageService, ProgressFileReader
from app.services.video_queue import DurableVideoQueue
from ai_engine.emotion.emotion_analyzer import EmotionAnalyzer, EmotionAblationMode


def test_stage_weights_sum_to_100():
    """Verify that stage weights always sum to exactly 100.0%."""
    total = sum(NORMALIZED_STAGE_WEIGHTS.values())
    assert abs(total - 100.0) < 1e-4, f"Stage weights sum to {total}, expected 100.0"


def test_job_state_manager_lifecycle():
    """Test job state transitions, progress updates, and completion."""
    manager = JobStateManager()
    job = manager.create_job(
        job_id="test_job_1",
        memory_id="mem_1",
        user_id="user_1",
        dimension="9:16",
        mood="calm",
    )
    assert job.status == JobStatus.QUEUED.value
    assert job.overall_progress == 0
    assert job.stage_index == 0
    assert job.total_stages == 12

    # Transition to analyzing_media
    manager.start_stage("test_job_1", "analyzing_media")
    job = manager.get_job("test_job_1")
    assert job is not None
    assert job.status == "processing"
    assert job.stage == "analyzing_media"
    assert job.stage_index == 2
    assert job.overall_progress > 0

    # Sub-progress update
    manager.update_stage_progress("test_job_1", 50, upload_speed_mbps=12.5)
    job = manager.get_job("test_job_1")
    assert job is not None
    assert job.stage_progress == 50
    assert job.upload_speed_mbps == 12.5

    # Complete job
    manager.complete_job("test_job_1", result_media_id="media_123", result_url="https://cdn.example.com/reel.mp4")
    job = manager.get_job("test_job_1")
    assert job is not None
    assert job.status == JobStatus.COMPLETED.value
    assert job.overall_progress == 100
    assert job.result_url == "https://cdn.example.com/reel.mp4"


def test_job_state_manager_cancellation():
    """Test job cancellation with tenant isolation."""
    manager = JobStateManager()
    manager.create_job(
        job_id="test_job_cancel",
        memory_id="mem_1",
        user_id="user_owner",
    )

    # Unauthorized cancel should fail
    assert manager.cancel_job("test_job_cancel", "wrong_user") is False

    # Authorized cancel should succeed
    assert manager.cancel_job("test_job_cancel", "user_owner") is True
    job = manager.get_job("test_job_cancel")
    assert job is not None
    assert job.status == JobStatus.CANCELLED.value


def test_cache_service_job_fingerprinting():
    """Verify deterministic deduplication fingerprinting."""
    cache = CacheService()
    fp1 = cache.compute_job_fingerprint("u1", "m1", ["p1", "p2"], "calm", "9:16")
    fp2 = cache.compute_job_fingerprint("u1", "m1", ["p2", "p1"], "calm", "9:16") # order shouldn't matter
    fp3 = cache.compute_job_fingerprint("u2", "m1", ["p1", "p2"], "calm", "9:16") # different user

    assert fp1 == fp2
    assert fp1 != fp3


def test_storage_service_error_categorization():
    """Test error categorization into actionable network and auth codes."""
    storage = StorageService()

    dns_err = Exception("[Errno -5] No address associated with hostname")
    timeout_err = Exception("HTTPSConnectionPool(host='...', port=443): Read timed out.")
    auth_err = Exception("401 Unauthorized: Invalid JWT")

    assert storage.classify_error(dns_err)[0] == "NETWORK_DNS_FAILURE"
    assert storage.classify_error(timeout_err)[0] == "NETWORK_TIMEOUT"
    assert storage.classify_error(auth_err)[0] == "STORAGE_AUTH_FAILURE"


def test_emotion_analyzer_ablation_modes():
    """Verify all 5 ablation modes function correctly and produce valid distributions."""
    analyzer = EmotionAnalyzer()

    mock_media = [
        {
            "id": "item_1",
            "url": "https://example.com/photo.jpg",
            "media_type": "image",
            "metadata": {
                "aesthetic_score": 0.85,
                "color_palette": ["#FFB74D", "#FFE082"],
                "brightness": 0.7,
                "contrast": 0.6,
            },
        },
        {
            "id": "item_2",
            "url": "https://example.com/clip.mp4",
            "media_type": "video",
            "metadata": {
                "aesthetic_score": 0.9,
                "color_palette": ["#4FC3F7"],
                "brightness": 0.6,
            },
        },
    ]

    modes = [
        EmotionAblationMode.A_NONE,
        EmotionAblationMode.B_VISUAL_ONLY,
        EmotionAblationMode.C_VISUAL_AUDIO,
        EmotionAblationMode.D_VISUAL_AUDIO_TEXT,
        EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
    ]

    for mode in modes:
        timeline = analyzer.build_emotion_timeline(
            media_items=mock_media,
            memory_title="Grand Canyon Sunset Trip",
            ablation_mode=mode,
        )
        assert timeline is not None
        assert timeline.overall_mood in (
            "calm", "nostalgia", "nostalgic", "energetic", "excitement",
            "joy", "joyful", "sadness", "love"
        )
        assert len(timeline.keypoints) == len(mock_media)


def test_durable_queue_operations():
    """Test durable queue enqueue and dequeue with memory journal."""
    queue = DurableVideoQueue()
    test_id = "test_queue_item_xyz"

    queue.enqueue(test_id)
    assert queue.length() > 0

    dequeued = queue.dequeue()
    assert dequeued is not None


def test_progress_file_reader_compatibility():
    """Verify ProgressFileReader inherits from io.BufferedReader and reports progress correctly."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        payload = b"MemoryVerse Video Content 12345" * 100
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        progress_calls = []

        def on_progress(pct: int, speed: float, bytes_read: int):
            progress_calls.append((pct, speed, bytes_read))

        total_size = len(payload)
        with ProgressFileReader(tmp_path, total_size, on_progress) as reader:
            assert isinstance(reader, io.BufferedReader), "ProgressFileReader must inherit from io.BufferedReader"
            data = reader.read()
            assert data == payload

        assert len(progress_calls) > 0
        assert progress_calls[-1][2] == total_size
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
