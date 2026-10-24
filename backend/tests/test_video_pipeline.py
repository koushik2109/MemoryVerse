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
import asyncio
import pytest
from app.core.cache_service import CacheService
from app.services.job_state_manager import (
    JobStateManager,
    JobStatus,
    NORMALIZED_STAGE_WEIGHTS,
    PIPELINE_STAGES_ORDER,
)
import io
from types import SimpleNamespace

from app.services.storage_service import StorageService, ProgressFileReader
from app.services.video_queue import DurableVideoQueue
from app.services.video_service import _filter_media_to_selected_ids
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


def test_image_quality_exposes_noise_signal():
    """Quality scoring must distinguish sharpness from high-frequency noise."""
    import numpy as np
    from PIL import Image
    from app.services.media_intelligence import score_image_quality

    with tempfile.TemporaryDirectory() as tmp_dir:
        rng = np.random.default_rng(7)
        clean = np.full((240, 240), 128, dtype=np.uint8)
        noisy = np.clip(clean.astype(np.int16) + rng.normal(0, 45, clean.shape), 0, 255).astype(np.uint8)
        clean_path = os.path.join(tmp_dir, "clean.jpg")
        noisy_path = os.path.join(tmp_dir, "noisy.jpg")
        Image.fromarray(clean).save(clean_path)
        Image.fromarray(noisy).save(noisy_path)

        clean_score = score_image_quality(clean_path)
        noisy_score = score_image_quality(noisy_path)

    assert "noise_score" in clean_score
    assert "noise_score" in noisy_score
    assert noisy_score["noise_score"] < clean_score["noise_score"]


def test_selected_media_ids_survive_job_state_serialization(monkeypatch):
    """Curated IDs must survive cache/worker state round-trips."""
    import app.services.job_state_manager as job_state_module

    persisted_updates = []

    class FakeQuery:
        def update(self, payload):
            persisted_updates.append(payload)
            return self

        def eq(self, *_args):
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class FakeSupabase:
        def table(self, _name):
            return FakeQuery()

    monkeypatch.setattr(
        job_state_module,
        "get_supabase_client",
        lambda: FakeSupabase(),
    )

    manager = JobStateManager()
    job = manager.create_job(
        job_id="test_curated_ids",
        memory_id="memory_1",
        user_id="user_1",
        selected_media_ids=["media_a", "media_b"],
    )

    snapshot = job.to_dict()
    restored = manager._deserialize_state(snapshot)
    manager.start_stage("test_curated_ids", "initializing")

    assert snapshot["selected_media_ids"] == ["media_a", "media_b"]
    assert restored.selected_media_ids == ["media_a", "media_b"]
    assert persisted_updates[-1]["script_metadata"]["selected_media_ids"] == [
        "media_a",
        "media_b",
    ]


def test_selected_media_filter_does_not_fall_back_to_unselected_items():
    """A stale curated ID must not cause the worker to render all memory media."""
    media = [
        {"id": "selected", "filename": "selected.jpg"},
        {"id": "unselected", "filename": "unselected.jpg"},
    ]

    assert _filter_media_to_selected_ids(media, ["selected"]) == [media[0]]
    assert _filter_media_to_selected_ids(media, ["missing"]) == []


def test_generate_video_persists_selected_media_ids(monkeypatch):
    """The API must put curated IDs in the durable video_jobs metadata."""
    from app.api import media as media_api
    from app.core.security import CurrentUser

    inserted_payloads = []
    created_jobs = []

    class FakeQuery:
        def __init__(self, operation):
            self.operation = operation

        def update(self, payload):
            return self

        def insert(self, payload):
            inserted_payloads.append(payload)
            return self

        def eq(self, *_args):
            return self

        def in_(self, *_args):
            return self

        def execute(self):
            if self.operation == "insert":
                return SimpleNamespace(data=[{"id": "job_curated"}])
            return SimpleNamespace(data=[])

    class FakeSupabase:
        def table(self, name):
            return FakeQuery("insert" if name == "video_jobs" else "update")

    monkeypatch.setattr(media_api, "get_supabase_client", lambda: FakeSupabase())
    monkeypatch.setattr(
        media_api.cache_service,
        "compute_job_fingerprint",
        lambda **kwargs: "fingerprint",
    )
    monkeypatch.setattr(
        media_api.cache_service,
        "get_active_job_by_fingerprint",
        lambda _fingerprint: None,
    )
    monkeypatch.setattr(
        media_api.cache_service,
        "set_active_job_by_fingerprint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        media_api.job_state_manager,
        "create_job",
        lambda **kwargs: created_jobs.append(kwargs),
    )
    monkeypatch.setattr(media_api.video_worker_pool, "submit_job", lambda _job_id: None)

    response = asyncio.run(
        media_api.generate_video(
            memory_id="memory_curated",
            media_ids=["media_a", "", "media_b"],
            current_user=CurrentUser(id="user_1", email="user@example.com"),
        )
    )

    assert response["job_id"] == "job_curated"
    assert inserted_payloads[0]["script_metadata"]["selected_media_ids"] == [
        "media_a",
        "media_b",
    ]
    assert created_jobs[0]["selected_media_ids"] == ["media_a", "media_b"]
