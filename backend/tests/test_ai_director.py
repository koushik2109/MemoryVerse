"""
Unit and Integration Tests for AI Memory Director, AspectRatioComposer, QualityValidator, and RevisionEngine
"""
import pytest
from PIL import Image
from app.schemas.ai_session import VideoSpecification, MemoryContext
from app.services.revision_engine import RevisionEngine, revision_engine
from app.services.ai_director_service import AIDirectorService, ai_director_service
from ai_engine.video_generation.aspect_ratio_composer import (
    AspectRatioComposer,
    BackgroundStrategy,
    CompositionFitMode,
    TargetAspectRatio,
)
from ai_engine.video_generation.quality_validator import QualityValidator


def test_aspect_ratio_composer_layout():
    """Verify portrait image in landscape canvas is NEVER stretched and calculates proper layout."""
    src_w, src_h = 1080, 1920  # Portrait 9:16
    target_w, target_h = 1280, 720  # Landscape 16:9

    layout = AspectRatioComposer.calculate_layout(src_w, src_h, target_w, target_h, margin_pct=0.07)
    assert layout["is_aspect_mismatch"] is True
    assert layout["fit_mode"] == CompositionFitMode.ADAPTIVE_BACKGROUND.value
    assert layout["fg_h"] <= target_h
    assert layout["fg_w"] < target_w

    # Verify aspect ratio of scaled foreground is preserved within 0.1%
    original_aspect = src_w / src_h
    scaled_aspect = layout["fg_w"] / layout["fg_h"]
    assert abs(original_aspect - scaled_aspect) < 0.01


def test_aspect_ratio_composer_compose_image():
    """Verify image composition generates expected target dimensions without distortion."""
    src_img = Image.new("RGB", (600, 1000), color=(200, 80, 100))
    target_size = (1280, 720)

    composed = AspectRatioComposer.compose_image(
        src_img,
        target_size=target_size,
        strategy=BackgroundStrategy.ADAPTIVE_CINEMATIC.value,
    )
    assert composed.size == target_size
    assert composed.mode == "RGB"


def test_revision_engine_specification_diff():
    """Verify specification diffs only invalidate affected pipeline stages."""
    spec_old = VideoSpecification(
        duration_target_seconds=30,
        aspect_ratio="16:9",
    )
    spec_old.music.style = "cinematic_warm"

    # 1. Music-only change
    spec_music_changed = spec_old.model_copy(deep=True)
    spec_music_changed.music.style = "energetic"

    changed_fields, invalidated = revision_engine.diff_specifications(spec_old, spec_music_changed)
    assert "music" in changed_fields
    assert "planning_story" not in invalidated
    assert "generating_audio" in invalidated
    assert "encoding_video" in invalidated

    # 2. Duration change
    spec_dur_changed = spec_old.model_copy(deep=True)
    spec_dur_changed.duration_target_seconds = 45

    changed_fields, invalidated = revision_engine.diff_specifications(spec_old, spec_dur_changed)
    assert "duration_target_seconds" in changed_fields
    assert "planning_story" in invalidated
    assert "composing_video" in invalidated


def test_ai_director_specification_mutation():
    """Test deterministic natural language mutation of VideoSpecification."""
    context = MemoryContext(
        memory_id="mem_test",
        user_id="user_test",
        title="Trip to Paris",
        media_count=12,
        photo_count=10,
        video_count=2,
        media_ids=[f"m_{i}" for i in range(12)],
        dominant_emotion="joy",
    )
    spec = ai_director_service._build_default_specification(context)

    # Test "make it 20 seconds"
    mutated_spec, reply, suggestions = ai_director_service._interpret_and_mutate_spec(
        user_text="Make it 20 seconds and vertical",
        spec=spec,
        context=context,
    )
    assert mutated_spec.duration_target_seconds == 20
    assert mutated_spec.aspect_ratio == "9:16"
    assert mutated_spec.orientation == "portrait"
    assert "20s" in reply
    assert "9:16" in reply

    # Test "change music to acoustic and add narration"
    mutated_spec2, reply2, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="Change music to acoustic and add voiceover narration",
        spec=mutated_spec,
        context=context,
    )
    assert mutated_spec2.music.style == "acoustic"
    assert mutated_spec2.narration.enabled is True


def test_target_aspect_ratio_dimensions():
    """Verify TargetAspectRatio returns correct dimensions for 16:9, 9:16, 1:1, and 4:3."""
    # Standard resolution
    assert TargetAspectRatio.get_dimensions("16:9", high_res=False) == (1280, 720)
    assert TargetAspectRatio.get_dimensions("9:16", high_res=False) == (720, 1280)
    assert TargetAspectRatio.get_dimensions("1:1", high_res=False) == (720, 720)
    assert TargetAspectRatio.get_dimensions("4:3", high_res=False) == (960, 720)

    # High resolution
    assert TargetAspectRatio.get_dimensions("16:9", high_res=True) == (1920, 1080)
    assert TargetAspectRatio.get_dimensions("9:16", high_res=True) == (1080, 1920)
    assert TargetAspectRatio.get_dimensions("1:1", high_res=True) == (1080, 1080)
    assert TargetAspectRatio.get_dimensions("4:3", high_res=True) == (1440, 1080)


def test_ai_director_extended_mutations():
    """Test 4:3, Instagram reel, mute music, quality profile, and conversational question interpretations."""
    context = MemoryContext(
        memory_id="mem_test",
        user_id="user_test",
        title="Birthday Celebration",
        media_count=8,
        photo_count=6,
        video_count=2,
        media_ids=[f"m_{i}" for i in range(8)],
        dominant_emotion="joy",
    )
    spec = ai_director_service._build_default_specification(context)

    # 1. Test 4:3 aspect ratio
    s1, r1, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="Change format to 4:3 classic",
        spec=spec,
        context=context,
    )
    assert s1.aspect_ratio == "4:3"
    assert "4:3" in r1

    # 2. Test Instagram Reel -> 9:16
    s2, r2, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="Optimize this for an Instagram Reel with high quality",
        spec=s1,
        context=context,
    )
    assert s2.aspect_ratio == "9:16"
    assert s2.quality_profile == "high_quality"

    # 3. Test mute music / no music
    s3, r3, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="Remove music and keep it silent or narration only",
        spec=s2,
        context=context,
    )
    assert s3.music.style == "none"

    # 4. Test question handling ("why did you choose this media?")
    _, r4, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="Why did you select these photos?",
        spec=s3,
        context=context,
    )
    assert "aesthetic score" in r4.lower() or "emotional" in r4.lower() or "quality" in r4.lower()


def test_approve_and_generate_job_schema():
    """Verify approve_and_generate strictly adheres to Supabase video_jobs columns."""
    from unittest.mock import MagicMock, patch

    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "job_12345"}])

    context = MemoryContext(
        memory_id="mem_123",
        user_id="user_123",
        title="Family Vacation",
        media_count=5,
        photo_count=5,
        video_count=0,
        media_ids=["m1", "m2", "m3", "m4", "m5"],
        dominant_emotion="joy",
    )
    spec = ai_director_service._build_default_specification(context)
    spec.aspect_ratio = "4:3"
    spec.quality_profile = "high_quality"
    session_id = "sess_test_schema_validation"
    ai_director_service._sessions[session_id] = {
        "session_id": session_id,
        "user_id": "user_123",
        "memory_id": "mem_123",
        "title": "Movie: Family Vacation",
        "context": context,
        "specification": spec,
        "messages": [],
        "suggestions": [],
        "created_at": "2026-09-22T00:00:00Z",
        "updated_at": "2026-09-22T00:00:00Z",
        "status": "active",
        "active_job_id": None,
    }

    with patch("app.services.ai_director_service.get_supabase_client", return_value=mock_supabase), \
         patch("app.core.cache_service.cache_service.get_active_job_by_fingerprint", return_value=None), \
         patch("app.services.video_queue.video_worker_pool.submit_job") as mock_submit:

        resp, job_id = ai_director_service.approve_and_generate(
            session_id=session_id,
            user_id="user_123",
            custom_spec=spec,
        )

        assert job_id == "job_12345"
        mock_submit.assert_called_once_with("job_12345")

        # Verify insert payload strictly adheres to valid columns
        mock_table.insert.assert_called_once()
        inserted_payload = mock_table.insert.call_args[0][0]

        valid_columns = {"id", "memory_id", "user_id", "status", "result_media_id", "error_message", "script_metadata"}
        invalid_keys = set(inserted_payload.keys()) - valid_columns
        assert not invalid_keys, f"Found unmapped top-level columns in video_jobs insert: {invalid_keys}"

        # Verify script_metadata contains specification and settings
        meta = inserted_payload["script_metadata"]
        assert meta["dimension"] == "4:3"
        assert meta["specification"]["quality_profile"] == "high_quality"


def test_ai_director_conversational_qa_and_storage():
    """Verify conversational Q&A for media counts, dates, and emotions plus ratio storage."""
    context = MemoryContext(
        memory_id="mem_water",
        user_id="user_koushik",
        title="Water Reflections",
        media_count=4,
        photo_count=3,
        video_count=1,
        event_date="2026-09-15",
        media_ids=["p1", "p2", "p3", "v1"],
        dominant_emotion="serene",
        emotion_confidence=0.88,
        emotion_distribution={"calm": 0.5, "serene": 0.5},
    )
    spec = ai_director_service._build_default_specification(context)

    # 1. Test media count question (exact user prompt from screenshot 4)
    _, reply_count, suggestions_count = ai_director_service._interpret_and_mutate_spec(
        user_text="give the number of videos and photos in the memory",
        spec=spec,
        context=context,
    )
    assert "3 photos and 1 video" in reply_count
    assert "4 moments" in reply_count
    assert "Generate Video" in suggestions_count

    # 2. Test date/timeline question
    _, reply_date, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="when was this memory captured?",
        spec=spec,
        context=context,
    )
    assert "2026-09-15" in reply_date

    # 3. Test emotion question
    _, reply_mood, _ = ai_director_service._interpret_and_mutate_spec(
        user_text="what emotion and mood does this memory have?",
        spec=spec,
        context=context,
    )
    assert "Serene" in reply_mood

    # 4. Test ratio video storage path
    from app.services.storage_service import storage_service
    p_16_9 = storage_service.get_ratio_video_path("16:9", "test_reel.mp4")
    assert "videos/16_9/test_reel.mp4" in p_16_9 or "videos/16_9\\test_reel.mp4" in p_16_9
    p_9_16 = storage_service.get_ratio_video_path("9:16", "test_reel.mp4")
    assert "videos/9_16/test_reel.mp4" in p_9_16 or "videos/9_16\\test_reel.mp4" in p_9_16


def test_cinematic_title_synthesis_and_intro_card():
    """Verify title/tagline synthesis and intro title card generation."""
    from app.services.video_service import _synthesize_cinematic_title_and_tagline, _create_ai_title_card_clip

    # 1. Verify semantic transformation removes prompt phrases and yields poetic titles
    title, tagline = _synthesize_cinematic_title_and_tagline(
        "Create A Memory Filtering The Images Where There Is Water In It",
        emotion="serene",
    )
    assert title == "Water Reflections"
    assert "serene waters" in tagline or "Echoes" in tagline

    # 2. Verify beach topic mapping
    title_b, tagline_b = _synthesize_cinematic_title_and_tagline(
        "filter the images where there is beach and sun",
        emotion="joyful",
    )
    assert title_b == "Golden Shores"
    assert "Sunlight" in tagline_b

    # 3. Verify title card clip creation
    clip = _create_ai_title_card_clip(
        title="Water Reflections",
        subtitle="September 2026",
        duration=2.5,
        target_size=(640, 360),
        emotion="serene",
    )
    assert clip.duration == 2.5
    assert clip.audio is not None
    # Sample a frame
    frame = clip.get_frame(1.0)
    assert frame.shape == (360, 640, 3)




