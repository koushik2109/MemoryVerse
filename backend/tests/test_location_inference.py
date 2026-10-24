from unittest.mock import patch

from app.services.location_inference import (
    EXIF_CONFIRMED,
    NO_VLM_KEY,
    VLM_CANDIDATE,
    LocationInferenceService,
)
from app.services.ai_service import AIService


def test_exif_coordinates_are_authoritative_and_skip_vlm():
    latitude = 37.421999912345
    longitude = -122.084057512345

    with patch("app.services.ai_service.call_structured_vlm") as mock_vlm:
        result = LocationInferenceService.infer(
            b"image bytes",
            "image/jpeg",
            exif_metadata={"latitude": latitude, "longitude": longitude},
        )

    assert result["decision_status"] == EXIF_CONFIRMED
    assert result["source"] == "exif"
    assert result["latitude"] == latitude
    assert result["longitude"] == longitude
    mock_vlm.assert_not_called()


def test_no_vlm_key_returns_safe_structured_fallback():
    with patch.object(LocationInferenceService, "_metadata_from_exif", return_value={}), \
         patch("app.services.location_inference.settings.LLM_API_KEY", None), \
         patch("app.services.location_inference.settings.LLM_PROVIDER", "openai"), \
         patch("app.services.ai_service.call_structured_vlm") as mock_vlm:
        result = LocationInferenceService.infer(b"image bytes", "image/jpeg")

    assert result["decision_status"] == NO_VLM_KEY
    assert result["candidate_location"] is None
    assert result["confidence"] == 0.0
    assert result["evidence"]
    mock_vlm.assert_not_called()


def test_structured_vlm_candidate_is_normalized_with_decision():
    vlm_result = {
        "candidate_location": "Yosemite National Park",
        "confidence": 0.87,
        "evidence": "The Half Dome landmark is clearly visible.",
        "decision_status": "candidate",
    }

    with patch.object(LocationInferenceService, "_metadata_from_exif", return_value={}), \
         patch("app.services.location_inference.settings.LLM_API_KEY", "test-key"), \
         patch("app.services.location_inference.settings.LLM_PROVIDER", "openrouter"), \
         patch("app.services.ai_service.call_structured_vlm", return_value=vlm_result) as mock_vlm:
        result = LocationInferenceService.infer(b"image bytes", "image/jpeg")

    assert result["candidate_location"] == "Yosemite National Park"
    assert result["confidence"] == 0.87
    assert result["evidence"] == "The Half Dome landmark is clearly visible."
    assert result["decision_status"] == VLM_CANDIDATE
    assert result["source"] == "vlm"
    mock_vlm.assert_called_once()


def test_exact_exif_coordinates_can_be_decided_against_requested_place():
    with patch.object(
        LocationInferenceService,
        "_reverse_geocode",
        return_value={
            "display_name": "Shimla, Himachal Pradesh, India",
            "address": {
                "city": "Shimla",
                "state": "Himachal Pradesh",
                "country": "India",
            },
        },
    ):
        result = LocationInferenceService.infer(
            b"image bytes",
            "image/jpeg",
            exif_metadata={"latitude": 31.1048, "longitude": 77.1734},
            location_hint="Create journal of Shimla",
        )

    assert result["decision_status"] == EXIF_CONFIRMED
    assert result["source"] == "exif+reverse_geocoder"
    assert result["candidate_location"] == "Shimla"
    assert result["latitude"] == 31.1048
    assert result["longitude"] == 77.1734


def test_local_curation_promotes_grounded_location_candidate():
    with patch.object(
        LocationInferenceService,
        "infer",
        return_value={
            "candidate_location": "Shimla",
            "confidence": 0.92,
            "decision_status": EXIF_CONFIRMED,
            "source": "exif",
        },
    ), patch(
        "app.services.ai_service.call_structured_vlm",
        return_value={
            "visual_description": "A mountain road and pine-covered slopes.",
            "is_subject_visible": False,
            "relevance": "uncertain",
            "relevance_score": 0.35,
            "confidence": 0.8,
            "evidence": "Landscape is consistent with a mountain trip.",
        },
    ):
        result = AIService._curate_single_item_vlm(
            {
                "id": "shimla_1",
                "filename": "IMG_0001.jpg",
                "bytes": b"image bytes",
                "mime_type": "image/jpeg",
                "timestamp": "2025-01-01T00:00:00Z",
            },
            "Create journal of Shimla",
        )

    assert result["recommended"] is True
    assert result["location_candidate"] == "Shimla"
    assert result["location_decision_status"] == EXIF_CONFIRMED
    assert result["relevance_score"] >= 0.90


def test_local_curation_accepts_platform_timestamp_strings():
    with patch.object(
        LocationInferenceService,
        "infer",
        return_value={"candidate_location": None, "confidence": 0.0, "decision_status": "no_candidate"},
    ), patch(
        "app.services.ai_service.call_structured_vlm",
        return_value={
            "visual_description": "A landscape",
            "is_subject_visible": False,
            "relevance": "uncertain",
            "relevance_score": 0.5,
            "evidence": "No specific landmark.",
        },
    ):
        result = AIService.curate_local_media(
            prompt="Create journal of Shimla",
            files_data=[
                {
                    "id": "asset-1",
                    "filename": "IMG_0001.jpg",
                    "bytes": b"image bytes",
                    "mime_type": "image/jpeg",
                    "timestamp": "2025-01-01T00:00:00Z",
                }
            ],
        )

    assert result["suggested_date"] == "2025-01-01T00:00:00+00:00"


def test_metadata_only_candidates_use_exact_coordinates_for_place_retrieval():
    with patch.object(
        LocationInferenceService,
        "_reverse_geocode",
        return_value={
            "display_name": "Shimla, Himachal Pradesh, India",
            "address": {"city": "Shimla", "state": "Himachal Pradesh", "country": "India"},
        },
    ), patch("app.services.ai_extractor.get_clip_model", return_value=None):
        result = AIService.filter_media_rag(
            prompt="Create journal of Shimla",
            candidates=[
                {
                    "id": "asset-1",
                    "filename": "asset-1.jpg",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "latitude": 31.1048,
                    "longitude": 77.1734,
                }
            ],
            top_k=1,
        )

    match = result["matched_media"][0]
    assert match["location_candidate"] == "Shimla"
    assert match["location_decision_status"] == EXIF_CONFIRMED
    assert match["recommended"] is True
    assert match["relevance_score"] >= 0.95
