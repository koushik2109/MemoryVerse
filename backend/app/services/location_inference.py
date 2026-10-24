"""EXIF-first location inference for uploaded media.

The upload pipeline already extracts EXIF metadata locally and the AI service
already exposes a structured VLM hook.  This module keeps the location decision
between those two existing pieces deliberately small:

* exact EXIF GPS coordinates always win and never trigger a VLM request;
* images without GPS can receive a visual location *candidate* from the VLM;
* missing credentials, unsupported media, malformed VLM output, and provider
  failures produce an explicit non-fatal decision instead of raising.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


EXIF_CONFIRMED = "exif_confirmed"
VLM_CANDIDATE = "vlm_candidate"
VLM_UNCERTAIN = "vlm_uncertain"
NO_CANDIDATE = "no_candidate"
NO_VLM_KEY = "no_vlm_key"
UNSUPPORTED_MEDIA = "unsupported_media"
VLM_ERROR = "vlm_error"


@dataclass(frozen=True)
class LocationInferenceResult:
    """Stable, serializable output of a location inference decision."""

    candidate_location: Optional[str] = None
    confidence: float = 0.0
    evidence: str = ""
    decision_status: str = NO_CANDIDATE
    source: str = "none"
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-safe representation for JSON persistence."""
        return asdict(self)


def _is_finite_coordinate(value: Any) -> bool:
    """Return whether a coordinate is a real finite number."""
    try:
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalise_confidence(value: Any, default: float = 0.0) -> float:
    """Clamp model confidence to the public 0..1 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    if not math.isfinite(confidence):
        confidence = default
    return round(max(0.0, min(1.0, confidence)), 4)


def _normalise_evidence(value: Any) -> str:
    """Keep evidence readable when a model returns a list or another scalar."""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _normalise_candidate(value: Any) -> Optional[str]:
    """Extract a human-readable location name from common VLM JSON shapes."""
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("location") or value.get("label")
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


class LocationInferenceService:
    """Infer a media location without allowing visual guesses to replace GPS."""

    @staticmethod
    def _metadata_from_exif(file_bytes: bytes, mime_type: str) -> dict[str, Any]:
        """Load existing local EXIF extraction for standalone callers.

        The upload path passes already-extracted metadata, so this lazy import
        avoids a module cycle while keeping this component useful in isolation.
        """
        try:
            from app.services.ai_extractor import AIExtractor

            return AIExtractor.extract_metadata(file_bytes, mime_type)
        except Exception as exc:
            logger.debug("EXIF metadata lookup skipped during location inference: %s", exc)
            return {}

    @staticmethod
    def _result_from_exif(metadata: Mapping[str, Any]) -> Optional[LocationInferenceResult]:
        """Build the authoritative result when both EXIF GPS values exist."""
        latitude = metadata.get("latitude")
        longitude = metadata.get("longitude")
        if not (_is_finite_coordinate(latitude) and _is_finite_coordinate(longitude)):
            return None

        # Do not round, geocode, or otherwise transform these values.  The
        # database update receives the same values that EXIF extraction found.
        return LocationInferenceResult(
            candidate_location=_normalise_candidate(metadata.get("location_name")),
            confidence=1.0,
            evidence="Exact latitude and longitude extracted from EXIF GPS metadata.",
            decision_status=EXIF_CONFIRMED,
            source="exif",
            latitude=latitude,
            longitude=longitude,
        )

    @staticmethod
    @lru_cache(maxsize=512)
    def _reverse_geocode(latitude: float, longitude: float) -> Optional[dict[str, Any]]:
        """Resolve exact coordinates to a place label through the configured tool.

        This is deliberately cached and only called when a caller supplied a
        human location hint.  Normal background ingestion therefore never
        sends coordinates to a third party merely to populate a label.
        """
        try:
            import httpx

            url = str(settings.LOCATION_GEOCODER_URL or "").strip()
            if not url:
                return None
            with httpx.Client(timeout=4.0) as client:
                response = client.get(
                    url,
                    params={
                        "lat": f"{latitude:.7f}",
                        "lon": f"{longitude:.7f}",
                        "format": "jsonv2",
                        "zoom": "10",
                    },
                    headers={"User-Agent": settings.LOCATION_GEOCODER_USER_AGENT},
                )
                response.raise_for_status()
                payload = response.json()
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.debug("Reverse geocoding skipped safely: %s", exc)
            return None

    @staticmethod
    def _location_hint_matches(address: Mapping[str, Any], location_hint: str) -> Optional[str]:
        """Return the matching address component, if the hint is grounded."""
        ignored = {
            "create", "make", "new", "a", "an", "the", "of", "for", "my",
            "and", "with", "journal", "memory", "memories", "photos", "photo",
            "images", "image", "pictures", "picture", "video", "videos", "trip",
            "travel", "journey", "landscape", "landscapes", "scenic", "beach",
            "mountain", "mountains", "nature", "city", "cityscape", "sunset",
            "sunrise", "snow", "water", "forest", "road", "from", "in", "at",
        }
        hint_terms = {
            token.lower()
            for token in re.findall(r"[a-z0-9]+", location_hint.lower())
            if len(token) > 2 and token.lower() not in ignored
        }
        if not hint_terms:
            return None

        display_name = str(address.get("display_name") or "")
        address_text = " ".join(str(value) for value in (address.get("address") or {}).values())
        searchable = f"{display_name} {address_text}".lower()
        if not all(term in searchable for term in hint_terms):
            return None

        components = address.get("address")
        if isinstance(components, Mapping):
            for key in ("city", "town", "village", "municipality", "county", "state", "country"):
                value = components.get(key)
                if value and any(term in str(value).lower() for term in hint_terms):
                    return str(value)
        return display_name or None

    @classmethod
    def _label_exact_coordinates(
        cls,
        result: LocationInferenceResult,
        location_hint: Optional[str],
    ) -> LocationInferenceResult:
        """Add a grounded place label without changing EXIF coordinates."""
        if result is None or result.candidate_location or not location_hint:
            return result
        if result.latitude is None or result.longitude is None:
            return result

        # City-level decisions do not need sub-meter geocoder precision. The
        # lookup is quantized to improve cache reuse across burst photos while
        # the original EXIF coordinates remain untouched in the result.
        payload = cls._reverse_geocode(
            round(float(result.latitude), 3),
            round(float(result.longitude), 3),
        )
        if not payload:
            return result
        label = cls._location_hint_matches(payload, location_hint)
        if not label:
            return result
        return LocationInferenceResult(
            candidate_location=label,
            confidence=1.0,
            evidence="Exact EXIF GPS coordinates reverse-geocoded to a place matching the requested location.",
            decision_status=EXIF_CONFIRMED,
            source="exif+reverse_geocoder",
            latitude=result.latitude,
            longitude=result.longitude,
        )

    @staticmethod
    def _no_key_result() -> LocationInferenceResult:
        return LocationInferenceResult(
            evidence="No LLM/VLM API key is configured; visual location inference was skipped.",
            decision_status=NO_VLM_KEY,
            source="none",
        )

    @staticmethod
    def _normalise_vlm_result(raw_result: Any) -> LocationInferenceResult:
        """Validate and normalize the structured response from the existing VLM hook."""
        if not isinstance(raw_result, Mapping):
            return LocationInferenceResult(
                evidence="The visual model did not return a structured location candidate.",
                decision_status=NO_CANDIDATE,
                source="vlm",
            )

        candidate = _normalise_candidate(
            raw_result.get("candidate_location")
            or raw_result.get("location")
            or raw_result.get("candidate")
        )
        confidence = _normalise_confidence(raw_result.get("confidence"), default=0.0)
        evidence = _normalise_evidence(raw_result.get("evidence"))

        # The model may return a decision label, but the service still derives
        # the safe status from the presence and strength of its candidate.
        requested_status = str(raw_result.get("decision_status") or "").strip().lower()
        if not candidate:
            status = NO_CANDIDATE
        elif requested_status in {VLM_CANDIDATE, VLM_UNCERTAIN}:
            status = requested_status
        elif confidence >= 0.65:
            status = VLM_CANDIDATE
        else:
            status = VLM_UNCERTAIN

        if not evidence:
            evidence = "Visual evidence was returned without a detailed explanation."

        return LocationInferenceResult(
            candidate_location=candidate,
            confidence=confidence,
            evidence=evidence,
            decision_status=status,
            source="vlm",
        )

    @staticmethod
    def infer(
        file_bytes: bytes,
        mime_type: str,
        exif_metadata: Optional[Mapping[str, Any]] = None,
        location_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return an EXIF-first structured location decision.

        ``exif_metadata`` should be the output of ``AIExtractor.extract_metadata``
        when available.  If omitted, it is loaded lazily for callers that use
        this component directly.  Only image bytes are sent to the VLM because
        the existing VLM hook is an image endpoint, not a video decoder.
        """
        metadata = dict(exif_metadata) if exif_metadata is not None else LocationInferenceService._metadata_from_exif(file_bytes, mime_type)

        exif_result = LocationInferenceService._result_from_exif(metadata)
        if exif_result is not None:
            return LocationInferenceService._label_exact_coordinates(
                exif_result,
                location_hint,
            ).to_dict()

        if not mime_type.lower().startswith("image/"):
            return LocationInferenceResult(
                evidence="Location inference requires an image preview; no image was available for visual inspection.",
                decision_status=UNSUPPORTED_MEDIA,
                source="none",
            ).to_dict()

        if not file_bytes:
            return LocationInferenceResult(
                evidence="The uploaded image was empty; visual location inference was skipped.",
                decision_status=UNSUPPORTED_MEDIA,
                source="none",
            ).to_dict()

        provider = str(settings.LLM_PROVIDER or "").strip().lower()
        api_key = str(settings.LLM_API_KEY or "").strip()
        if not api_key or provider == "none":
            return LocationInferenceService._no_key_result().to_dict()

        prompt = (
            "Inspect this image for visual evidence of its geographic location. "
            "Identify a recognizable landmark, named place, city, region, or country only when visible evidence supports it. "
            "Generic scenery is not enough for a specific place; use null when the location cannot be grounded. "
            "Do not infer from filenames or hidden metadata, and do not invent GPS coordinates. "
            "Return ONLY a JSON object with this schema: "
            '{"candidate_location": "<place name or null>", '
            '"confidence": <number from 0.0 to 1.0>, '
            '"evidence": "<concise visible evidence>", '
            '"decision_status": "candidate" | "uncertain" | "no_candidate"}'
        )
        default_fallback = {
            "candidate_location": None,
            "confidence": 0.0,
            "evidence": "The visual model did not provide a grounded location candidate.",
            "decision_status": "no_candidate",
        }

        try:
            # Import lazily so no VLM implementation or optional provider is
            # initialized when EXIF or the no-key path already resolved it.
            from app.services.ai_service import call_structured_vlm

            raw_result = call_structured_vlm(
                file_bytes,
                prompt,
                default_fallback=default_fallback,
            )
            return LocationInferenceService._normalise_vlm_result(raw_result).to_dict()
        except Exception as exc:
            logger.warning("Location VLM inference failed safely: %s", exc)
            return LocationInferenceResult(
                evidence="Visual location inference failed; the media was retained without a location decision.",
                decision_status=VLM_ERROR,
                source="vlm",
            ).to_dict()


def infer_location(
    file_bytes: bytes,
    mime_type: str,
    exif_metadata: Optional[Mapping[str, Any]] = None,
    location_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Functional facade for callers that do not need the service class."""
    return LocationInferenceService.infer(
        file_bytes,
        mime_type,
        exif_metadata,
        location_hint=location_hint,
    )


location_inference_service = LocationInferenceService()
