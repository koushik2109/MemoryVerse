"""
MemoryVerse - Revision Engine & Selective Pipeline Invalidation
Computes structured diffs between VideoSpecifications and determines minimal invalidated pipeline stages.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
from app.schemas.ai_session import VideoSpecification, VideoRevisionResponse


# Stage dependency mapping: which pipeline stages need to rerun when a property changes
PROPERTY_STAGE_INVALIDATION: Dict[str, Set[str]] = {
    "music": {"generating_audio", "composing_video", "encoding_video", "uploading", "finalizing"},
    "narration": {"generating_audio", "composing_video", "encoding_video", "uploading", "finalizing"},
    "duration_target_seconds": {
        "planning_story",
        "generating_motion",
        "generating_audio",
        "composing_video",
        "encoding_video",
        "uploading",
        "finalizing",
    },
    "media_selection": {
        "selecting_media",
        "planning_story",
        "generating_motion",
        "generating_audio",
        "composing_video",
        "encoding_video",
        "uploading",
        "finalizing",
    },
    "emotion": {
        "detecting_emotion",
        "planning_story",
        "generating_motion",
        "generating_audio",
        "composing_video",
        "encoding_video",
        "uploading",
        "finalizing",
    },
    "motion": {"generating_motion", "composing_video", "encoding_video", "uploading", "finalizing"},
    "composition": {"composing_video", "encoding_video", "uploading", "finalizing"},
    "aspect_ratio": {"generating_motion", "composing_video", "encoding_video", "uploading", "finalizing"},
}


class RevisionEngine:
    """
    Computes specification diffs and invalidates only affected pipeline stages.
    """

    @staticmethod
    def compute_specification_hash(spec: VideoSpecification) -> str:
        """Deterministic SHA-256 fingerprint of normalized specification."""
        data = spec.model_dump()
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def diff_specifications(
        cls,
        old_spec: VideoSpecification,
        new_spec: VideoSpecification,
    ) -> Tuple[List[str], List[str]]:
        """
        Compares old and new specifications.
        Returns:
            changed_fields: list of top-level field names that differ.
            invalidated_stages: list of pipeline stage names that must re-execute.
        """
        old_dict = old_spec.model_dump()
        new_dict = new_spec.model_dump()

        changed_fields: List[str] = []
        invalidated_stages_set: Set[str] = set()

        for key, new_val in new_dict.items():
            if key == "version":
                continue
            old_val = old_dict.get(key)
            if old_val != new_val:
                changed_fields.append(key)
                impacted = PROPERTY_STAGE_INVALIDATION.get(key, {"planning_story", "composing_video", "encoding_video"})
                invalidated_stages_set.update(impacted)

        # Preserve canonical pipeline ordering
        stage_order = [
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
        ordered_invalidated = [s for s in stage_order if s in invalidated_stages_set]

        return changed_fields, ordered_invalidated

    @classmethod
    def create_revision(
        cls,
        parent_revision_id: str | None,
        job_id: str,
        old_spec: VideoSpecification,
        new_spec: VideoSpecification,
    ) -> VideoRevisionResponse:
        """
        Creates a new VideoRevision record tracking changed fields and invalidated stages.
        """
        changed_fields, invalidated_stages = cls.diff_specifications(old_spec, new_spec)
        spec_hash = cls.compute_specification_hash(new_spec)
        rev_id = f"rev_{uuid.uuid4().hex[:12]}"

        return VideoRevisionResponse(
            revision_id=rev_id,
            parent_revision_id=parent_revision_id,
            job_id=job_id,
            specification_hash=spec_hash,
            changed_fields=changed_fields,
            invalidated_stages=invalidated_stages,
            status="queued",
            created_at=datetime.now(timezone.utc),
        )


revision_engine = RevisionEngine()
