"""
Storyteller Agent
Constructs a narrative arc (Hook, Build-up, Climax, Resolution) grounded in
an Evidence Manifest built from retrieved multi-modal memories.
"""
import logging
from typing import Any, Dict, List, Optional
from ai_engine.agents.storyteller.narrative_arc import segment_assets_into_phases, NARRATIVE_PHASES, PHASE_DURATION_RATIOS

logger = logging.getLogger(__name__)


class StorytellerAgent:
    """
    Transforms curated media assets and audio cues into a structured, episodic story.
    """

    def build_evidence_manifest(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Creates an explicit factual grounding manifest tracking timestamps,
        captions, entities, and ambient tags for each media asset.
        """
        manifest_entries = []
        for item in items:
            manifest_entries.append({
                "id": item.get("id", "unknown"),
                "file_path": item.get("file_path", ""),
                "timestamp": item.get("timestamp", ""),
                "caption": item.get("auto_caption", ""),
                "modality": item.get("source_modality", "image"),
                "ambient": item.get("ambient_category", ""),
                "quality_score": item.get("quality_score", 0.8),
            })

        return {
            "total_assets": len(manifest_entries),
            "assets": manifest_entries,
            "temporal_span": {
                "start": manifest_entries[0]["timestamp"] if manifest_entries else None,
                "end": manifest_entries[-1]["timestamp"] if manifest_entries else None,
            },
        }

    def generate_narrative_arc(
        self,
        query: str,
        manifest: Dict[str, Any],
        target_duration: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Generates 4 narrative phases mapped to specific evidence IDs.
        """
        assets = manifest.get("assets", [])
        n_assets = len(assets)

        # Segment assets across the 4 phases
        hook_asset = assets[0] if n_assets > 0 else {}
        buildup_asset = assets[1] if n_assets > 1 else hook_asset
        climax_asset = assets[2] if n_assets > 2 else (assets[-1] if n_assets > 0 else {})
        res_asset = assets[3] if n_assets > 3 else (assets[-1] if n_assets > 0 else {})

        # Script phases with ground-truth linkages
        phases = [
            {
                "phase": "hook",
                "target_seconds": round(target_duration * 0.20, 1),
                "narration": f"Remember when we set out for {query}? Every moment began with quiet anticipation.",
                "referenced_media_ids": [hook_asset.get("id", "")] if hook_asset else [],
                "visual_focus": hook_asset.get("caption", "Setting the scene"),
                "tone": "inviting",
            },
            {
                "phase": "build_up",
                "target_seconds": round(target_duration * 0.30, 1),
                "narration": f"As the day unfolded, things came alive. {buildup_asset.get('caption', 'The journey continued with laughter and discovery.')}",
                "referenced_media_ids": [buildup_asset.get("id", "")] if buildup_asset else [],
                "visual_focus": buildup_asset.get("caption", "Rising excitement"),
                "tone": "dynamic",
            },
            {
                "phase": "climax",
                "target_seconds": round(target_duration * 0.30, 1),
                "narration": f"And then came the highlight: {climax_asset.get('caption', 'an unforgettable snapshot etched in time.')}",
                "referenced_media_ids": [climax_asset.get("id", "")] if climax_asset else [],
                "visual_focus": climax_asset.get("caption", "The peak highlight"),
                "tone": "exuberant",
            },
            {
                "phase": "resolution",
                "target_seconds": round(target_duration * 0.20, 1),
                "narration": f"Even now, looking back, that warmth remains untouched. A memory worth holding on to.",
                "referenced_media_ids": [res_asset.get("id", "")] if res_asset else [],
                "visual_focus": res_asset.get("caption", "Warm lingering reflection"),
                "tone": "reflective",
            },
        ]

        full_story_text = " ".join(p["narration"] for p in phases)

        return {
            "title": f"Story of {query.title()}",
            "target_duration": target_duration,
            "phases": phases,
            "full_narration": full_story_text,
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node execution."""
        scored_items = state.get("scored_items", [])
        query = state.get("query", "My Memory")
        target_duration = float(state.get("target_duration", 30.0))
        if target_duration <= 0:
            target_duration = 30.0

        # Build ground-truth evidence manifest
        evidence_manifest = self.build_evidence_manifest(scored_items)

        # Generate narrative arc grounded in evidence
        script = self.generate_narrative_arc(
            query=query,
            manifest=evidence_manifest,
            target_duration=target_duration,
        )

        return {
            "evidence_manifest": evidence_manifest,
            "script": script,
        }
