"""
Video Director Agent
Compiles production-ready shot-lists, Ken Burns camera keyframes,
and audio-visual synchronization specifications.
"""
import logging
from typing import Any, Dict, List, Optional
from ai_engine.agents.video.motion_planner import get_motion_for_shot, interpolate_motion_frame

logger = logging.getLogger(__name__)


class VideoDirectorAgent:
    """
    Transforms an audited memory script and curated media into a cinematic shot list
    with dynamic Ken Burns camera motions, transitions, and audio sync.
    """

    def __init__(self, fps: int = 30, resolution: str = "1080x1920"):
        self.fps = fps
        self.resolution = resolution  # Portrait 9:16 default for mobile Flutter playback

    def generate_ken_burns_motion(self, index: int) -> Dict[str, Any]:
        """
        Generates progressive cinematic camera motion parameters:
        alternates between zoom-in, zoom-out, pan-left, and pan-right.
        """
        return get_motion_for_shot(index)

    def compile_shot_list(
        self,
        script: Dict[str, Any],
        scored_items: List[Dict[str, Any]],
        target_duration: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Builds the sequential shot list synchronized with narration phases.
        """
        phases = script.get("phases", [])
        shots: List[Dict[str, Any]] = []
        current_time = 0.0

        for idx, phase in enumerate(phases):
            phase_duration = float(phase.get("target_seconds", target_duration / max(1, len(phases))))
            ref_ids = phase.get("referenced_media_ids", [])

            # Find matching media item or fallback to scored_items[idx]
            media_item = None
            if ref_ids:
                matching = [it for it in scored_items if it.get("id") == ref_ids[0]]
                if matching:
                    media_item = matching[0]

            if not media_item and scored_items:
                media_item = scored_items[idx % len(scored_items)]

            asset_path = media_item.get("file_path", "") if media_item else "/media/memories/placeholder.jpg"
            modality = media_item.get("source_modality", "image") if media_item else "image"

            camera_motion = self.generate_ken_burns_motion(idx)
            transition = "crossfade" if idx > 0 else "cut"
            transition_duration = 0.6 if transition == "crossfade" else 0.0

            shot = {
                "shot_index": idx + 1,
                "phase": phase.get("phase", f"shot_{idx+1}"),
                "start_time": round(current_time, 2),
                "end_time": round(current_time + phase_duration, 2),
                "duration": round(phase_duration, 2),
                "asset_path": asset_path,
                "asset_type": modality,
                "camera_motion": camera_motion,
                "transition_in": transition,
                "transition_duration": transition_duration,
                "narration": phase.get("narration", ""),
                "visual_focus": phase.get("visual_focus", ""),
            }

            shots.append(shot)
            current_time += phase_duration

        return shots

    def build_final_output(
        self,
        script: Dict[str, Any],
        shots: List[Dict[str, Any]],
        audit_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assembles the final production package."""
        total_duration = sum(s["duration"] for s in shots)
        return {
            "title": script.get("title", "Memory Recap"),
            "resolution": self.resolution,
            "fps": self.fps,
            "total_duration": round(total_duration, 2),
            "num_shots": len(shots),
            "total_shots": len(shots),
            "shots": shots,
            "narration_script": script.get("full_narration", ""),
            "audited": audit_result.get("audit_passed", True),
            "hallucination_rate": audit_result.get("hallucination_rate", 0.0),
            "status": "ready_for_render",
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node execution."""
        script = state.get("script", {})
        scored_items = state.get("scored_items", [])
        audit_result = state.get("audit_result", {})
        target_duration = float(state.get("target_duration", 30.0))

        shots = self.compile_shot_list(script, scored_items, target_duration=target_duration)
        final_output = self.build_final_output(script, shots, audit_result)

        return {
            "shot_list": shots,
            "final_output": final_output,
        }
