"""
Video Generation Package
Exports the static video renderer, composition helpers, subtitles, and transitions.
"""
from ai_engine.video_generation.renderer import VideoRenderer
from ai_engine.video_generation.subtitles import render_subtitle_frame
from ai_engine.video_generation.transitions import blend_crossfade, apply_dip_to_black

__all__ = [
    "VideoRenderer",
    "render_subtitle_frame",
    "blend_crossfade",
    "apply_dip_to_black",
]
