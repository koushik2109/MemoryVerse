"""
Video Generation Package
Exports VideoRenderer, Ken Burns helpers, procedural AudioSynthesizer, subtitles, and transitions.
"""
from ai_engine.video_generation.renderer import VideoRenderer
from ai_engine.video_generation.ken_burns import get_crop_window
from ai_engine.video_generation.audio_synth import synthesize_ambient_soundtrack, generate_chord_tone
from ai_engine.video_generation.subtitles import render_subtitle_frame
from ai_engine.video_generation.transitions import blend_crossfade, apply_dip_to_black

__all__ = [
    "VideoRenderer",
    "get_crop_window",
    "synthesize_ambient_soundtrack",
    "generate_chord_tone",
    "render_subtitle_frame",
    "blend_crossfade",
    "apply_dip_to_black",
]
