"""
Video Director - Motion Planner
Computes dynamic Ken Burns camera trajectories, smooth zoom/pan keyframes,
and transition timings for portrait (9:16) and landscape (16:9) memory videos.
"""
from typing import Dict, Any, List, Tuple


MOTION_PRESETS = [
    {
        "name": "zoom_in_center",
        "type": "zoom_in",
        "start_scale": 1.00,
        "end_scale": 1.15,
        "start_pos": (0.5, 0.5),
        "end_pos": (0.5, 0.5),
        "easing": "ease_in_out",
    },
    {
        "name": "pan_left_to_right",
        "type": "pan_right",
        "start_scale": 1.10,
        "end_scale": 1.10,
        "start_pos": (0.42, 0.5),
        "end_pos": (0.58, 0.5),
        "easing": "linear",
    },
    {
        "name": "zoom_out_reveal",
        "type": "zoom_out",
        "start_scale": 1.18,
        "end_scale": 1.02,
        "start_pos": (0.5, 0.5),
        "end_pos": (0.5, 0.5),
        "easing": "ease_out",
    },
    {
        "name": "pan_right_to_left",
        "type": "pan_left",
        "start_scale": 1.10,
        "end_scale": 1.10,
        "start_pos": (0.58, 0.5),
        "end_pos": (0.42, 0.5),
        "easing": "linear",
    },
]


def get_motion_for_shot(shot_index: int) -> Dict[str, Any]:
    """Returns the motion preset mapped to the sequential shot index."""
    preset = MOTION_PRESETS[shot_index % len(MOTION_PRESETS)]
    return dict(preset)


def interpolate_motion_frame(
    motion: Dict[str, Any],
    progress: float,  # 0.0 to 1.0
) -> Tuple[float, float, float]:
    """
    Interpolates scale and center (x, y) at normalized time progress.
    Returns (current_scale, center_x, center_y).
    """
    p = max(0.0, min(1.0, progress))

    # Smooth cubic ease-in-out curve
    if motion.get("easing") == "ease_in_out":
        t = p * p * (3.0 - 2.0 * p)
    elif motion.get("easing") == "ease_out":
        t = 1.0 - (1.0 - p) * (1.0 - p)
    else:
        t = p

    s_scale = motion.get("start_scale", 1.0)
    e_scale = motion.get("end_scale", 1.15)
    cur_scale = s_scale + (e_scale - s_scale) * t

    s_pos = motion.get("start_pos", (0.5, 0.5))
    e_pos = motion.get("end_pos", (0.5, 0.5))
    cur_x = s_pos[0] + (e_pos[0] - s_pos[0]) * t
    cur_y = s_pos[1] + (e_pos[1] - s_pos[1]) * t

    return cur_scale, cur_x, cur_y
