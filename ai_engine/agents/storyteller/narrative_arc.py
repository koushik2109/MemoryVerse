"""
Storyteller - Narrative Arc
Defines the 4-phase episodic narrative structure (Hook, Build-up, Climax, Resolution)
and calculates duration, emotional tone, and beat pacing curves.
"""
from typing import List, Dict, Any, Optional

NARRATIVE_PHASES = ["hook", "build_up", "climax", "resolution"]

PHASE_DURATION_RATIOS = {
    "hook": 0.20,        # 20% opening attention grabber
    "build_up": 0.30,    # 30% rising context and progression
    "climax": 0.30,      # 30% peak emotional moment / action
    "resolution": 0.20,  # 20% nostalgic reflection / closure
}

def segment_assets_into_phases(
    assets: List[Dict[str, Any]],
    total_duration: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Distributes available evidence assets across the 4 dramatic phases
    and assigns target pacing durations.
    """
    n_assets = len(assets)
    if n_assets == 0:
        return []

    # Map assets sequentially into phases
    hook_asset = assets[0]
    buildup_asset = assets[1] if n_assets > 1 else hook_asset
    climax_asset = assets[2] if n_assets > 2 else (assets[-1] if n_assets > 0 else hook_asset)
    res_asset = assets[3] if n_assets > 3 else (assets[-1] if n_assets > 0 else hook_asset)

    phases = [
        {
            "phase": "hook",
            "phase_index": 0,
            "target_seconds": round(total_duration * PHASE_DURATION_RATIOS["hook"], 1),
            "tone": "inviting",
            "energy_level": 0.5,
            "primary_asset": hook_asset,
            "referenced_media_ids": [hook_asset.get("id", "")],
        },
        {
            "phase": "build_up",
            "phase_index": 1,
            "target_seconds": round(total_duration * PHASE_DURATION_RATIOS["build_up"], 1),
            "tone": "energetic",
            "energy_level": 0.7,
            "primary_asset": buildup_asset,
            "referenced_media_ids": [buildup_asset.get("id", "")],
        },
        {
            "phase": "climax",
            "phase_index": 2,
            "target_seconds": round(total_duration * PHASE_DURATION_RATIOS["climax"], 1),
            "tone": "triumphant",
            "energy_level": 0.95,
            "primary_asset": climax_asset,
            "referenced_media_ids": [climax_asset.get("id", "")],
        },
        {
            "phase": "resolution",
            "phase_index": 3,
            "target_seconds": round(total_duration * PHASE_DURATION_RATIOS["resolution"], 1),
            "tone": "reflective",
            "energy_level": 0.4,
            "primary_asset": res_asset,
            "referenced_media_ids": [res_asset.get("id", "")],
        },
    ]
    return phases
