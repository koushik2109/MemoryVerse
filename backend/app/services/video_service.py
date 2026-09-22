import json
import os
import uuid
import logging
import httpx
import math
import numpy as np
from datetime import datetime, timezone
from typing import cast, Any
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageEnhance
from moviepy import (
    AudioArrayClip,
    ColorClip,
    CompositeVideoClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)

from app.core.db import get_supabase_client
from app.config.settings import settings
from app.services import media_intelligence as mi
from ai_engine.video_generation.tts_engine import EmotionTTSEngine
from app.services.job_state_manager import job_state_manager, JobStatus
from app.services.storage_service import storage_service
from ai_engine.emotion.emotion_analyzer import emotion_analyzer
from ai_engine.video_generation.aspect_ratio_composer import (
    AspectRatioComposer,
    BackgroundStrategy,
    TargetAspectRatio,
)
from ai_engine.video_generation.quality_validator import QualityValidator

logger = logging.getLogger(__name__)


def _create_ambient_audio(duration: float, sample_rate: int = 44100) -> AudioArrayClip:
    """
    Synthesizes a soothing, warm ambient chord progression (Cmaj7 -> Am7 -> Fmaj7 -> G)
    with smooth stereo panning, soft attack/decay, and seamless fade-in/fade-out.
    """
    total_samples = max(1, int(duration * sample_rate))
    t_arr = np.linspace(0, duration, total_samples, endpoint=False)
    audio = np.zeros((total_samples, 2), dtype=np.float32)

    chords = [
        [130.81, 196.00, 246.94, 329.63], # Cmaj7
        [110.00, 164.81, 196.00, 261.63], # Am7
        [87.31, 130.81, 164.81, 220.00],  # Fmaj7
        [98.00, 146.83, 196.00, 246.94],  # G
    ]
    chord_len = 4.0

    for i, t in enumerate(t_arr):
        chord_idx = int(t / chord_len) % len(chords)
        chord_t = t % chord_len
        notes = chords[chord_idx]

        env = min(1.0, chord_t / 0.3) * math.exp(-chord_t * 0.25)

        val_l = 0.0
        val_r = 0.0
        for n_idx, freq in enumerate(notes):
            wave = math.sin(2 * math.pi * freq * t) + 0.30 * math.sin(4 * math.pi * freq * t)
            pan = 0.4 + 0.2 * (n_idx % 2)
            val_l += wave * (1.0 - pan)
            val_r += wave * pan

        audio[i, 0] = val_l * env * 0.14
        audio[i, 1] = val_r * env * 0.14

    fade_in_len = min(total_samples, int(1.0 * sample_rate))
    fade_out_len = min(total_samples, int(2.0 * sample_rate))
    if total_samples > fade_in_len:
        audio[:fade_in_len, :] *= np.linspace(0, 1, fade_in_len)[:, None]
    if total_samples > fade_out_len:
        audio[-fade_out_len:, :] *= np.linspace(1, 0, fade_out_len)[:, None]

    return AudioArrayClip(audio, fps=sample_rate)


# ── AI Story Planner ─────────────────────────────────────────────────────────

_STORY_PLAN_SYSTEM = """\
You are an AI editorial director for MemoryVerse, a personal memory journal platform.

Your job is NOT to fill a template. Your job is to understand a real human memory and tell its story.

---
EDITORIAL PROCESS — follow this reasoning order before writing any JSON:

1. UNDERSTAND THE EVENT: What happened here? What kind of memory is this?
2. IDENTIFY THE KEY MOMENTS: Which pieces of media capture genuinely important, unique, or emotional moments?
3. ELIMINATE WEAK CONTENT: Mark low-quality, duplicate, or generic/boring footage as excluded.
4. FIND THE STORY SHAPE: Does the available evidence support an arc (setup → peak → closing)?
   Or is this a single beautiful highlight? Or a montage of several equal moments?
   Adapt the structure to the evidence — do NOT force a rigid template.
5. SELECT PACING: Hero moments deserve longer durations. Quick transitions earn short ones.
   Never pad weak content just to reach a scene count.
6. WRITE EVIDENCE-BASED NARRATION:
   - Base every sentence on what is actually visible or spoken in the evidence.
   - If a transcript shows someone said something meaningful, you may reference that moment.
   - Do NOT invent names, relationships, locations, activities, or emotions.
   - Do NOT reproduce transcript text word-for-word as narration.
   - If there is no evidence for something, do not say it.

---
HARD RULES:
- ONLY use media_id values from the provided manifest. Never invent an ID.
- For video segments: start_time and end_time MUST be within the listed segment boundaries.
- start_time must be < end_time for video scenes.
- duration_seconds must be positive.
- Images have start_time=0.0 and end_time=0.0.
- Preferred motions: zoom_in, zoom_out, pan_left, pan_right, slow_zoom_in, slow_zoom_out, steady, static.
- Preferred transitions: fade, dissolve, cut.
- Return ONLY valid JSON — no markdown fences, no explanation text.
"""


def _build_media_manifest(
    selected_media: list[dict],
    title: str,
    date_str: str,
    location: str | None,
) -> tuple[str, set[str]]:
    """
    Build a rich, evidence-first media manifest string for the editorial director LLM.
    Returns (manifest_text, valid_ids_set).
    """
    valid_ids: set[str] = {str(item.get("id", "")) for item in selected_media}
    lines: list[str] = []

    # Event context header
    loc_str = location or "unknown location"
    lines.append(f"EVENT: \"{title}\" | Date: {date_str or 'unknown date'} | Location: {loc_str}")
    lines.append(f"TOTAL CANDIDATE ITEMS: {len(selected_media)}")
    lines.append("")
    lines.append("MEDIA MANIFEST (one entry per candidate):")

    for item in selected_media:
        mid      = str(item.get("id", ""))
        mtype    = (item.get("media_type") or "image").lower()
        taken    = str(item.get("taken_at") or item.get("created_at") or "")[:10]
        q_score  = item.get("quality_score", 0.5)
        story_v  = item.get("story_value", item.get("selection_score", q_score))
        is_dup   = item.get("is_duplicate", False)
        loc_name = item.get("location_name") or location or ""

        # Description — prefer richer VLM description over CLIP tag list
        description = item.get("description") or ""
        ai_tags     = (item.get("metadata") or {}).get("ai_tags", {})
        if not description:
            scenes_list  = item.get("scenes") or ai_tags.get("scenes", [])
            objects_list = item.get("objects") or ai_tags.get("objects", [])
            desc_parts = []
            if scenes_list:
                desc_parts.append(", ".join(scenes_list[:3]))
            if objects_list:
                desc_parts.append("features: " + ", ".join(objects_list[:3]))
            description = "; ".join(desc_parts) if desc_parts else "no description available"

        people = item.get("people") or ai_tags.get("people_count", "")
        reason = item.get("selection_reason") or ""

        parts = [f"  [id={mid}]"]
        parts.append(f"type={mtype}")
        parts.append(f"date={taken}")
        if loc_name:
            parts.append(f"location={loc_name}")
        parts.append(f"quality={q_score:.2f}")
        parts.append(f"story_value={story_v:.2f}")
        if is_dup:
            parts.append("NOTE=near-duplicate(skip if better version exists)")

        if mtype == "video":
            start_t = float(item.get("start_time", 0.0) or 0.0)
            end_t   = float(item.get("end_time",   0.0) or 0.0)
            dur     = float(item.get("duration", end_t - start_t) or 0.0)
            imp     = item.get("importance_score", story_v)
            parts.append(f"segment=[{start_t:.1f}s-{end_t:.1f}s]")
            parts.append(f"segment_duration={dur:.1f}s")
            parts.append(f"visual_importance={imp:.2f}")

            # Transcript — include timestamped phrases if available
            transcript = item.get("transcript") or []
            speech_text = item.get("speech_text") or ""
            if transcript and isinstance(transcript, list):
                ts_parts = []
                for seg in transcript:
                    ts = f"[{seg.get('start_time', 0):.1f}s-{seg.get('end_time', 0):.1f}s]: \"{seg.get('text', '').strip()}\""
                    ts_parts.append(ts)
                parts.append(f"speech_timestamps={'; '.join(ts_parts)}")
            elif speech_text:
                parts.append(f"speech=\"{speech_text}\"")
            else:
                parts.append("audio=ambient")
        else:
            # Image-specific
            blur_b = item.get("blur_score", "")
            if blur_b != "":
                parts.append(f"blur={float(blur_b):.2f}")

        if description:
            parts.append(f"description=\"{description}\"")
        if people and people not in ("unknown", "no people", ""):
            parts.append(f"people={people}")
        if reason and mtype == "video":
            parts.append(f"editorial_note=\"{reason}\"")

        lines.append("  " + " | ".join(parts))

    return "\n".join(lines), valid_ids


def _validate_and_patch_story_plan(
    raw_plan: dict,
    valid_ids: set[str],
    selected_media: list[dict],
) -> dict | None:
    """
    Validate LLM-generated StoryPlan:
    - Remove scenes referencing non-existent media IDs.
    - Clamp video timestamps to actual segment bounds.
    - Ensure duration_seconds > 0 and start_time < end_time.
    - Ensure required fields exist.
    Returns the patched plan if it has ≥1 valid scenes, else None.
    """
    scenes_in = raw_plan.get("scenes") or []
    if not isinstance(scenes_in, list):
        return None

    # Build a quick lookup: media_id → segment bounds (for video items)
    seg_bounds: dict[str, tuple[float, float]] = {}
    for item in selected_media:
        mid = str(item.get("id", ""))
        mtype = (item.get("media_type") or "image").lower()
        if mtype == "video":
            s = float(item.get("start_time", 0.0) or 0.0)
            e = float(item.get("end_time",   0.0) or 0.0)
            if e > s:
                seg_bounds[mid] = (s, e)

    valid_scenes = []
    for i, scene in enumerate(scenes_in):
        if not isinstance(scene, dict):
            continue
        mid = str(scene.get("media_id", ""))
        if mid not in valid_ids:
            logger.warning(f"StoryPlan scene {scene.get('scene_id', i)} references unknown media_id={mid!r} — dropped.")
            continue

        # Ensure required fields
        if not scene.get("scene_id"):
            scene["scene_id"] = f"s{i+1}"
        scene.setdefault("purpose", "rising_action")
        scene.setdefault("motion", "zoom_in")
        scene.setdefault("transition", "fade")
        scene.setdefault("narration", "")
        scene.setdefault("start_time", 0.0)
        scene.setdefault("end_time",   0.0)

        start_t = float(scene.get("start_time") or 0.0)
        end_t   = float(scene.get("end_time")   or 0.0)
        dur     = float(scene.get("duration_seconds") or 0.0)

        # Clamp video timestamps to actual segment bounds
        if mid in seg_bounds:
            seg_s, seg_e = seg_bounds[mid]
            start_t = max(seg_s, min(start_t, seg_e - 0.5))
            end_t   = max(start_t + 0.5, min(end_t, seg_e))
            if end_t <= start_t:
                end_t = seg_e
            scene["start_time"] = round(start_t, 2)
            scene["end_time"]   = round(end_t,   2)
            if dur <= 0:
                scene["duration_seconds"] = round(end_t - start_t, 2)
        else:
            # Image scene
            scene["start_time"] = 0.0
            scene["end_time"]   = 0.0
            if dur <= 0:
                scene["duration_seconds"] = 4.5

        if scene.get("duration_seconds", 0) <= 0:
            scene["duration_seconds"] = 4.5

        valid_scenes.append(scene)

    if not valid_scenes:
        return None

    raw_plan["scenes"] = valid_scenes
    return raw_plan


def _generate_ai_story_plan(
    title: str,
    date_str: str,
    location: str | None,
    selected_media: list[dict[str, Any]],
    target_duration: float = 30.0,
) -> dict[str, Any]:
    """
    Ask the LLM editorial director to generate a STRUCTURED story plan referencing actual media IDs
    and video segment timestamps. Falls back to an editorial deterministic plan if the LLM is
    unavailable or returns invalid output.
    """
    manifest_text, valid_ids = _build_media_manifest(selected_media, title, date_str, location)

    user_prompt = (
        f"You are reviewing a personal memory titled \"{title}\""
        f"{f' from {date_str}' if date_str else ''}"
        f"{f' at {location}' if location else ''}.\n\n"
        f"Target total video duration: {target_duration} seconds.\n\n"
        f"{manifest_text}\n\n"
        "Follow your editorial process:\n"
        "1. Understand what this memory is about based on the evidence above.\n"
        "2. Identify the most meaningful, unique, and visually strong moments.\n"
        "3. Decide which media to skip (duplicates, low-quality, redundant).\n"
        "4. Choose an appropriate story structure for the available evidence.\n"
        "5. Write evidence-grounded narration for each selected scene.\n\n"
        "Return ONLY this JSON — no markdown, no explanation:\n"
        "{\n"
        '  "title": "A specific, meaningful title for this memory",\n'
        '  "narrative_summary": "One or two sentences describing what actually happened in this memory",\n'
        '  "scenes": [\n'
        '    {\n'
        '      "scene_id": "s1",\n'
        '      "purpose": "opening",\n'
        '      "media_id": "<exact id from manifest>",\n'
        '      "start_time": 0.0,\n'
        '      "end_time": 0.0,\n'
        '      "duration_seconds": 5.0,\n'
        '      "motion": "slow_zoom_in",\n'
        '      "transition": "fade",\n'
        '      "narration": "An evidence-based sentence about this specific moment"\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    raw_plan: dict[str, Any] | None = None
    try:
        if settings.LLM_API_KEY and settings.LLM_PROVIDER != "none":
            from app.services.ai_service import _call_llm
            resp = _call_llm(
                prompt=user_prompt,
                context=_STORY_PLAN_SYSTEM,
            )
            raw_text = resp.text.strip()
            # Strip accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()
            raw_plan = json.loads(raw_text)
    except Exception as e:
        logger.warning(f"AI story plan generation failed ({type(e).__name__}: {e}), using deterministic fallback.")

    # Validate and patch LLM output
    if raw_plan:
        validated = _validate_and_patch_story_plan(raw_plan, valid_ids, selected_media)
        if validated:
            return validated
        logger.warning("AI story plan had no valid scenes after validation. Using deterministic fallback.")

    # ── Deterministic Editorial Fallback ────────────────────────────────────
    # Quality floor: skip items below 0.30 story_value unless they are the
    # only content available (edge case: single-item vault).
    QUALITY_FLOOR = 0.30
    editorial_candidates: list[dict[str, Any]] = []
    seen_video_counts: dict[str, int] = {}
    for item in selected_media:
        mid   = str(item.get("id", ""))
        mtype = (item.get("media_type") or "image").lower()
        sv    = item.get("story_value", item.get("quality_score", 0.5))
        if mtype == "video":
            c = seen_video_counts.get(mid, 0)
            if c >= 2:
                continue
            seen_video_counts[mid] = c + 1
        if item.get("is_duplicate", False):
            continue
        if sv < QUALITY_FLOOR:
            logger.debug(f"Fallback: skipping {mid} (story_value={sv:.2f} < quality floor {QUALITY_FLOOR})")
            continue
        editorial_candidates.append(item)

    # If quality floor eliminated everything, revert to unfiltered (minus duplicates only)
    if not editorial_candidates:
        editorial_candidates = [i for i in selected_media if not i.get("is_duplicate", False)]

    # Sort by story_value descending so best content leads
    editorial_candidates.sort(
        key=lambda x: x.get("story_value", x.get("quality_score", 0.5)),
        reverse=True,
    )

    motions = ["zoom_in", "pan_left", "zoom_out", "pan_right", "slow_zoom_in"]
    scenes_fb: list[dict[str, Any]] = []
    total = len(editorial_candidates)
    title_dur = 2.5
    avail_scene_dur = max(5.0, float(target_duration) - title_dur)
    base_dur = max(2.5, min(8.0, avail_scene_dur / max(1, total))) if total > 0 else 4.0

    raw_durations: list[float] = []
    for i, item in enumerate(editorial_candidates):
        mid   = str(item.get("id", ""))
        mtype = (item.get("media_type") or "image").lower()
        q     = item.get("quality_score", item.get("story_value", 0.5))

        start_t = float(item.get("start_time", 0.0) or 0.0)
        end_t   = float(item.get("end_time",   0.0) or 0.0)

        if i == 0:
            purpose = "opening"
            dur     = max(2.5, min(7.0, base_dur * 1.1))
        elif i == total - 1:
            purpose = "closing"
            dur     = max(2.5, min(7.0, base_dur * 1.0))
        elif q >= 0.72 or mtype == "video":
            purpose = "main_moment"
            dur     = max(3.0, min(8.0, base_dur * 1.25))
        else:
            purpose = "rising_action"
            dur     = max(2.0, min(6.0, base_dur * 0.9))

        if mtype == "video":
            if end_t > start_t:
                dur = min(8.0, max(2.5, round(end_t - start_t, 2)))
            else:
                dur = max(dur, 4.0)
                end_t = start_t + dur

        raw_durations.append(dur)

        # Build evidence-based human narration (never echoing raw prompt commands)
        description = item.get("description") or item.get("selection_reason") or ""
        speech_text = item.get("speech_text") or ""
        if speech_text:
            words = speech_text.strip().split()
            short = " ".join(words[:8]) + ("..." if len(words) > 8 else "")
            narration = f"A memorable moment: \"{short}\""
        elif description and description not in ("Action footage moment", "no description available"):
            clean_desc = description.split(";")[0].split(":")[0].strip().capitalize()
            if len(clean_desc) > 80:
                clean_desc = clean_desc[:77] + "..."
            narration = clean_desc
        else:
            narrative_arcs = [
                "Every meaningful journey begins with a quiet, authentic moment.",
                "Gentle light and timeless connections captured in focus.",
                "Laughter, warmth, and the joy of shared experiences.",
                "Reflecting on precious moments that stay with us forever.",
            ]
            narration = narrative_arcs[i % len(narrative_arcs)]

        scenes_fb.append({
            "scene_id":         f"s{i+1}",
            "purpose":          purpose,
            "media_id":         mid,
            "start_time":       round(start_t, 2),
            "end_time":         round(end_t,   2),
            "duration_seconds": dur,
            "motion":           "static" if mtype == "video" else motions[i % len(motions)],
            "transition":       "fade",
            "narration":        narration,
        })

    # Duration normalization: enforce exact target duration alignment
    total_raw = sum(raw_durations)
    if total_raw > 0 and abs(total_raw - avail_scene_dur) > 0.5:
        norm_factor = avail_scene_dur / total_raw
        for sc in scenes_fb:
            sc["duration_seconds"] = max(2.0, round(sc["duration_seconds"] * norm_factor, 2))

    loc_desc = f" at {location}" if location else ""
    return {
        "title":            title,
        "narrative_summary": f"A collection of moments from {title}{loc_desc}.",
        "scenes":           scenes_fb,
    }



def _create_ai_cinematic_scene_clip(
    img_path: str,
    duration: float = 3.5,
    target_size: tuple[int, int] = (1280, 720),
    motion_type: str = "zoom_in",
    caption: str | None = None
) -> VideoClip:
    """
    Ultra-Fast AI Motion Synthesizer:
    Pre-renders keyframes ONCE and uses high-speed Numpy matrix interpolation.
    """
    tw, th = target_size
    orig_img = Image.open(img_path).convert("RGB")
    ow, oh = orig_img.size

    # 1. Adaptive cinematic background (never stretch or distort)
    bg_base = AspectRatioComposer.generate_background(
        orig_img, tw, th, strategy=BackgroundStrategy.ADAPTIVE_CINEMATIC.value
    )

    # 2. Foreground base with aspect-ratio preservation & cinematic grading
    layout = AspectRatioComposer.calculate_layout(ow, oh, tw, th, margin_pct=0.07)
    fg_w, fg_h = layout["fg_w"], layout["fg_h"]
    fg_base = orig_img.resize((fg_w, fg_h), cast(Any, Image.Resampling.LANCZOS))

    # Apply cinematic grade: rich contrast, vibrant warm highlights
    try:
        enh_con = ImageEnhance.Contrast(fg_base)
        fg_base = enh_con.enhance(1.08)
        enh_col = ImageEnhance.Color(fg_base)
        fg_base = enh_col.enhance(1.10)
    except Exception:
        pass

    border_img = Image.new("RGBA", (fg_w + 8, fg_h + 8), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(cast(Any, border_img))
    b_draw.rectangle([0, 0, fg_w + 7, fg_h + 7], outline=(255, 255, 255, 80), width=2)
    border_img.paste(cast(Any, fg_base), (4, 4))

    # Pre-render Start Keyframe (p = 0.0)
    z0 = 1.00 if "zoom_in" in motion_type else (1.10 if "zoom_out" in motion_type else 1.05)
    sx0 = -15 if "pan_right" in motion_type else (15 if "pan_left" in motion_type else 0)
    sy0 = -8 if "zoom_in" in motion_type else (8 if "zoom_out" in motion_type else 0)

    w0, h0 = int(border_img.width * z0), int(border_img.height * z0)
    fg0 = border_img.resize((w0, h0), cast(Any, Image.Resampling.BILINEAR)) if z0 != 1.0 else border_img
    canvas0 = bg_base.copy()
    canvas0.paste(fg0, ((tw - w0) // 2 + sx0, (th - h0) // 2 + sy0), fg0)
    # Pre-render End Keyframe (p = 1.0)
    z1 = 1.10 if "zoom_in" in motion_type else (1.00 if "zoom_out" in motion_type else 1.05)
    sx1 = 15 if "pan_right" in motion_type else (-15 if "pan_left" in motion_type else 0)
    sy1 = 8 if "zoom_in" in motion_type else (-8 if "zoom_out" in motion_type else 0)

    w1, h1 = int(border_img.width * z1), int(border_img.height * z1)
    fg1 = border_img.resize((w1, h1), cast(Any, Image.Resampling.BILINEAR)) if z1 != 1.0 else border_img
    canvas1 = bg_base.copy()
    canvas1.paste(fg1, ((tw - w1) // 2 + sx1, (th - h1) // 2 + sy1), fg1)

    if caption and caption.strip():
        try:
            from ai_engine.video_generation.subtitles import render_subtitle_frame
            canvas0 = render_subtitle_frame(cast(Any, canvas0), caption.strip(), font_size=28, bottom_margin=int(th * 0.12))
            canvas1 = render_subtitle_frame(cast(Any, canvas1), caption.strip(), font_size=28, bottom_margin=int(th * 0.12))
        except Exception as sub_err:
            logger.debug(f"Subtitle overlay skipped: {sub_err}")

    arr0 = np.array(canvas0, dtype=np.float32)
    arr1 = np.array(canvas1, dtype=np.float32)

    def make_frame(t: float) -> Any:
        p = max(0.0, min(1.0, t / duration))
        ease_p = 0.5 - 0.5 * math.cos(math.pi * p)

        # High-speed numpy vector interpolation
        f = (1.0 - ease_p) * arr0 + ease_p * arr1

        # Smooth Fade-in / Fade-out
        fade_dur = 0.35
        if t < fade_dur:
            f *= (t / fade_dur)
        elif t > (duration - fade_dur):
            f *= max(0.0, (duration - t) / fade_dur)

        return f.astype(np.uint8)

    return VideoClip(make_frame, duration=duration)


def _synthesize_cinematic_title_and_tagline(raw_title: str, emotion: str = "nostalgic") -> tuple[str, str]:
    """
    Synthesizes an elevated, cinematic title and evocative poetic tagline for the opening card.
    Uses Gemini / LLM when available to create artistic titles, and falls back to a deep
    semantic transformer that turns raw query prompts (e.g. 'Create A Memory Filtering The Images Where There Is Water In It')
    into elegant titles (e.g. 'Water Reflections', 'Echoes of light and serene waters').
    """
    # 1. Attempt Gemini / LLM synthesis if configured
    try:
        if settings.LLM_API_KEY and settings.LLM_PROVIDER != "none":
            from app.services.ai_service import _call_llm
            prompt = (
                f"You are an award-winning cinematic film director creating an opening title card for a personal memory movie.\n"
                f"Memory Title or User Query: \"{raw_title}\"\n"
                f"Emotion / Mood: \"{emotion}\"\n\n"
                "Tasks:\n"
                "1. Craft a concise, poetic, cinematic movie title (2 to 4 words, Title Case, max 26 characters). Never echo user command phrases like 'create a memory' or 'filter images'.\n"
                "2. Craft a poetic, evocative opening tagline (5 to 8 words).\n\n"
                "Return ONLY valid JSON:\n"
                "{\"title\": \"...\", \"tagline\": \"...\"}"
            )
            resp = _call_llm(prompt=prompt, context="")
            raw_text = resp.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()
            data = json.loads(raw_text)
            llm_title = data.get("title", "").strip()
            llm_tagline = data.get("tagline", "").strip()
            if llm_title and llm_tagline:
                if len(llm_title) > 28:
                    llm_title = llm_title[:26].rsplit(" ", 1)[0]
                return llm_title, llm_tagline
    except Exception as e:
        logger.debug(f"Gemini title synthesis fallback: {e}")

    # 2. Rich Semantic Transformer Fallback
    clean_t = raw_title.strip()
    lower = clean_t.lower()

    for prefix in [
        "create a memory filtering the images where there is ",
        "create a memory filtering the images where ",
        "create a memory filtering ",
        "create a memory about ",
        "create a memory of ",
        "create a memory ",
        "filter the images where there is ",
        "filter images with ",
        "filter where there is ",
        "filter images where ",
        "find images with ",
    ]:
        if lower.startswith(prefix):
            remainder = clean_t[len(prefix):].strip(" .'\":")
            clean_t = remainder.title()
            lower = clean_t.lower()
            break

    if "where there is" in lower:
        parts = lower.split("where there is", 1)
        clean_t = parts[1].strip(" .'\":").title() + " Moments"
        lower = clean_t.lower()

    # Topic-specific poetic mapping
    topic_titles = {
        "water": ("Water Reflections", "Echoes of light and serene waters"),
        "ocean": ("Ocean Whispers", "Tides of peace and endless blue"),
        "sea": ("Coastal Echoes", "Where the waves meet timeless horizons"),
        "beach": ("Golden Shores", "Sunlight, sand, and cherished days"),
        "sunset": ("Golden Hour Horizons", "Where the sun meets timeless serenity"),
        "sunrise": ("Dawn of New Days", "First light upon cherished memories"),
        "mountain": ("Highland Trails", "Journeys high above the clouds"),
        "hiking": ("Paths Less Traveled", "Finding beauty in every stride"),
        "family": ("Generations of Joy", "Cherished bonds that never fade"),
        "kids": ("Childhood Wonders", "Laughter that lights the world"),
        "friends": ("Shared Laughter", "Unforgettable nights and bright smiles"),
        "travel": ("Wanderlust Chronicles", "Paths traveled and stories found"),
        "trip": ("Journey Through Time", "Moments discovered along the way"),
        "vacation": ("Summer Getaways", "Carefree days and warm breezes"),
        "wedding": ("Everlasting Grace", "Two souls and a lifetime of love"),
        "love": ("Moments in Devotion", "Gentle whispers of the heart"),
        "dog": ("Faithful Companions", "Little paws and boundless love"),
        "pet": ("Gentle Companions", "Warm purrs and playful days"),
        "food": ("Flavors & Gatherings", "Warmth around the shared table"),
        "nature": ("Echoes of Nature", "Wild whispers in the gentle breeze"),
    }

    for keyword, (mapped_title, mapped_tagline) in topic_titles.items():
        if keyword in lower:
            return mapped_title, mapped_tagline

    if len(clean_t) > 28:
        clean_t = clean_t[:26].rsplit(" ", 1)[0] + "..."

    e = (emotion or "nostalgic").lower()
    taglines = {
        "calm": "Quiet moments held gently in time",
        "nostalgic": "Echoes of cherished days",
        "joyful": "Bright smiles and joyful laughter",
        "reflective": "Reflecting on journeys that shape us",
        "dramatic": "An unforgettable story unfolds",
        "celebratory": "Moments of triumphant celebration",
        "serene": "Peaceful memories by the waterside",
    }
    tagline = taglines.get(e, "Timeless moments held forever")
    return clean_t or "Cherished Moments", tagline


def _create_ai_title_card_clip(
    title: str,
    subtitle: str,
    duration: float = 2.5,
    target_size: tuple[int, int] = (1280, 720),
    bg_img_path: str | None = None,
    emotion: str = "nostalgic",
) -> VideoClip:
    """
    Ultra-Cinematic Opening Title Card Clip with animated Ken Burns zoom,
    luminous radial ambient lighting, floating golden bokeh, double-bordered
    glass badge, diamond accent, and multi-layer typography.
    """
    tw, th = target_size
    clean_title, tagline = _synthesize_cinematic_title_and_tagline(title, emotion=emotion)

    # 1. Base image background with blur and rich dark vignette
    if bg_img_path and os.path.exists(bg_img_path):
        with Image.open(bg_img_path) as orig:
            orig = orig.convert("RGB")
            ow, oh = orig.size
            scale = max(tw / ow, th / oh) * 1.18
            bw, bh = int(ow * scale), int(oh * scale)
            bg = orig.resize((bw, bh), cast(Any, Image.Resampling.BILINEAR))
            l = (bw - tw) // 2
            t = (bh - th) // 2
            bg = bg.crop((l, t, l + tw, t + th)).filter(cast(Any, ImageFilter.GaussianBlur(radius=22)))
            # Dark cinematic wash
            dark = Image.new("RGBA", (tw, th), (15, 8, 24, 210))
            bg = Image.alpha_composite(cast(Any, bg.convert("RGBA")), dark).convert("RGB")
    else:
        bg = Image.new("RGB", (tw, th), color=(18, 10, 28))

    # Pre-render End Zoom Keyframe (bg zoomed 1.06x)
    bw1, bh1 = int(tw * 1.06), int(th * 1.06)
    bg1 = bg.resize((bw1, bh1), cast(Any, Image.Resampling.BILINEAR))
    l1 = (bw1 - tw) // 2
    t1 = (bh1 - th) // 2
    bg1 = bg1.crop((l1, t1, l1 + tw, t1 + th))

    # 2. Add Luminous Center Radial Glow (Cinematic spotlight effect)
    cx, cy = tw // 2, th // 2
    glow = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow)
    max_r = int(min(tw, th) * 0.48)
    for r in range(max_r, 0, -25):
        alpha = int(40 * (1.0 - r / max_r))
        g_draw.ellipse([cx - r, cy - 25 - r, cx + r, cy - 25 + r], fill=(232, 85, 125, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=30))

    bg = Image.alpha_composite(cast(Any, bg.convert("RGBA")), glow).convert("RGB")
    bg1 = Image.alpha_composite(cast(Any, bg1.convert("RGBA")), glow).convert("RGB")

    # Typography setup
    try:
        font_tag = ImageFont.truetype("arial.ttf", size=max(15, int(th * 0.028)))
        font_title = ImageFont.truetype("arial.ttf", size=max(28, int(th * 0.068)))
        font_tagline = ImageFont.truetype("arial.ttf", size=max(18, int(th * 0.036)))
        font_date = ImageFont.truetype("arial.ttf", size=max(15, int(th * 0.028)))
    except Exception:
        try:
            font_tag = ImageFont.load_default(size=max(15, int(th * 0.028)))
            font_title = ImageFont.load_default(size=max(28, int(th * 0.068)))
            font_tagline = ImageFont.load_default(size=max(18, int(th * 0.036)))
            font_date = ImageFont.load_default(size=max(15, int(th * 0.028)))
        except TypeError:
            font_tag = font_title = font_tagline = font_date = ImageFont.load_default()

    # Pre-defined floating golden bokeh particles
    bokeh_particles = [
        (cx - 280, cy - 140, 6, 80),
        (cx + 260, cy - 120, 8, 70),
        (cx - 320, cy + 80, 5, 60),
        (cx + 310, cy + 110, 9, 75),
        (cx - 150, cy - 180, 4, 90),
        (cx + 170, cy - 190, 5, 85),
        (cx - 80, cy + 160, 7, 65),
        (cx + 90, cy + 170, 6, 70),
    ]

    def draw_elements(target_bg: Image.Image, accent_progress: float = 1.0, drift: int = 0) -> Image.Image:
        canvas = target_bg.copy()
        draw = ImageDraw.Draw(cast(Any, canvas))

        # Floating subtle golden bokeh motes
        for bx, by, br, b_alpha in bokeh_particles:
            px = bx + drift
            py = by - drift // 2
            draw.ellipse([px - br, py - br, px + br, py + br], fill=(255, 215, 170))

        # 1. Top Glassmorphic Badge Pill
        badge_text = "✦   M E M O R Y V E R S E   C I N E M A   ✦"
        pill_w = max(260, int(tw * 0.42))
        pill_h = 34
        pill_x0 = cx - pill_w // 2
        pill_y0 = cy - 95
        # Frosted glass background
        draw.rounded_rectangle(
            [pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h],
            radius=17,
            fill=(38, 18, 48),
            outline=(255, 175, 195),
            width=1,
        )
        draw.text((cx, pill_y0 + 17), badge_text, font=cast(Any, font_tag), fill=(255, 195, 215), anchor="mm")

        # 2. Main Title with Ambient Glow and Deep 3D Shadow
        # Ambient rim glow
        draw.text((cx, cy - 26), clean_title, font=cast(Any, font_title), fill=(180, 60, 95), anchor="mm")
        # Deep drop shadow
        draw.text((cx + 2, cy - 23), clean_title, font=cast(Any, font_title), fill=(8, 4, 12), anchor="mm")
        # Crisp champagne white hero text
        draw.text((cx, cy - 25), clean_title, font=cast(Any, font_title), fill=(255, 253, 248), anchor="mm")

        # 3. Expanding Glowing Rose Accent Bar with Center Diamond
        max_bar_w = min(int(tw * 0.52), 340)
        bar_w = int(max_bar_w * max(0.0, min(1.0, accent_progress)))
        by = cy + 18
        if bar_w > 12:
            bx0 = cx - bar_w // 2
            bx1 = cx + bar_w // 2
            # Left wing
            draw.line([(bx0, by), (cx - 10, by)], fill=(232, 85, 125), width=2)
            draw.line([(bx0 + 6, by), (cx - 12, by)], fill=(255, 195, 210), width=1)
            # Right wing
            draw.line([(cx + 10, by), (bx1, by)], fill=(232, 85, 125), width=2)
            draw.line([(cx + 12, by), (bx1 - 6, by)], fill=(255, 195, 210), width=1)
        # Center glowing gold diamond ornament
        draw.polygon([(cx, by - 6), (cx + 6, by), (cx, by + 6), (cx - 6, by)], fill=(255, 220, 160))

        # 4. Poetic Tagline
        draw.text((cx, cy + 44), tagline, font=cast(Any, font_tagline), fill=(238, 226, 246), anchor="mm")

        # 5. Date / Location Pill
        if subtitle:
            draw.text((cx, cy + 80), subtitle, font=cast(Any, font_date), fill=(195, 185, 212), anchor="mm")

        return canvas

    card0 = draw_elements(bg, accent_progress=0.15, drift=0)
    card1 = draw_elements(bg1, accent_progress=1.0, drift=8)
    arr0 = np.array(card0, dtype=np.float32)
    arr1 = np.array(card1, dtype=np.float32)

    def make_frame(t: float) -> Any:
        p = max(0.0, min(1.0, t / duration))
        ease_p = 0.5 - 0.5 * math.cos(math.pi * p)
        f = (1.0 - ease_p) * arr0 + ease_p * arr1

        # Smooth cinematic fade-in and dissolve out
        fade_dur = 0.38
        if t < fade_dur:
            f *= (t / fade_dur)
        elif t > (duration - fade_dur):
            f *= max(0.0, (duration - t) / fade_dur)
        return f.astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    sr = 44100
    n_samples = int(duration * sr)
    silent_audio = AudioArrayClip(np.zeros((n_samples, 2), dtype=np.float32), fps=sr)
    return clip.with_audio(silent_audio)


def _create_ai_outro_card_clip(
    duration: float = 1.5,
    target_size: tuple[int, int] = (1280, 720),
) -> VideoClip:
    """
    Polished Outro Resolution Card Clip ensuring timeline synchronization.
    """
    tw, th = target_size
    bg = Image.new("RGB", (tw, th), color=(14, 8, 22))
    draw = ImageDraw.Draw(cast(Any, bg))

    try:
        font_tag = ImageFont.truetype("arial.ttf", size=max(18, int(th * 0.035)))
    except Exception:
        try:
            font_tag = ImageFont.load_default(size=max(18, int(th * 0.035)))
        except TypeError:
            font_tag = ImageFont.load_default()

    draw.text((tw // 2, th // 2), "✦  MemoryVerse  ✦", font=cast(Any, font_tag), fill=(215, 185, 235), anchor="mm")
    arr = np.array(bg, dtype=np.float32)

    def make_frame(t: float) -> Any:
        f = arr.copy()
        fade_dur = min(0.4, duration / 2)
        if t < fade_dur:
            f *= (t / fade_dur)
        elif t > (duration - fade_dur):
            f *= max(0.0, (duration - t) / fade_dur)
        return f.astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    sr = 44100
    n_samples = int(duration * sr)
    silent_audio = AudioArrayClip(np.zeros((n_samples, 2), dtype=np.float32), fps=sr)
    return clip.with_audio(silent_audio)



def _try_generate_external_ai_video(prompt: str, image_url: str) -> str | None:
    """
    If AI_VIDEO_API_KEY or REPLICATE_API_KEY is configured in settings,
    call an external AI Video Generation model (e.g. Replicate Stable Video Diffusion / Luma / Runway).
    Returns downloaded video file path if successful, else None.
    """
    api_key = settings.AI_VIDEO_API_KEY or settings.REPLICATE_API_KEY
    if not api_key:
        return None

    try:
        import time
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        # Replicate Stable Video Diffusion API endpoint
        body = {
            "version": "3f0457e4619da25d0d9834164b73b069d2ebc2b0b464a93edb8d9600e129188e",
            "input": {
                "input_image": image_url,
                "motion_bucket_id": 127,
                "fps": 14,
                "video_length": "14_frames_with_svd"
            }
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post("https://api.replicate.com/v1/predictions", json=body, headers=headers)
            if resp.status_code != 201:
                logger.warning(f"Replicate API error: {resp.text}")
                return None
            
            pred = resp.json()
            pred_id = pred.get("id")
            if not pred_id:
                return None

            # Poll prediction until finished (max 60s)
            for _ in range(12):
                time.sleep(5)
                poll_resp = client.get(f"https://api.replicate.com/v1/predictions/{pred_id}", headers=headers)
                pdata = poll_resp.json()
                if pdata.get("status") == "succeeded":
                    out_url = pdata.get("output")
                    if isinstance(out_url, list):
                        out_url = out_url[0]
                    if out_url:
                        vid_bytes = client.get(out_url).content
                        tmp_path = f"/tmp/ai_gen_{uuid.uuid4().hex}.mp4"
                        with open(tmp_path, "wb") as vf:
                            vf.write(vid_bytes)
                        return tmp_path
                elif pdata.get("status") == "failed":
                    break
    except Exception as e:
        logger.warning(f"External AI video generation failed, falling back to AI Motion Synthesizer: {e}")

    return None


class VideoService:
    @staticmethod
    def execute_job_by_id(job_id: str) -> None:
        """Called by background video worker pool for durable, recoverable execution."""
        job = job_state_manager.get_job(job_id)
        if not job:
            logger.error(f"Cannot execute job {job_id}: not found in JobStateManager.")
            return

        # Check if job was already rendered and only needs upload retry (preserves artifact!)
        if job.staged_file_path and os.path.exists(job.staged_file_path) and job.status in ("failed", "retrying"):
            logger.info(f"Job {job_id} has existing staged video {job.staged_file_path}. Retrying upload directly without re-rendering.")
            filename = os.path.basename(job.staged_file_path).replace(f"{job_id}_", "")
            job_state_manager.transition_stage(job_id, "uploading", task_description="Retrying upload of staged video reel...")
            success, public_url, storage_path, err_code = storage_service.upload_video_with_retry(
                job_id=job.id,
                user_id=job.user_id,
                staged_filepath=job.staged_file_path,
                output_filename=filename,
            )
            if success and public_url:
                job_state_manager.transition_stage(job_id, "finalizing", task_description="Saving video record...")
                supabase = get_supabase_client()
                file_size = os.path.getsize(job.staged_file_path)
                media_data = {
                    "memory_id": job.memory_id,
                    "owner_id": job.user_id,
                    "filename": filename,
                    "storage_path": storage_path,
                    "url": public_url,
                    "thumbnail_url": None,
                    "media_type": "video",
                    "file_size": file_size,
                    "mime_type": "video/mp4",
                    "duration": 30,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                insert_res = supabase.table("media").insert(media_data).execute()
                if insert_res.data:
                    new_media_records = cast(list[dict[str, Any]], insert_res.data)
                    new_media = new_media_records[0]
                    job_state_manager.complete_job(job.id, result_media_id=str(new_media["id"]), result_url=public_url)
                    storage_service.cleanup_staged_file(job.staged_file_path)
                    return
            else:
                job_state_manager.fail_job(job.id, error_code=err_code or "RETRY_UPLOAD_FAILED", error_message="Upload retry failed. Artifact preserved.")
                return

        VideoService.process_video_job(
            job_id=job.id,
            memory_id=job.memory_id,
            user_id=job.user_id,
            dimension=job.dimension,
            mood=job.mood,
            selected_media_ids=job.selected_media_ids,
        )

    @staticmethod
    def process_video_job(
        job_id: str,
        memory_id: str,
        user_id: str,
        dimension: str | None = None,
        mood: str | None = "calm",
        selected_media_ids: list[str] | None = None,
    ) -> None:
        import tempfile
        import shutil

        supabase = get_supabase_client()

        def update_job_status(status: str, error_msg: str | None = None, result_media_id: str | None = None):
            now = datetime.now(timezone.utc).isoformat()
            data = {"status": status, "updated_at": now}
            if error_msg:
                data["error_message"] = error_msg
            if result_media_id:
                data["result_media_id"] = result_media_id
            try:
                supabase.table("video_jobs").update(data).eq("id", job_id).execute()
            except Exception as e:
                logger.debug(f"DB update fallback: {e}")

        update_job_status("processing")
        job_state_manager.transition_stage(
            job_id, "initializing", task_description="Initializing generation environment and fetching memory details..."
        )

        tmp_dir = tempfile.mkdtemp(prefix="memoryverse_reel_")

        # Resolve full specification from job_state_manager or DB script_metadata
        cached_job = job_state_manager.get_job(job_id)
        spec: dict[str, Any] = {}
        if cached_job and cached_job.specification:
            spec = cached_job.specification
        else:
            try:
                db_job = supabase.table("video_jobs").select("script_metadata").eq("id", job_id).execute()
                if db_job.data and isinstance(db_job.data, list) and len(db_job.data) > 0:
                    first_record = cast(dict[str, Any], db_job.data[0])
                    meta = first_record.get("script_metadata")
                    if isinstance(meta, dict):
                        found_spec = meta.get("specification")
                        if isinstance(found_spec, dict):
                            spec = cast(dict[str, Any], found_spec)
                        else:
                            spec = cast(dict[str, Any], meta)
            except Exception as e:
                logger.debug(f"Could not load job spec: {e}")

        if not dimension and spec.get("dimension"):
            dimension = spec.get("dimension")
        if (not mood or mood == "calm") and spec.get("mood"):
            mood = spec.get("mood")
        if not selected_media_ids and spec.get("selected_media_ids"):
            selected_media_ids = spec.get("selected_media_ids")

        target_duration = float(spec.get("target_duration") or spec.get("duration_seconds") or 30.0)
        quality_profile = str(spec.get("quality_profile") or "balanced").lower()
        music_style = str(spec.get("music_style") or mood or "calm").lower()

        try:
            # 1. Fetch memory details for title & date
            mem_res = supabase.table("memories").select("*").eq("id", memory_id).execute()
            memory_title: str = "My Memory"
            memory_date: str = ""
            vault_id = None
            if mem_res.data:
                mem_records = cast(list[dict[str, Any]], mem_res.data)
                mem = mem_records[0]
                memory_title = str(mem.get("title") or "My Memory")
                raw_date = str(mem.get("event_date") or mem.get("created_at") or "")
                if raw_date:
                    try:
                        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        memory_date = dt.strftime("%B %d, %Y")
                    except Exception:
                        memory_date = raw_date[:10]
                vault_id = mem.get("vault_id")

            job_state_manager.transition_stage(
                job_id, "retrieving_memory", task_description=f"Retrieving source assets for '{memory_title}'..."
            )

            # ── 2. Fetch ALL source media (with retry grace period for concurrent uploads) ────
            import time
            all_media: list[dict[str, Any]] = []
            for attempt in range(5):
                res = supabase.table("media")\
                    .select("*")\
                    .eq("memory_id", memory_id)\
                    .order("created_at", desc=False)\
                    .limit(60)\
                    .execute()

                all_media = [
                    item for item in cast(list[dict[str, Any]], res.data or [])
                    if "reels/" not in (item.get("storage_path") or "")
                ]
                if all_media:
                    break
                if attempt < 4:
                    time.sleep(2.0)

            if not all_media:
                job_state_manager.fail_job(job_id, "NO_MEDIA_FOUND", "No source media items found in this memory.")
                update_job_status("failed", "No source media items found in this memory.")
                return

            # If user selected specific pictures/media IDs, filter to those
            if selected_media_ids:
                selected_set = {mid for mid in selected_media_ids}
                filtered = [m for m in all_media if str(m.get("id")) in selected_set]
                if filtered:
                    all_media = filtered

            all_media_ids = [str(item.get("id", "")) for item in all_media]

            # ── 3. Fetch CLIP embeddings for all media (already computed post-upload) ──
            embeddings_map: dict[str, list[float]] = {}
            try:
                emb_res = supabase.table("media_embeddings")\
                    .select("media_id, embedding")\
                    .in_("media_id", all_media_ids)\
                    .execute()
                emb_records = cast(list[dict[str, Any]], emb_res.data or [])
                for row in emb_records:
                    mid = str(row.get("media_id", ""))
                    emb = row.get("embedding") or []
                    if isinstance(emb, list):
                        embeddings_map[mid] = [float(x) for x in emb]
            except Exception as emb_err:
                logger.warning(f"Could not fetch CLIP embeddings: {emb_err}")

            # ── 4. Near-duplicate detection (uses CLIP cosine similarity) ─────────────
            cluster_map = mi.detect_near_duplicates(all_media_ids, embeddings_map, threshold=0.92)

            # ── 5. Download only the pre-selected candidates ──────────────────────────
            # We download all items first to enable actual quality scoring,
            # then let the intelligence pipeline make its final selection.
            downloaded: dict[str, str] = {}  # media_id → local file path

            with httpx.Client(timeout=25.0) as http_client:
                for idx, item in enumerate(all_media):
                    mid = str(item.get("id", ""))
                    mtype = (item.get("media_type") or "image").lower()
                    ext = ".mp4" if mtype == "video" else ".jpg"
                    file_path = os.path.join(tmp_dir, f"media_{idx:03d}_{mid[:8]}{ext}")

                    saved = False
                    storage_path = item.get("storage_path") or ""
                    if storage_path:
                        try:
                            file_bytes = supabase.storage.from_("memories").download(storage_path)
                            if file_bytes:
                                with open(file_path, "wb") as f:
                                    f.write(file_bytes)
                                saved = True
                        except Exception as dl_err:
                            logger.warning(f"Storage download failed ({storage_path}): {dl_err}")

                    if not saved:
                        url = item.get("url") or item.get("thumbnail_url") or ""
                        if url:
                            try:
                                r = http_client.get(url)
                                r.raise_for_status()
                                with open(file_path, "wb") as f:
                                    f.write(r.content)
                                saved = True
                            except Exception as http_err:
                                logger.warning(f"HTTP download failed ({url}): {http_err}")

                    if saved:
                        downloaded[mid] = file_path
                    else:
                        logger.warning(f"Skipping media {mid} — could not download")

            if not downloaded:
                update_job_status("failed", "No media could be downloaded.")
                return

            # ── 6. Score quality & analyze video segments for each item ───────────────
            quality_scores: dict[str, dict[str, float]] = {}
            candidate_pool: list[dict[str, Any]] = []

            for item in all_media:
                mid = str(item.get("id", ""))
                if mid not in downloaded:
                    continue
                mtype = (item.get("media_type") or "image").lower()
                if mtype == "image":
                    q = mi.score_image_quality(downloaded[mid])
                    quality_scores[mid] = q
                    candidate_pool.append({
                        **item,
                        "quality_score": q.get("overall_score", 0.5),
                        "selection_score": q.get("overall_score", 0.5),
                    })
                else:
                    # Video: analyze shot/scene segments and extract visual understanding
                    v_segs = mi.analyze_video_segments(item, downloaded[mid], tmp_dir)
                    for seg in v_segs:
                        seg_id = seg["segment_id"]
                        quality_scores[seg_id] = {
                            "blur_score": seg["quality_score"],
                            "brightness_score": 0.75,
                            "resolution_score": 0.9,
                            "overall_score": seg["story_value"],
                        }
                        candidate_pool.append({
                            **item,
                            "id": seg["media_id"], # Keep original media_id for storage/DB lookup
                            "segment_id": seg_id,
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "duration": seg["duration"],
                            "scenes": seg["scenes"],
                            "objects": seg["objects"],
                            "people": seg["people"],
                            "description": seg["description"],
                            "quality_score": seg["quality_score"],
                            "story_value": seg["story_value"],
                            "selection_score": seg["story_value"],
                            "selection_reason": seg["selection_reason"],
                        })

            # ── 7. Intelligence pipeline: select best candidates ──────────────────────
            selected_media = mi.select_story_candidates(
                media_items=candidate_pool,
                quality_scores=quality_scores,
                cluster_map=cluster_map,
                embeddings=embeddings_map,
                target_count=15,
                min_quality_threshold=0.18,
            )

            if not selected_media:
                logger.warning("Intelligence pipeline selected 0 items — falling back to candidate pool.")
                selected_media = candidate_pool[:10]

            logger.info(
                f"Video {job_id}: {len(all_media)} raw media → "
                f"{len(candidate_pool)} evaluated candidates/segments → "
                f"{len(selected_media)} selected for story"
            )

            # ── 8. Target dimensions ──────────────────────────────────────────────────
            is_high_res = (quality_profile == "high_quality")
            TARGET_SIZE = TargetAspectRatio.get_dimensions(dimension or "16:9", high_res=is_high_res)

            # ── 9. Generate AI story plan ─────────────────────────────────────────────
            story_plan = _generate_ai_story_plan(
                title=memory_title,
                date_str=memory_date,
                location=None,
                selected_media=selected_media,
                target_duration=target_duration,
            )
            scenes: list[dict[str, Any]] = story_plan.get("scenes") or []

            # Build media_id → local path lookup
            path_by_id: dict[str, str] = {
                str(item.get("id", "")): downloaded[str(item.get("id", ""))]
                for item in selected_media
                if str(item.get("id", "")) in downloaded
            }
            media_by_id: dict[str, dict[str, Any]] = {
                str(item.get("id", "")): item for item in selected_media
            }

            # ── 10. Render: story plan drives every clip ──────────────────────────────
            clips: list[Any] = []

            # Opening title card — background from first selected image, if any
            first_img_path: str | None = None
            for item in selected_media:
                if (item.get("media_type") or "image").lower() == "image":
                    mid = str(item.get("id", ""))
                    first_img_path = downloaded.get(mid)
                    break

            title_duration = 2.5
            title_clip = _create_ai_title_card_clip(
                title=str(story_plan.get("title") or memory_title),
                subtitle=memory_date,
                duration=title_duration,
                target_size=TARGET_SIZE,
                bg_img_path=first_img_path,
                emotion=mood or "nostalgic",
            )
            clips.append(title_clip)

            # ── 10b. Synthesize Emotion-Aware TTS Narration & Render Scenes ─────────
            tts_engine = EmotionTTSEngine()
            speech_segments: list[tuple[float, Any]] = []
            current_timeline = title_duration

            for scene in scenes:
                scene_media_id = str(scene.get("media_id", ""))
                scene_path = path_by_id.get(scene_media_id)
                if not scene_path or not os.path.exists(scene_path):
                    logger.warning(f"Scene {scene.get('scene_id')} references missing media {scene_media_id} — skipping")
                    continue

                dur       = float(scene.get("duration_seconds") or 4.0)
                motion    = str(scene.get("motion") or "zoom_in")
                purpose   = str(scene.get("purpose") or "rising_action")
                item_data = media_by_id.get(scene_media_id, {})
                mtype     = (item_data.get("media_type") or "image").lower()
                narration = (scene.get("narration") or "").strip()

                # Synthesize emotion-aware speech voiceover for this scene
                if narration:
                    try:
                        speech_arr, speech_dur, _ = tts_engine.synthesize_sync(
                            text=narration,
                            emotion=mood or "calm",
                        )
                        if speech_dur > 0:
                            # Adjust scene duration so spoken narration finishes naturally
                            dur = max(dur, speech_dur + 0.6)
                            speech_segments.append((current_timeline + 0.2, speech_arr))
                    except Exception as tts_err:
                        logger.warning(f"Scene {scene.get('scene_id')} TTS synthesis failed: {tts_err}")

                try:
                    if mtype == "image":
                        # Check for external AI video generation (optional)
                        ai_vid_path: str | None = None
                        if settings.AI_VIDEO_API_KEY or settings.REPLICATE_API_KEY:
                            ai_vid_path = _try_generate_external_ai_video(
                                f"cinematic memory of {memory_title}", scene_path
                            )

                        if ai_vid_path and os.path.exists(ai_vid_path):
                            scene_clip = (
                                cast(Any, VideoFileClip(ai_vid_path))
                                .resized(new_size=TARGET_SIZE)
                                .with_audio(None)
                            )
                        else:
                            scene_clip = _create_ai_cinematic_scene_clip(
                                img_path=scene_path,
                                duration=dur,
                                target_size=TARGET_SIZE,
                                motion_type=motion,
                                caption=narration,
                            )
                        clips.append(scene_clip)
                        current_timeline += dur

                    else:  # video
                        v_clip = VideoFileClip(scene_path)
                        vw, vh = v_clip.size
                        tw, th = TARGET_SIZE
                        scale = min(tw / vw, th / vh)
                        nw, nh = int(vw * scale), int(vh * scale)
                        if abs((vw / vh) - (tw / th)) > 0.02:
                            # Adaptive composition: letterbox/pillarbox with dark cinematic background
                            v_resized = v_clip.resized(new_size=(nw, nh))
                            bg_clip = ColorClip(size=TARGET_SIZE, color=(14, 10, 24), duration=v_clip.duration)
                            v_clip = CompositeVideoClip([bg_clip, cast(Any, v_resized).with_position(("center", "center"))])
                        else:
                            v_clip = v_clip.resized(new_size=TARGET_SIZE)
                        
                        start_t = float(scene.get("start_time", 0.0) or 0.0)
                        end_t = float(scene.get("end_time", 0.0) or 0.0)
                        total_v_dur = float(v_clip.duration or 0.0)

                        if end_t > start_t and total_v_dur > 0:
                            actual_end = min(total_v_dur, end_t)
                            actual_start = min(start_t, actual_end - 0.5)
                            v_clip = v_clip.subclipped(max(0.0, actual_start), actual_end)
                        elif total_v_dur > (dur + 2.0):
                            v_clip = v_clip.subclipped(0, min(total_v_dur, dur + 2.0))

                        clips.append(v_clip)
                        current_timeline += float(v_clip.duration or dur)

                except Exception as clip_err:
                    logger.error(f"Error building clip for scene {scene.get('scene_id')} ({scene_media_id}): {clip_err}")

            if not clips:
                update_job_status("failed", "All clip rendering failed.")
                return

            final_clip = concatenate_videoclips(clips, method="compose")
            target_dur = float(target_duration or 20.0)
            rendered_dur = float(final_clip.duration or current_timeline)
            if rendered_dur > (target_dur + 0.3):
                final_clip = final_clip.subclipped(0, target_dur)
                reel_total_dur = target_dur
            elif rendered_dur < (target_dur - 0.5):
                deficit = target_dur - rendered_dur
                outro_clip = _create_ai_outro_card_clip(duration=deficit, target_size=TARGET_SIZE)
                final_clip = concatenate_videoclips([final_clip, outro_clip], method="compose")
                reel_total_dur = target_dur
            else:
                reel_total_dur = rendered_dur

            sr = 44100
            total_samples = max(1, int(reel_total_dur * sr))

            # ── 11. Audio: Ambient BGM + Emotion-Aware TTS Narration with Audio Ducking ──
            try:
                if music_style in ("none", "mute", "off"):
                    # Only speech track (or silent background)
                    if speech_segments:
                        speech_track = np.zeros((total_samples, 2), dtype=np.float32)
                        for seg_start_t, s_arr in speech_segments:
                            start_idx = int(seg_start_t * sr)
                            end_idx = min(total_samples, start_idx + len(s_arr))
                            slice_len = end_idx - start_idx
                            if slice_len > 0:
                                speech_track[start_idx:end_idx] += s_arr[:slice_len]
                        ambient_music = AudioArrayClip(speech_track, fps=sr)
                        final_clip = final_clip.with_audio(ambient_music)
                    else:
                        final_clip = final_clip.with_audio(None)
                else:
                    from ai_engine.video_generation.audio_synth import synthesize_ambient_soundtrack
                    effective_mood = music_style if music_style in (
                        "calm", "nostalgic", "joyful", "reflective", "uplifting", "acoustic", "warm", "emotional", "lofi"
                    ) else (mood or "calm")
                    raw_bgm = synthesize_ambient_soundtrack(
                        mood=effective_mood,
                        duration_seconds=reel_total_dur,
                        sample_rate=sr,
                    )
                    bgm_stereo = np.column_stack([raw_bgm, raw_bgm])

                    # Build speech ducking envelope
                    speech_mask = np.zeros(total_samples, dtype=np.float32)
                    speech_track = np.zeros((total_samples, 2), dtype=np.float32)

                    for seg_start_t, s_arr in speech_segments:
                        start_idx = int(seg_start_t * sr)
                        end_idx = min(total_samples, start_idx + len(s_arr))
                        slice_len = end_idx - start_idx
                        if slice_len > 0:
                            speech_mask[start_idx:end_idx] = 1.0
                            speech_track[start_idx:end_idx] += s_arr[:slice_len]

                    # Smooth ducking transitions (0.25s cosine fade)
                    fade_samples = max(1, int(0.25 * sr))
                    kernel = np.hanning(fade_samples * 2)
                    kernel /= kernel.sum()
                    speech_mask_smooth = np.convolve(speech_mask, kernel, mode="same")
                    speech_mask_smooth = np.clip(speech_mask_smooth, 0.0, 1.0)

                    # Duck BGM: 55% normal volume, ducks down to 18% during voice narration
                    bgm_gain = 0.55 * (1.0 - 0.68 * speech_mask_smooth)
                    ducked_bgm = bgm_stereo * bgm_gain[:, None]

                    # Composite final mixed soundtrack
                    mixed_audio = ducked_bgm + speech_track
                    mixed_audio = np.clip(mixed_audio, -0.98, 0.98)

                    ambient_music = AudioArrayClip(mixed_audio, fps=sr)
                    final_clip = final_clip.with_audio(ambient_music)
            except Exception as audio_err:
                logger.warning(f"Soundtrack composition failed: {audio_err}, falling back to ambient generator")
                ambient_music = _create_ambient_audio(reel_total_dur)
                final_clip = final_clip.with_audio(ambient_music)

            # ── 12. Write output MP4 to local staging area ─────────────────────────────
            job_state_manager.transition_stage(
                job_id, "encoding_video", task_description="Encoding high-efficiency H.264 video reel..."
            )
            output_filename = f"memory_video_{uuid.uuid4().hex}.mp4"
            staged_path = storage_service.get_staging_path(job_id, output_filename)
            job_state_manager.set_staged_file(job_id, staged_path)
            cpu_threads = max(2, (os.cpu_count() or 4) - 1)

            # Quality profile configuration
            if quality_profile == "high_quality":
                render_preset = "medium"
                render_fps = 30
                render_bitrate = "6000k"
            elif quality_profile == "fast":
                render_preset = "ultrafast"
                render_fps = 24
                render_bitrate = "2000k"
            else:  # balanced
                render_preset = "fast"
                render_fps = 24
                render_bitrate = "3500k"

            final_clip.write_videofile(
                staged_path,
                codec="libx264",
                audio_codec="aac",
                preset=render_preset,
                logger=None,
                fps=render_fps,
                bitrate=render_bitrate,
                threads=cpu_threads,
                ffmpeg_params=["-movflags", "+faststart"],
            )
            reel_duration = final_clip.duration or 0
            final_clip.close()
            for c in clips:
                if hasattr(c, "close"):
                    try:
                        c.close()
                    except Exception:
                        pass

            # ── 12a. Persist to ratio-partitioned local videos directory ──────────────
            persistent_video_path = storage_service.save_to_ratio_storage(
                src_path=staged_path,
                ratio=dimension or "16:9",
                filename=output_filename,
            )
            logger.info(f"Job {job_id}: Video permanently saved in ratio storage: {persistent_video_path}")

            # ── 12b. Output Quality Validation Check ─────────────────────────────────
            try:
                val_result = QualityValidator.validate_video(
                    filepath=staged_path,
                    expected_aspect_ratio=dimension or "16:9",
                    min_duration_seconds=3.0,
                )
                logger.info(f"Quality validation for job {job_id}: is_valid={val_result.is_valid}, diagnostics={val_result.diagnostics}")
                if val_result.warnings:
                    logger.warning(f"Quality validation warnings for job {job_id}: {val_result.warnings}")
            except Exception as qv_err:
                logger.warning(f"Quality validation diagnostic check skipped: {qv_err}")

            # ── 13. Resilient Streaming Upload with Backoff & Jitter ───────────────────
            job_state_manager.transition_stage(
                job_id, "uploading", task_description="Streaming generated video reel to cloud storage..."
            )
            upload_ok, public_url, reel_storage_path, upload_err_code = storage_service.upload_video_with_retry(
                job_id=job_id,
                user_id=user_id,
                staged_filepath=staged_path,
                output_filename=output_filename,
            )
            if not upload_ok or not public_url:
                job_state_manager.fail_job(
                    job_id,
                    error_code=upload_err_code or "UPLOAD_FAILED",
                    error_message="Failed to upload video after retries. Video is safely preserved and ready for retry.",
                )
                update_job_status("failed", "Failed to upload to storage after retries.")
                return

            # ── 14. Finalizing Database Record ─────────────────────────────────────────
            job_state_manager.transition_stage(
                job_id, "finalizing", task_description="Finalizing database record and video stream URL..."
            )
            first_media_thumb: str | None = None
            if selected_media:
                first_media_thumb = (
                    selected_media[0].get("thumbnail_url")
                    or selected_media[0].get("url")
                )

            media_data = {
                "memory_id":    memory_id,
                "vault_id":     vault_id,
                "owner_id":     user_id,
                "filename":     output_filename,
                "storage_path": reel_storage_path,
                "url":          public_url,
                "thumbnail_url":first_media_thumb,
                "media_type":   "video",
                "file_size":    os.path.getsize(staged_path),
                "mime_type":    "video/mp4",
                "duration":     int(reel_duration),
                "created_at":   datetime.now(timezone.utc).isoformat(),
            }

            insert_res = supabase.table("media").insert(media_data).execute()
            if not insert_res.data:
                job_state_manager.fail_job(job_id, "DB_INSERT_FAILED", "Failed to save generated video metadata.")
                update_job_status("failed", "Failed to save generated video metadata.")
                return

            ins_records = cast(list[dict[str, Any]], insert_res.data)
            new_media = ins_records[0]
            media_id_str = str(new_media["id"])
            job_state_manager.complete_job(job_id, result_media_id=media_id_str, result_url=public_url)
            update_job_status("completed", result_media_id=media_id_str)
            storage_service.cleanup_staged_file(staged_path)

            logger.info(
                f"Video job {job_id} complete: {output_filename} "
                f"({len(selected_media)} scenes, {reel_duration:.1f}s, "
                f"{media_data['file_size'] // 1024}KB)"
            )

        except Exception as e:
            logger.exception(f"Video job {job_id} failed: {e}")
            job_state_manager.fail_job(job_id, "PROCESSING_ERROR", str(e))
            update_job_status("failed", str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
