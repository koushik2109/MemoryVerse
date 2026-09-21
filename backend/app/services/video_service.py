import json
import os
import uuid
import logging
import httpx
import math
import numpy as np
from datetime import datetime, timezone
from typing import cast, Any
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from moviepy import VideoClip, VideoFileClip, AudioArrayClip, concatenate_videoclips

from app.core.db import get_supabase_client
from app.config.settings import settings
from app.services import media_intelligence as mi
from ai_engine.video_generation.tts_engine import EmotionTTSEngine

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

    for i, item in enumerate(editorial_candidates):
        mid   = str(item.get("id", ""))
        mtype = (item.get("media_type") or "image").lower()
        q     = item.get("quality_score", item.get("story_value", 0.5))

        start_t = float(item.get("start_time", 0.0) or 0.0)
        end_t   = float(item.get("end_time",   0.0) or 0.0)

        if i == 0:
            purpose = "opening"
            dur     = 5.0
        elif i == total - 1:
            purpose = "closing"
            dur     = 4.5
        elif q >= 0.72 or mtype == "video":
            purpose = "main_moment"
            dur     = 6.0
        else:
            purpose = "rising_action"
            dur     = 3.5

        if mtype == "video":
            if end_t > start_t:
                dur = min(8.0, max(3.0, round(end_t - start_t, 2)))
            else:
                dur = max(dur, 5.0)
                end_t = start_t + dur

        # Build evidence-based fallback narration
        description = item.get("description") or item.get("selection_reason") or ""
        speech_text = item.get("speech_text") or ""
        if speech_text:
            # Trim to first ~8 words to avoid overly long sentences
            words = speech_text.strip().split()
            short = " ".join(words[:8]) + ("..." if len(words) > 8 else "")
            narration = f"A memorable moment: \"{short}\""
        elif description and description not in ("Action footage moment",):
            # Capitalise and use as-is, but strip internal editorial reasons
            narration = description.split(":")[0].strip().capitalize()
            if len(narration) > 120:
                narration = narration[:117] + "..."
        else:
            narration = f"A moment from {title}."

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

    # 1. Fast ambient background
    small_w, small_h = max(1, tw // 4), max(1, th // 4)
    bg_small = orig_img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=6))
    bg_base = bg_small.resize((tw, th), Image.Resampling.BILINEAR)
    dark_overlay = Image.new("RGBA", (tw, th), (15, 10, 25, 140))
    bg_base = Image.alpha_composite(bg_base.convert("RGBA"), dark_overlay).convert("RGB")

    # 2. Foreground base
    max_fw, max_fh = int(tw * 0.86), int(th * 0.86)
    scale_fg = min(max_fw / ow, max_fh / oh)
    fg_w, fg_h = int(ow * scale_fg), int(oh * scale_fg)
    fg_base = orig_img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    border_img = Image.new("RGBA", (fg_w + 8, fg_h + 8), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(border_img)
    b_draw.rectangle([0, 0, fg_w + 7, fg_h + 7], outline=(255, 255, 255, 80), width=2)
    border_img.paste(fg_base, (4, 4))

    # Pre-render Start Keyframe (p = 0.0)
    z0 = 1.00 if "zoom_in" in motion_type else (1.10 if "zoom_out" in motion_type else 1.05)
    sx0 = -15 if "pan_right" in motion_type else (15 if "pan_left" in motion_type else 0)
    sy0 = -8 if "zoom_in" in motion_type else (8 if "zoom_out" in motion_type else 0)

    w0, h0 = int(border_img.width * z0), int(border_img.height * z0)
    fg0 = border_img.resize((w0, h0), Image.Resampling.BILINEAR) if z0 != 1.0 else border_img
    canvas0 = bg_base.copy()
    canvas0.paste(fg0, ((tw - w0) // 2 + sx0, (th - h0) // 2 + sy0), fg0)
    # Pre-render End Keyframe (p = 1.0)
    z1 = 1.10 if "zoom_in" in motion_type else (1.00 if "zoom_out" in motion_type else 1.05)
    sx1 = 15 if "pan_right" in motion_type else (-15 if "pan_left" in motion_type else 0)
    sy1 = 8 if "zoom_in" in motion_type else (-8 if "zoom_out" in motion_type else 0)

    w1, h1 = int(border_img.width * z1), int(border_img.height * z1)
    fg1 = border_img.resize((w1, h1), Image.Resampling.BILINEAR) if z1 != 1.0 else border_img
    canvas1 = bg_base.copy()
    canvas1.paste(fg1, ((tw - w1) // 2 + sx1, (th - h1) // 2 + sy1), fg1)

    if caption and caption.strip():
        try:
            from ai_engine.video_generation.subtitles import render_subtitle_frame
            canvas0 = render_subtitle_frame(canvas0, caption.strip(), font_size=28, bottom_margin=int(th * 0.12))
            canvas1 = render_subtitle_frame(canvas1, caption.strip(), font_size=28, bottom_margin=int(th * 0.12))
        except Exception as sub_err:
            logger.debug(f"Subtitle overlay skipped: {sub_err}")

    arr0 = np.array(canvas0, dtype=np.float32)
    arr1 = np.array(canvas1, dtype=np.float32)

    def make_frame(t: float) -> np.ndarray:
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


def _create_ai_title_card_clip(
    title: str,
    subtitle: str,
    duration: float = 2.5,
    target_size: tuple[int, int] = (1280, 720),
    bg_img_path: str | None = None
) -> VideoClip:
    """
    Ultra-Fast Opening Title Card Clip (pre-renders canvas ONCE).
    """
    tw, th = target_size
    if bg_img_path and os.path.exists(bg_img_path):
        with Image.open(bg_img_path) as orig:
            orig = orig.convert("RGB")
            ow, oh = orig.size
            scale = max(tw / ow, th / oh)
            bw, bh = int(ow * scale), int(oh * scale)
            bg = orig.resize((bw, bh), Image.Resampling.BILINEAR)
            l = (bw - tw) // 2
            t = (bh - th) // 2
            bg = bg.crop((l, t, l + tw, t + th)).filter(ImageFilter.GaussianBlur(radius=16))
            dark = Image.new("RGBA", (tw, th), (10, 5, 22, 210))
            bg = Image.alpha_composite(bg.convert("RGBA"), dark).convert("RGB")
    else:
        bg = Image.new("RGB", (tw, th), color=(14, 10, 24))

    # Draw title card text ONCE
    draw = ImageDraw.Draw(bg)

    try:
        font_tag = ImageFont.truetype("arial.ttf", size=22)
        font_title = ImageFont.truetype("arial.ttf", size=42)
        font_sub = ImageFont.truetype("arial.ttf", size=26)
    except Exception:
        try:
            font_tag = ImageFont.load_default(size=22)
            font_title = ImageFont.load_default(size=42)
            font_sub = ImageFont.load_default(size=26)
        except TypeError:
            font_tag = font_title = font_sub = ImageFont.load_default()

    tag_text = "✦  MEMORYVERSE AI REEL  ✦"
    draw.text((tw // 2, th // 2 - 80), tag_text, font=font_tag, fill=(215, 185, 255), anchor="mm")

    disp_title = title if len(title) <= 36 else title[:33] + "..."
    draw.text((tw // 2, th // 2 - 25), disp_title, font=font_title, fill=(255, 255, 255), anchor="mm")

    if subtitle:
        draw.text((tw // 2, th // 2 + 40), subtitle, font=font_sub, fill=(185, 175, 205), anchor="mm")

    arr_title = np.array(bg, dtype=np.float32)

    def make_frame(t: float) -> np.ndarray:
        f = arr_title.copy()
        fade_dur = 0.35
        if t < fade_dur:
            f *= (t / fade_dur)
        elif t > (duration - fade_dur):
            f *= max(0.0, (duration - t) / fade_dur)
        return f.astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    # Attach silent audio track so concatenate_videoclips with audio-bearing video clips succeeds seamlessly
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
            supabase.table("video_jobs").update(data).eq("id", job_id).execute()

        update_job_status("processing")

        tmp_dir = tempfile.mkdtemp(prefix="memoryverse_reel_")

        try:
            # 1. Fetch memory details for title & date
            mem_res = supabase.table("memories").select("*").eq("id", memory_id).execute()
            memory_title = "My Memory"
            memory_date = ""
            vault_id = None
            if mem_res.data:
                mem = mem_res.data[0]
                memory_title = mem.get("title") or "My Memory"
                raw_date = mem.get("event_date") or mem.get("created_at") or ""
                if raw_date:
                    try:
                        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        memory_date = dt.strftime("%B %d, %Y")
                    except Exception:
                        memory_date = raw_date[:10]
                vault_id = mem.get("vault_id")

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
                update_job_status("failed", "No source media items found in this memory.")
                return

            # If user selected specific pictures/media IDs, filter to those
            if selected_media_ids:
                selected_set = {str(mid) for mid in selected_media_ids}
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
                for row in (emb_res.data or []):
                    mid = str(row.get("media_id", ""))
                    emb = row.get("embedding") or []
                    if isinstance(emb, list):
                        embeddings_map[mid] = emb
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
            if dimension == "9:16":
                TARGET_SIZE = (720, 1280)
            else:
                TARGET_SIZE = (1280, 720)

            # ── 9. Generate AI story plan ─────────────────────────────────────────────
            story_plan = _generate_ai_story_plan(
                title=memory_title,
                date_str=memory_date,
                location=None,
                selected_media=selected_media,
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

            title_duration = 3.0
            title_clip = _create_ai_title_card_clip(
                title=story_plan.get("title") or memory_title,
                subtitle=memory_date,
                duration=title_duration,
                target_size=TARGET_SIZE,
                bg_img_path=first_img_path,
            )
            clips.append(title_clip)

            # ── 10b. Synthesize Emotion-Aware TTS Narration & Render Scenes ─────────
            tts_engine = EmotionTTSEngine()
            speech_segments: list[tuple[float, np.ndarray]] = []
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
                                VideoFileClip(ai_vid_path)
                                .resized(new_size=TARGET_SIZE)
                                .without_audio()
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
            reel_total_dur = float(final_clip.duration or current_timeline)
            sr = 44100
            total_samples = max(1, int(reel_total_dur * sr))

            # ── 11. Audio: Ambient BGM + Emotion-Aware TTS Narration with Audio Ducking ──
            try:
                from ai_engine.video_generation.audio_synth import synthesize_ambient_soundtrack
                raw_bgm = synthesize_ambient_soundtrack(
                    mood=mood or "calm",
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

            # ── 12. Write output MP4 ──────────────────────────────────────────────────
            output_filename = f"memory_video_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(tmp_dir, output_filename)
            cpu_threads = max(2, (os.cpu_count() or 4) - 1)

            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast",
                logger=None,
                fps=24,
                threads=cpu_threads,
            )
            reel_duration = final_clip.duration or 0
            final_clip.close()
            for c in clips:
                if hasattr(c, "close"):
                    try:
                        c.close()
                    except Exception:
                        pass

            # ── 13. Upload to Supabase Storage ────────────────────────────────────────
            reel_storage_path = f"{user_id}/reels/{output_filename}"
            with open(output_path, "rb") as f:
                output_bytes = f.read()

            supabase.storage.from_("memories").upload(
                file=output_bytes,
                path=reel_storage_path,
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )
            signed_res = supabase.storage.from_("memories").create_signed_url(reel_storage_path, 31536000)
            public_url = signed_res.get("signedURL") or signed_res.get("signed_url") or ""

            # Use thumbnail from first selected item if available
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
                "file_size":    len(output_bytes),
                "mime_type":    "video/mp4",
                "duration":     int(reel_duration),
                "created_at":   datetime.now(timezone.utc).isoformat(),
            }

            insert_res = supabase.table("media").insert(media_data).execute()
            if not insert_res.data:
                update_job_status("failed", "Failed to save generated video metadata.")
                return

            new_media = insert_res.data[0]
            update_job_status("completed", result_media_id=new_media["id"])
            logger.info(
                f"Video job {job_id} complete: {output_filename} "
                f"({len(selected_media)} scenes, {reel_duration:.1f}s, "
                f"{len(output_bytes) // 1024}KB)"
            )

        except Exception as e:
            logger.exception(f"Video job {job_id} failed: {e}")
            update_job_status("failed", str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
