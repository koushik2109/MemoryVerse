"""
Media Intelligence Pipeline for MemoryVerse Video Generation.

Provides deterministic, dependency-free analysis of media items BEFORE
they are passed to the story planner or renderer.

Capabilities:
  - Image technical quality scoring  (blur, brightness, resolution)
  - Video quality estimation          (metadata-based proxy)
  - Near-duplicate detection          (CLIP cosine similarity, existing embeddings)
  - Story candidate selection         (quality + diversity ranking)

All functions use only numpy, scipy (already installed), and Pillow.
No new packages required.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Image Quality Scoring ─────────────────────────────────────────────────────

def _laplacian_variance(arr: Any) -> float:
    """
    Estimate image sharpness via Laplacian variance.
    Uses scipy.ndimage.laplace if available, falls back to numpy 2nd-order gradient.
    Higher value = sharper image.
    """
    try:
        from scipy.ndimage import laplace
        lap = laplace(arr)
        return float(np.var(lap))
    except Exception:
        gy2 = np.diff(arr, n=2, axis=0)
        gx2 = np.diff(arr, n=2, axis=1)
        return float(np.var(gy2) + np.var(gx2))


def score_image_quality(img_path: str) -> Dict[str, float]:
    """
    Compute technical quality metrics for a local image file.

    Returns dict with keys:
      blur_score        0.0–1.0   (1.0 = sharpest)
      brightness_score  0.0–1.0   (1.0 = well exposed)
      resolution_score  0.0–1.0   (1.0 = high resolution)
      overall_score     0.0–1.0   (weighted combination)

    On any failure, returns conservative defaults of 0.5 so the item
    is not unfairly penalised.
    """
    try:
        with Image.open(img_path) as pil_img:
            gray = pil_img.convert("L")
            arr = np.array(gray, dtype=np.float32)
            w, h = gray.size

        # 1. Blur (Laplacian variance)
        lap_var = _laplacian_variance(arr)
        # Calibration: ~30 = blurry phone burst, ~300 = average sharp, ~3000+ = crisp
        blur_score = float(min(1.0, lap_var / 400.0))

        # 2. Brightness (penalise very dark or overexposed images)
        mean_lum = float(np.mean(arr))  # 0–255
        if mean_lum < 12.0:
            brightness_score = 0.05   # essentially black
        elif mean_lum > 248.0:
            brightness_score = 0.15   # completely blown out
        else:
            # Peak around 110–155.  Gentle penalty toward extremes.
            dist = abs(mean_lum - 132.0)
            brightness_score = max(0.10, 1.0 - dist / 145.0)

        # 3. Resolution
        pixels = w * h
        if pixels < 40_000:          # < ~200×200
            resolution_score = 0.05
        elif pixels < 300_000:       # < ~550×550
            resolution_score = 0.40
        elif pixels < 1_000_000:     # < 1 MP
            resolution_score = 0.70
        elif pixels < 4_000_000:     # < 4 MP
            resolution_score = 0.90
        else:
            resolution_score = 1.00

        overall = (
            0.55 * blur_score
            + 0.30 * brightness_score
            + 0.15 * resolution_score
        )

        return {
            "blur_score":       round(blur_score, 3),
            "brightness_score": round(brightness_score, 3),
            "resolution_score": round(resolution_score, 3),
            "overall_score":    round(min(1.0, overall), 3),
        }

    except Exception as e:
        logger.warning(f"Image quality scoring failed for {img_path}: {e}")
        return {
            "blur_score":       0.5,
            "brightness_score": 0.5,
            "resolution_score": 0.5,
            "overall_score":    0.5,
        }


def score_video_quality(media_item: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate video quality from DB metadata.

    File size used as a bitrate/quality proxy.
    Videos get a flat +0.10 bonus over the raw score because they represent
    intentional, non-duplicable recordings.
    """
    file_size = media_item.get("file_size") or 0
    duration = media_item.get("duration") or 0

    size_mb = file_size / (1024 * 1024)
    if size_mb < 0.5:
        size_score = 0.20    # Tiny — possibly corrupt or just a few frames
    elif size_mb < 2:
        size_score = 0.50
    elif size_mb < 20:
        size_score = 0.80
    else:
        size_score = 1.00

    duration_score = 0.80    # default when duration unknown
    if duration and duration < 2:
        duration_score = 0.30   # Too short to be useful

    overall = 0.60 * size_score + 0.40 * duration_score
    return {
        "blur_score":       0.75,
        "brightness_score": 0.75,
        "resolution_score": size_score,
        "overall_score":    round(min(1.0, overall + 0.10), 3),
    }


# ── Video Scene Detection & Understanding ─────────────────────────────────────

def detect_video_scenes(
    video_path: str,
    min_duration: float = 2.5,
    max_duration: float = 14.0,
) -> List[Tuple[float, float]]:
    """
    Detect candidate shot/scene boundaries within a video file using FFmpeg scene detection.

    - Uses bundled FFmpeg select filter with scene score threshold (0.28).
    - Merges segments shorter than min_duration.
    - Subdivides segments longer than max_duration into balanced chunks.
    - Returns list of (start_time, end_time) in seconds.
    """
    import re
    import subprocess
    import imageio_ffmpeg

    # 1. Obtain video total duration
    total_duration: float = 0.0
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(video_path) as vf:
            total_duration = float(vf.duration or 0.0)
    except Exception:
        pass

    if total_duration <= 0.0:
        total_duration = 10.0  # Fallback assumption

    # If video is already very short, return single segment
    if total_duration <= max_duration:
        return [(0.0, round(total_duration, 2))]

    # 2. Run FFmpeg scene change filter
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-i", video_path,
        "-filter:v", "select=gt(scene\\,0.28),showinfo",
        "-f", "null",
        "-"
    ]

    cut_points: List[float] = [0.0]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        for line in proc.stderr.split("\n"):
            if "showinfo" in line and "pts_time:" in line:
                m = re.search(r"pts_time:([0-9.]+)", line)
                if m:
                    t = float(m.group(1))
                    if 0.5 < t < (total_duration - 0.5):
                        cut_points.append(t)
    except Exception as e:
        logger.warning(f"FFmpeg scene detection error for {video_path}: {e}")

    cut_points.append(total_duration)
    cut_points = sorted(list(set(cut_points)))

    # 3. Form raw segments between cuts
    raw_segments: List[Tuple[float, float]] = []
    for i in range(len(cut_points) - 1):
        s_start = cut_points[i]
        s_end = cut_points[i + 1]
        if s_end > s_start:
            raw_segments.append((s_start, s_end))

    # 4. Merge short segments (< min_duration)
    merged_segments: List[Tuple[float, float]] = []
    for seg in raw_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev_s, prev_e = merged_segments[-1]
            seg_s, seg_e = seg
            if (seg_e - seg_s) < min_duration:
                # Merge into previous segment
                merged_segments[-1] = (prev_s, seg_e)
            elif (prev_e - prev_s) < min_duration:
                # Extend previous segment with current
                merged_segments[-1] = (prev_s, seg_e)
            else:
                merged_segments.append(seg)

    # 5. Split overly long segments (> max_duration)
    final_segments: List[Tuple[float, float]] = []
    for seg_s, seg_e in merged_segments:
        dur = seg_e - seg_s
        if dur > max_duration:
            # Subdivide into 5-8s sub-segments
            n_splits = int(np.ceil(dur / 7.0))
            step = dur / n_splits
            for k in range(n_splits):
                sub_s = seg_s + k * step
                sub_e = min(seg_e, seg_s + (k + 1) * step)
                if (sub_e - sub_s) >= min_duration:
                    final_segments.append((round(sub_s, 2), round(sub_e, 2)))
        else:
            final_segments.append((round(seg_s, 2), round(seg_e, 2)))

    if not final_segments:
        final_segments = [(0.0, round(total_duration, 2))]

    return final_segments


def extract_representative_frame(
    video_path: str,
    timestamp: float,
    output_path: str
) -> bool:
    """
    Extract a single high-quality representative frame from a video at a specified timestamp.
    """
    import subprocess
    import imageio_ffmpeg
    import os

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", str(round(timestamp, 3)),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        return res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.warning(f"Frame extraction failed at {timestamp}s: {e}")
        return False


def analyze_video_segments(
    media_item: Dict[str, Any],
    video_path: str,
    tmp_dir: str
) -> List[Dict[str, Any]]:
    """
    Analyze a video file into discrete candidate segments with visual understanding.

    Pipeline:
      1. Detect scene/shot cuts via FFmpeg
      2. Extract representative frame for each segment
      3. Score image quality on the frame (sharpness, exposure)
      4. Run CLIP zero-shot feature extraction for semantic understanding
      5. Compute importance and story value scores
      6. Return list of timestamped candidate segments
    """
    import os
    from app.services.ai_extractor import AIExtractor

    media_id = str(media_item.get("id", "vid"))
    segments = detect_video_scenes(video_path)
    analyzed: List[Dict[str, Any]] = []

    for idx, (start_t, end_t) in enumerate(segments):
        seg_dur = end_t - start_t
        if seg_dur < 1.5:
            continue

        # Extract representative frame at 45% into segment
        t_sample = start_t + seg_dur * 0.45
        frame_path = os.path.join(tmp_dir, f"vframe_{media_id[:8]}_seg{idx}.jpg")
        extracted = extract_representative_frame(video_path, t_sample, frame_path)

        q_score = 0.6
        scenes_detected: List[str] = []
        objects_detected: List[str] = []
        people_detected: str = "unknown"
        clip_emb: List[float] = []

        if extracted and os.path.exists(frame_path):
            try:
                # 1. Technical quality
                q_dict = score_image_quality(frame_path)
                q_score = q_dict.get("overall_score", 0.6)

                # 2. Semantic features via CLIP
                with open(frame_path, "rb") as fh:
                    f_bytes = fh.read()
                clip_emb, f_tags = AIExtractor.extract_features(f_bytes, "image/jpeg")
                scenes_detected = f_tags.get("scenes", [])
                objects_detected = f_tags.get("objects", [])
                people_detected = f_tags.get("people_count", "unknown")
            except Exception as fe_err:
                logger.warning(f"Frame analysis error for seg {idx}: {fe_err}")

        # Importance & Story Value calculation
        # Base importance from visual quality & duration
        importance = 0.40 * q_score + 0.30 * min(1.0, seg_dur / 6.0)

        # Content bonuses
        if people_detected in ("group of people", "one person"):
            importance += 0.15
        if scenes_detected:
            # Scenic/celebratory bonus
            if any(s in ("sunset", "party", "beach", "mountains", "nature") for s in scenes_detected):
                importance += 0.15

        # Story value balances importance with uniqueness
        story_val = min(1.0, importance + 0.05)

        # Natural description
        desc_parts: List[str] = []
        if scenes_detected:
            desc_parts.append(f"{scenes_detected[0]} scene")
        if people_detected and people_detected not in ("unknown", "no people"):
            desc_parts.append(f"with {people_detected}")
        if objects_detected:
            desc_parts.append(f"featuring {objects_detected[0]}")
        desc = " ".join(desc_parts).capitalize() if desc_parts else "Action footage moment"

        analyzed.append({
            "id": media_id,
            "segment_id": f"{media_id}_seg{idx}",
            "media_id": media_id,
            "filename": media_item.get("filename", f"video_{media_id[:6]}.mp4"),
            "media_type": "video",
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "duration": round(seg_dur, 2),
            "quality_score": round(q_score, 3),
            "importance_score": round(min(1.0, importance), 3),
            "story_value": round(story_val, 3),
            "selection_score": round(story_val, 3),
            "selection_reason": f"Video segment ({start_t:.1f}s-{end_t:.1f}s): {desc}",
            "description": desc,
            "scenes": scenes_detected,
            "objects": objects_detected,
            "people": people_detected,
            "embedding": clip_emb,
            "created_at": media_item.get("created_at"),
            "taken_at": media_item.get("taken_at"),
        })

    # Whisper Audio Transcription & Speech Understanding
    try:
        from app.services.transcription_service import transcribe_video, associate_transcripts_to_segments
        transcripts = transcribe_video(video_path, tmp_dir=tmp_dir)
        if transcripts:
            logger.info(f"Transcribed {len(transcripts)} speech segments for {media_id}")
            analyzed = associate_transcripts_to_segments(analyzed, transcripts)
    except Exception as trans_err:
        logger.warning(f"Audio transcription skipped for {media_id}: {trans_err}")

    return analyzed



# ── Near-Duplicate Detection ──────────────────────────────────────────────────

def detect_near_duplicates(
    media_ids: List[str],
    embeddings: Dict[str, List[float]],
    threshold: float = 0.92,
) -> Dict[str, int]:
    """
    Cluster visually near-identical items using cosine similarity on CLIP embeddings
    already stored in the media_embeddings table.

    Items sharing the same cluster_id are near-duplicates; the selection step
    will keep only the best-quality representative from each group.

    Args:
        media_ids:  ordered list of candidate media IDs
        embeddings: {media_id: CLIP vector}  (from media_embeddings table)
        threshold:  cosine similarity threshold — 0.92 catches burst photos and
                    tight crops of the same scene; lower values are more aggressive

    Returns:
        {media_id: cluster_id}   — unique int per cluster; singletons get own cluster
    """
    has_emb: List[str] = [
        mid for mid in media_ids
        if mid in embeddings
        and embeddings[mid]
        and any(v != 0.0 for v in embeddings[mid])
    ]
    no_emb: List[str] = [mid for mid in media_ids if mid not in has_emb]

    result: Dict[str, int] = {}
    counter = 0

    if len(has_emb) >= 2:
        mat = np.array([embeddings[mid] for mid in has_emb], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        sim = np.dot(normed, normed.T)  # (N, N)

        # Union-Find with path compression
        parent = list(range(len(has_emb)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(has_emb)):
            for j in range(i + 1, len(has_emb)):
                if float(sim[i, j]) >= threshold:
                    union(i, j)

        root_to_cluster: Dict[int, int] = {}
        for i, mid in enumerate(has_emb):
            root = find(i)
            if root not in root_to_cluster:
                root_to_cluster[root] = counter
                counter += 1
            result[mid] = root_to_cluster[root]
    else:
        for mid in has_emb:
            result[mid] = counter
            counter += 1

    # Videos and unprocessed items always get unique clusters
    for mid in no_emb:
        result[mid] = counter
        counter += 1

    return result


# ── Story Candidate Selection ─────────────────────────────────────────────────

def select_story_candidates(
    media_items: List[Dict[str, Any]],
    quality_scores: Dict[str, Dict[str, float]],
    cluster_map: Dict[str, int],
    embeddings: Dict[str, List[float]],
    target_count: int = 12,
    min_quality_threshold: float = 0.18,
    max_segments_per_video: int = 2,
) -> List[Dict[str, Any]]:
    """
    Select the best and most diverse media items & video highlights for the story journal.

    Editorial Rules:
      1. Per-cluster deduplication: one representative per burst/near-duplicate visual cluster.
      2. Quality filter: discard low-exposure, blurred, or low-story-value items.
      3. Video Segment Budgeting: A single video can contribute at most 1-2 top distinct highlights,
         skipping boring setup or repetitive stretches.
      4. Story Relevance Scoring: Combines visual quality (50%), story value (30%), uniqueness (20%).
      5. Chronological Re-ordering: Selected highlights are arranged in natural timeline order.
    """
    if not media_items:
        return []

    def _get_item_score(it: Dict[str, Any]) -> float:
        k = str(it.get("segment_id") or it.get("id", ""))
        entry = quality_scores.get(k, {})
        val = entry.get("overall_score")
        if val is None:
            val = it.get("story_value")
        try:
            return float(val) if val is not None else 0.5
        except (ValueError, TypeError):
            return 0.5

    # Step 1: Best representative per cluster
    cluster_best: Dict[int, Tuple[float, str]] = {}
    for item in media_items:
        item_key = str(item.get("segment_id") or item.get("id", ""))
        q = _get_item_score(item)
        cid = cluster_map.get(item_key, -1)
        if cid not in cluster_best or cluster_best[cid][0] < q:
            cluster_best[cid] = (q, item_key)

    winners: set = {winner_key for (_, winner_key) in cluster_best.values()}

    # Step 2: Quality & Relevance filtering
    candidates: List[Dict[str, Any]] = []
    video_segment_counts: Dict[str, int] = {}  # media_id -> count of selected segments

    # Pre-calculate peak scores per video to prune low-value setup segments
    video_peak_scores: Dict[str, float] = {}
    for item in media_items:
        if (item.get("media_type") or "image").lower() == "video":
            mid = str(item.get("media_id") or item.get("id", ""))
            score = _get_item_score(item)
            if mid not in video_peak_scores or score > video_peak_scores[mid]:
                video_peak_scores[mid] = score

    # Sort all input media by quality/story_value first so top moments get priority
    sorted_pool = sorted(
        media_items,
        key=_get_item_score,
        reverse=True,
    )

    for item in sorted_pool:
        item_key = str(item.get("segment_id") or item.get("id", ""))
        if item_key not in winners:
            continue

        # Sensitive Content & Document Exclusion (Privacy & Safety Policy)
        desc = (item.get("description") or "").lower()
        fname = (item.get("file_name") or item.get("filename") or "").lower()
        tags = [str(t).lower() for t in item.get("tags") or []]
        is_sensitive = item.get("is_sensitive", False) or any(
            k in fname or k in desc or k in tags
            for k in [
                "screenshot", "screen_shot", "invoice", "receipt",
                "passport", "id_card", "credit_card", "tax_return",
                "document_scan", "sensitive_doc", "nsfw"
            ]
        )
        if is_sensitive and not item.get("explicitly_included", False):
            logger.debug(f"Excluding sensitive/document media {item_key} from storytelling candidate pool")
            continue

        q = _get_item_score(item)
        mtype = (item.get("media_type") or "image").lower()

        if q < min_quality_threshold:
            logger.debug(f"Skipping low-quality candidate {item_key} (score={q:.2f})")
            continue

        # For video segments: enforce editorial selection
        if mtype == "video":
            parent_id = str(item.get("media_id") or item.get("id", ""))
            curr_count = video_segment_counts.get(parent_id, 0)
            
            # 1. Budget constraint: max segments per video
            if curr_count >= max_segments_per_video:
                logger.debug(f"Skipping video segment {item_key}: budget reached for video {parent_id}")
                continue

            # 2. Relative quality constraint: skip boring setup if peak celebration exists in same video
            peak = video_peak_scores.get(parent_id, q)
            if peak - q > 0.11:
                logger.debug(f"Skipping video segment {item_key}: score {q:.2f} significantly below video peak {peak:.2f}")
                continue

            video_segment_counts[parent_id] = curr_count + 1

        # Uniqueness score from cluster size
        cid = cluster_map.get(item_key, -1)
        cluster_size = sum(1 for c in cluster_map.values() if c == cid)
        uniqueness = max(0.10, 1.0 - (cluster_size - 1) * 0.25)

        story_v = item.get("story_value")
        s_val = float(story_v) if story_v is not None else q
        selection_score = 0.50 * q + 0.30 * s_val + 0.20 * uniqueness

        # Format human-readable reason
        ai_tags = (item.get("metadata") or {}).get("ai_tags", {})
        scenes = item.get("scenes") or ai_tags.get("scenes", [])
        objects = item.get("objects") or ai_tags.get("objects", [])
        people = item.get("people") or ai_tags.get("people_count", "")
        parts: List[str] = []
        if q >= 0.75:
            parts.append("highlight moment")
        if scenes:
            parts.append(scenes[0])
        if objects:
            parts.append(objects[0])
        if people and people not in ("unknown", "no people"):
            parts.append(people)

        candidates.append({
            **item,
            "selection_score": round(selection_score, 3),
            "quality_score": round(q, 3),
            "selection_reason": item.get("selection_reason") or (", ".join(parts) or "general content"),
        })

    # Step 3: Top candidate selection up to target_count
    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    selected = candidates[:target_count]

    # Step 4: Re-sort chronologically (photos by timestamp, video segments by start_time)
    def _timeline_sort_key(x: Dict[str, Any]) -> str:
        date_part = str(x.get("taken_at") or x.get("created_at") or "")
        time_offset = f"{float(x.get('start_time', 0.0)):06.2f}"
        return f"{date_part}_{time_offset}"

    selected.sort(key=_timeline_sort_key)

    logger.info(
        "MediaIntelligence select_story_candidates: "
        f"{len(media_items)} input candidates → "
        f"{len(winners)} after dedup → "
        f"{len(candidates)} editorially qualified → "
        f"{len(selected)} selected for story"
    )
    return selected
