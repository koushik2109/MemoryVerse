"""
Transcription Service — Whisper-based speech understanding for videos.

Pipeline:
  Video file
  ↓
  Audio stream verification (FFmpeg)
  ↓
  Audio extraction (16kHz mono PCM WAV)
  ↓
  Whisper transcription with VAD
  ↓
  Timestamped transcript segments
  ↓
  Association with video scene segments
"""

import os
import re
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global singleton to keep Whisper model loaded in memory across jobs
_whisper_model: Optional[Any] = None


def get_whisper_model(model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """
    Lazy-load and cache the faster-whisper model.
    Reuses model in memory to avoid per-video reloading overhead.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{model_name}' on {device} ({compute_type})...")
            _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            logger.info("faster-whisper model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize faster-whisper model: {e}")
            return None
    return _whisper_model


def has_audio_stream(video_path: str) -> bool:
    """
    Check if the video file contains an audio stream with audible content.
    Uses FFmpeg stderr stream inspection.
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-i", video_path,
        "-vn",
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # Look for Audio stream or max_volume
        if "Audio:" in proc.stderr or "max_volume:" in proc.stderr:
            return True
        return False
    except Exception as e:
        logger.warning(f"Audio stream check failed for {video_path}: {e}")
        return True  # Permissive fallback: attempt extraction


def extract_audio_from_video(video_path: str, output_wav_path: str) -> bool:
    """
    Extract audio track to 16kHz mono 16-bit PCM WAV (optimal for Whisper).
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        return proc.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100
    except Exception as e:
        logger.warning(f"Audio extraction failed for {video_path}: {e}")
        return False


def transcribe_video(
    video_path: str,
    tmp_dir: Optional[str] = None,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transcribe spoken audio from a video file into timestamped segments.

    Returns:
        List of dicts:
        [
            {
                "start_time": 12.4,
                "end_time": 16.8,
                "text": "Look at that sunset!",
                "confidence": 0.92
            }, ...
        ]
    """
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found for transcription: {video_path}")
        return []

    # 1. Quick check if audio exists
    if not has_audio_stream(video_path):
        logger.info(f"No audio stream found in {video_path}, skipping transcription.")
        return []

    # 2. Extract audio to temp WAV
    cleanup_tmp = False
    if not tmp_dir or not os.path.exists(tmp_dir):
        tmp_dir = tempfile.mkdtemp(prefix="mv_whisper_")
        cleanup_tmp = True

    wav_path = os.path.join(tmp_dir, f"audio_{os.path.basename(video_path)}.wav")
    extracted = extract_audio_from_video(video_path, wav_path)
    if not extracted:
        if cleanup_tmp:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return []

    results: List[Dict[str, Any]] = []
    try:
        model = get_whisper_model()
        if model is None:
            logger.warning("Whisper model unavailable, returning empty transcript.")
            return []

        # Run transcription with Voice Activity Detection (VAD) filter
        segments_gen, info = model.transcribe(
            wav_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=2,
        )

        for seg in segments_gen:
            clean_text = seg.text.strip()
            # Ignore hallucinated empty/repetitive strings
            if clean_text and len(clean_text) > 1:
                results.append({
                    "start_time": round(float(seg.start), 2),
                    "end_time": round(float(seg.end), 2),
                    "text": clean_text,
                    "confidence": round(float(getattr(seg, "avg_logprob", 0.0)), 3),
                })
    except Exception as e:
        logger.warning(f"Whisper transcription failed for {video_path}: {e}")
    finally:
        # Clean up temp WAV file immediately
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        if cleanup_tmp:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def associate_transcripts_to_segments(
    video_segments: List[Dict[str, Any]],
    transcripts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Associate timestamped transcript chunks with existing video scene segments.

    Enriches each video segment dict with:
      - transcript: List[Dict] (overlapping speech segments)
      - speech_text: str (concatenated spoken dialogue)
      - speech_detected: bool
    """
    if not video_segments:
        return []

    for seg in video_segments:
        s_start = float(seg.get("start_time", 0.0))
        s_end = float(seg.get("end_time", s_start + float(seg.get("duration", 5.0))))

        matching_transcripts: List[Dict[str, Any]] = []
        for t in transcripts:
            t_start = float(t.get("start_time", 0.0))
            t_end = float(t.get("end_time", t_start + 1.0))

            # Overlap condition: intervals intersect
            overlap = max(0.0, min(t_end, s_end) - max(t_start, s_start))
            if overlap > 0.25 or (s_start <= t_start < s_end):
                matching_transcripts.append(t)

        seg["transcript"] = matching_transcripts
        if matching_transcripts:
            spoken_texts = [t["text"] for t in matching_transcripts]
            combined_speech = " ".join(spoken_texts)
            seg["speech_text"] = combined_speech
            seg["speech_detected"] = True

            # Speech bonus for editorial selection
            seg["story_value"] = round(min(1.0, float(seg.get("story_value", 0.5)) + 0.08), 3)
            seg["selection_score"] = round(min(1.0, float(seg.get("selection_score", 0.5)) + 0.08), 3)

            # Enrich description
            existing_desc = seg.get("description", "")
            seg["description"] = f'{existing_desc} [Speech: "{combined_speech[:40]}..."]' if existing_desc else f'Speech: "{combined_speech[:50]}"'
        else:
            seg["speech_text"] = ""
            seg["speech_detected"] = False

    return video_segments
