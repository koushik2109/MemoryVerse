"""
MemoryVerse - QualityValidator
Post-render quality and compliance diagnostic engine for rendered video reels.
Validates aspect ratio, resolution, duration, audio presence, black frames, and video integrity.
"""
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationResult:
    is_valid: bool
    resolution: Tuple[int, int]
    aspect_ratio: str
    duration_seconds: float
    fps: float
    has_audio: bool
    audio_sample_rate: int
    black_frame_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "aspect_ratio": self.aspect_ratio,
            "duration_seconds": round(self.duration_seconds, 2),
            "fps": round(self.fps, 2),
            "has_audio": self.has_audio,
            "audio_sample_rate": self.audio_sample_rate,
            "black_frame_count": self.black_frame_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
        }


class QualityValidator:
    """
    Validates rendered MP4 files prior to cloud storage upload.
    """

    @classmethod
    def probe_video_file(cls, filepath: str) -> Dict[str, Any]:
        """Runs ffprobe on the target MP4 file and parses streams."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file does not exist: {filepath}")

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            filepath,
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(res.stdout)
        except Exception as e:
            logger.warning(f"ffprobe execution failed, falling back to basic checks: {e}")
            return {}

    @classmethod
    def validate_video(
        cls,
        filepath: str,
        expected_aspect_ratio: str = "16:9",
        min_duration_seconds: float = 5.0,
        max_duration_seconds: float = 180.0,
    ) -> QualityValidationResult:
        """
        Validates video integrity and conformance against expected parameters.
        """
        errors: List[str] = []
        warnings: List[str] = []
        diagnostics: Dict[str, Any] = {}

        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        if file_size < 1000:
            errors.append(f"File size too small ({file_size} bytes). Likely corrupted.")
            return QualityValidationResult(
                is_valid=False,
                resolution=(0, 0),
                aspect_ratio="unknown",
                duration_seconds=0.0,
                fps=0.0,
                has_audio=False,
                audio_sample_rate=0,
                black_frame_count=0,
                errors=errors,
                warnings=warnings,
                diagnostics={"file_size": file_size},
            )

        probe_data = cls.probe_video_file(filepath)
        streams = probe_data.get("streams", [])
        fmt = probe_data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not video_stream:
            errors.append("No video stream found in rendered artifact.")
            width, height, fps = 0, 0, 0.0
        else:
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            fps_str = video_stream.get("r_frame_rate", "24/1")
            try:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 24.0
            except Exception:
                fps = 24.0

        duration_sec = float(fmt.get("duration", 0.0))
        if duration_sec < min_duration_seconds:
            errors.append(f"Video duration ({duration_sec:.1f}s) is shorter than minimum {min_duration_seconds}s.")
        elif duration_sec > max_duration_seconds:
            warnings.append(f"Video duration ({duration_sec:.1f}s) exceeds typical limit {max_duration_seconds}s.")

        # Aspect ratio check
        actual_aspect = "unknown"
        if width > 0 and height > 0:
            ratio_val = width / height
            if abs(ratio_val - (16 / 9)) < 0.15:
                actual_aspect = "16:9"
            elif abs(ratio_val - (9 / 16)) < 0.15:
                actual_aspect = "9:16"
            elif abs(ratio_val - 1.0) < 0.15:
                actual_aspect = "1:1"
            else:
                actual_aspect = f"{width}:{height}"

            if expected_aspect_ratio and actual_aspect != expected_aspect_ratio:
                warnings.append(
                    f"Rendered aspect ratio ({actual_aspect}) differs from requested ({expected_aspect_ratio})."
                )

        has_audio = audio_stream is not None
        audio_sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0

        if not has_audio:
            warnings.append("No audio stream detected in final render.")

        # Black frame detector (fast probe)
        black_frame_count = 0
        try:
            detect_cmd = [
                "ffmpeg",
                "-i", filepath,
                "-vf", "blackdetect=d=1.5:pic_th=0.98",
                "-an",
                "-f", "null",
                "-",
            ]
            black_res = subprocess.run(detect_cmd, capture_output=True, text=True)
            black_frame_count = black_res.stderr.count("black_start")
            if black_frame_count > 2:
                warnings.append(f"Detected {black_frame_count} prolonged black frame intervals.")
        except Exception:
            pass

        diagnostics["bitrate_kbps"] = int(fmt.get("bit_rate", 0)) // 1000 if fmt.get("bit_rate") else 0
        diagnostics["codec"] = video_stream.get("codec_name", "unknown") if video_stream else "unknown"

        is_valid = len(errors) == 0

        return QualityValidationResult(
            is_valid=is_valid,
            resolution=(width, height),
            aspect_ratio=actual_aspect,
            duration_seconds=duration_sec,
            fps=fps,
            has_audio=has_audio,
            audio_sample_rate=audio_sample_rate,
            black_frame_count=black_frame_count,
            errors=errors,
            warnings=warnings,
            diagnostics=diagnostics,
        )
