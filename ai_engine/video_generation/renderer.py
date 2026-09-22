"""
Video Composition Renderer
Orchestrates shot lists, Ken Burns camera motion, subtitle overlays,
and procedural ambient background music into final cinematic MP4 video recaps.
"""
import os
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

try:
    from moviepy import VideoClip, VideoFileClip, AudioArrayClip, concatenate_videoclips
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

    class _DummyClip:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self

        def __getattr__(self, name: str) -> Any:
            return self

    VideoClip: Any = _DummyClip
    VideoFileClip: Any = _DummyClip
    AudioArrayClip: Any = _DummyClip

    def concatenate_videoclips(*args: Any, **kwargs: Any) -> Any:
        return _DummyClip()

from ai_engine.video_generation.ken_burns import get_crop_window
from ai_engine.video_generation.subtitles import render_subtitle_frame
from ai_engine.video_generation.audio_synth import synthesize_ambient_soundtrack
from ai_engine.video_generation.tts_engine import EmotionTTSEngine

logger = logging.getLogger(__name__)


class VideoRenderer:
    """
    Assembles curated media assets into cinematic videos with camera motion,
    subtitles, and synchronized ambient audio.
    """

    def __init__(self, fps: int = 24, resolution: str = "720x1280"):
        self.fps = fps
        self.resolution = resolution  # Default portrait (720x1280) for mobile Flutter app

    def _parse_resolution(self, res_str: str) -> Tuple[int, int]:
        """Parses '720x1280' or '1280x720' into (width, height)."""
        try:
            parts = res_str.lower().split("x")
            return int(parts[0]), int(parts[1])
        except Exception:
            return 720, 1280

    def render_spec(
        self,
        shot_list: List[Dict[str, Any]],
        audio_mood: str = "calm",
        target_duration: float = 30.0,
        output_mp4_path: str = "/tmp/recap_preview.mp4",
    ) -> Dict[str, Any]:
        """
        Builds the video specification and timeline cues.
        """
        total_shots = len(shot_list)
        return {
            "output_path": output_mp4_path,
            "duration_seconds": target_duration,
            "fps": self.fps,
            "resolution": self.resolution,
            "total_shots": total_shots,
            "audio_mood": audio_mood,
            "status": "ready_for_render",
            "shots": shot_list,
        }

    def create_image_shot_clip(
        self,
        img_path: str,
        duration: float = 4.0,
        target_size: Tuple[int, int] = (720, 1280),
        motion_type: str = "zoom_in",
        narration: Optional[str] = None,
    ) -> Any:
        """
        Renders a cinematic Ken Burns pan-and-zoom clip with subtitle overlay.
        Pre-computes start and end keyframes for high-performance vectorized rendering.
        """
        if not HAS_MOVIEPY:
            raise RuntimeError("MoviePy is required for video rendering")

        tw, th = target_size
        with Image.open(img_path) as orig:
            orig_img = orig.convert("RGB")
            ow, oh = orig_img.size

            # Ambient blurred backdrop
            small_w, small_h = max(1, tw // 4), max(1, th // 4)
            bg_small = orig_img.resize((small_w, small_h), Image.Resampling.BILINEAR)
            bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=8))
            bg_base = bg_small.resize((tw, th), Image.Resampling.BILINEAR)
            dark_overlay = Image.new("RGBA", (tw, th), (12, 8, 20, 150))
            bg_base = Image.alpha_composite(bg_base.convert("RGBA"), dark_overlay).convert("RGB")

            # Foreground frame
            max_fw, max_fh = int(tw * 0.88), int(th * 0.88)
            scale_fg = min(max_fw / ow, max_fh / oh)
            fg_w, fg_h = max(1, int(ow * scale_fg)), max(1, int(oh * scale_fg))
            fg_base = orig_img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

            border_img = Image.new("RGBA", (fg_w + 6, fg_h + 6), (0, 0, 0, 0))
            b_draw = ImageDraw.Draw(border_img)
            b_draw.rectangle([0, 0, fg_w + 5, fg_h + 5], outline=(255, 255, 255, 90), width=2)
            border_img.paste(fg_base, (3, 3))

        # Start keyframe (p = 0.0)
        z0 = 1.00 if "zoom_in" in motion_type else (1.10 if "zoom_out" in motion_type else 1.05)
        sx0 = -18 if "pan_right" in motion_type else (18 if "pan_left" in motion_type else 0)
        sy0 = -8 if "zoom_in" in motion_type else (8 if "zoom_out" in motion_type else 0)

        w0, h0 = int(border_img.width * z0), int(border_img.height * z0)
        fg0 = border_img.resize((w0, h0), Image.Resampling.BILINEAR) if z0 != 1.0 else border_img
        canvas0 = bg_base.copy()
        canvas0.paste(fg0, ((tw - w0) // 2 + sx0, (th - h0) // 2 + sy0), fg0)

        # End keyframe (p = 1.0)
        z1 = 1.10 if "zoom_in" in motion_type else (1.00 if "zoom_out" in motion_type else 1.05)
        sx1 = 18 if "pan_right" in motion_type else (-18 if "pan_left" in motion_type else 0)
        sy1 = 8 if "zoom_in" in motion_type else (-8 if "zoom_out" in motion_type else 0)

        w1, h1 = int(border_img.width * z1), int(border_img.height * z1)
        fg1 = border_img.resize((w1, h1), Image.Resampling.BILINEAR) if z1 != 1.0 else border_img
        canvas1 = bg_base.copy()
        canvas1.paste(fg1, ((tw - w1) // 2 + sx1, (th - h1) // 2 + sy1), fg1)

        # Subtitle overlay if present
        if narration and narration.strip():
            sub_text = narration.strip()
            canvas0 = render_subtitle_frame(canvas0, sub_text, font_size=28, bottom_margin=int(th * 0.12))
            canvas1 = render_subtitle_frame(canvas1, sub_text, font_size=28, bottom_margin=int(th * 0.12))

        arr0 = np.array(canvas0, dtype=np.float32)
        arr1 = np.array(canvas1, dtype=np.float32)

        fade_dur = min(0.35, duration * 0.15)

        def make_frame(t: float) -> np.ndarray:
            p = max(0.0, min(1.0, t / duration))
            # Smooth cosine ease curve
            ease_p = 0.5 - 0.5 * math.cos(math.pi * p)
            f = (1.0 - ease_p) * arr0 + ease_p * arr1

            # Cross-fade in & out
            if t < fade_dur and fade_dur > 0:
                f *= (t / fade_dur)
            elif t > (duration - fade_dur) and fade_dur > 0:
                f *= max(0.0, (duration - t) / fade_dur)

            return np.clip(f, 0, 255).astype(np.uint8)

        return VideoClip(make_frame, duration=duration)

    def create_title_card_clip(
        self,
        title: str,
        subtitle: str = "",
        duration: float = 3.0,
        target_size: Tuple[int, int] = (720, 1280),
        bg_img_path: Optional[str] = None,
    ) -> Any:
        """
        Renders a cinematic title card with glowing branding and soft background blur.
        """
        if not HAS_MOVIEPY:
            raise RuntimeError("MoviePy is required for video rendering")

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
                bg = bg.crop((l, t, l + tw, t + th)).filter(ImageFilter.GaussianBlur(radius=18))
                dark = Image.new("RGBA", (tw, th), (10, 6, 20, 215))
                bg = Image.alpha_composite(bg.convert("RGBA"), dark).convert("RGB")
        else:
            bg = Image.new("RGB", (tw, th), color=(14, 10, 24))

        draw = ImageDraw.Draw(bg)
        try:
            font_tag = ImageFont.truetype("arial.ttf", size=20)
            font_title = ImageFont.truetype("arial.ttf", size=36)
            font_sub = ImageFont.truetype("arial.ttf", size=22)
        except Exception:
            try:
                font_tag = ImageFont.load_default(size=20)
                font_title = ImageFont.load_default(size=36)
                font_sub = ImageFont.load_default(size=22)
            except TypeError:
                font_tag = font_title = font_sub = ImageFont.load_default()

        tag_text = "✦  MEMORYVERSE AI REEL  ✦"
        draw.text((tw // 2, th // 2 - 70), tag_text, font=font_tag, fill=(215, 185, 255), anchor="mm")

        disp_title = title if len(title) <= 32 else title[:29] + "..."
        draw.text((tw // 2, th // 2 - 15), disp_title, font=font_title, fill=(255, 255, 255), anchor="mm")

        if subtitle:
            draw.text((tw // 2, th // 2 + 45), subtitle, font=font_sub, fill=(185, 175, 205), anchor="mm")

        arr_title = np.array(bg, dtype=np.float32)

        def make_frame(t: float) -> np.ndarray:
            f = arr_title.copy()
            fade_dur = 0.4
            if t < fade_dur:
                f *= (t / fade_dur)
            elif t > (duration - fade_dur):
                f *= max(0.0, (duration - t) / fade_dur)
            return np.clip(f, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration)
        sr = 44100
        n_samples = int(duration * sr)
        silent_audio = AudioArrayClip(np.zeros((n_samples, 2), dtype=np.float32), fps=sr)
        return clip.with_audio(silent_audio)

    def render_video(
        self,
        shot_list: List[Dict[str, Any]],
        output_mp4_path: str,
        title: str = "Memory Recap",
        subtitle: str = "",
        audio_mood: str = "calm",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes full video composition:
        1. Compiles opening title card.
        2. Renders each shot with Ken Burns camera curves and subtitle overlays.
        3. Synthesizes a matching 4-chord ambient soundtrack.
        4. Writes the final MP4 video file.
        """
        if not HAS_MOVIEPY:
            raise RuntimeError("MoviePy is required for video rendering")

        if target_size is None:
            target_size = self._parse_resolution(self.resolution)

        clips = []

        # Find first valid image path for title backdrop
        first_img = None
        for s in shot_list:
            p = s.get("asset_path") or s.get("file_path") or ""
            if p and os.path.exists(p) and (s.get("asset_type") == "image" or not p.endswith(".mp4")):
                first_img = p
                break

        # 1. Opening title card
        title_duration = 2.5
        title_clip = self.create_title_card_clip(
            title=title,
            subtitle=subtitle,
            duration=title_duration,
            target_size=target_size,
            bg_img_path=first_img,
        )
        clips.append(title_clip)

        # 2. Render each scene shot with emotion-aware speech synchronization
        tts_engine = EmotionTTSEngine()
        speech_segments: List[Tuple[float, np.ndarray]] = []
        current_timeline = title_duration

        for idx, shot in enumerate(shot_list):
            asset_path = shot.get("asset_path") or shot.get("file_path") or ""
            if not asset_path or not os.path.exists(asset_path):
                logger.warning(f"Shot {idx + 1} asset not found: {asset_path}")
                continue

            duration = float(shot.get("duration") or shot.get("duration_seconds") or 4.0)
            motion_info = shot.get("camera_motion", {})
            motion_type = motion_info.get("motion_type", "zoom_in") if isinstance(motion_info, dict) else str(motion_info or "zoom_in")
            narration = (shot.get("narration") or "").strip()
            asset_type = shot.get("asset_type", "image")

            # Synthesize emotion-aware speech for narration
            if narration:
                try:
                    speech_arr, speech_dur, _ = tts_engine.synthesize_sync(
                        text=narration,
                        emotion=audio_mood,
                    )
                    if speech_dur > 0:
                        # Ensure shot is long enough for the speech to complete naturally
                        duration = max(duration, speech_dur + 0.6)
                        speech_segments.append((current_timeline + 0.2, speech_arr))
                except Exception as tts_err:
                    logger.warning(f"Shot {idx + 1} TTS synthesis failed: {tts_err}")

            try:
                if asset_type == "video" or asset_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                    v_clip = VideoFileClip(asset_path).resized(new_size=target_size)
                    v_dur = float(v_clip.duration or 0.0)
                    start_t = float(shot.get("start_time", 0.0) or 0.0)
                    end_t = float(shot.get("end_time", 0.0) or 0.0)

                    if end_t > start_t and v_dur > 0:
                        v_clip = v_clip.subclipped(max(0.0, start_t), min(v_dur, end_t))
                    elif v_dur > (duration + 1.0):
                        v_clip = v_clip.subclipped(0, min(v_dur, duration))
                    clips.append(v_clip)
                    current_timeline += float(v_clip.duration or duration)
                else:
                    clip = self.create_image_shot_clip(
                        img_path=asset_path,
                        duration=duration,
                        target_size=target_size,
                        motion_type=motion_type,
                        narration=narration,
                    )
                    clips.append(clip)
                    current_timeline += duration
            except Exception as e:
                logger.error(f"Failed to render shot {idx + 1} ({asset_path}): {e}")

        if not clips:
            raise RuntimeError("No clips could be successfully rendered.")

        final_clip = concatenate_videoclips(clips, method="compose")
        total_duration = float(final_clip.duration or current_timeline)
        sr = 44100
        total_samples = max(1, int(total_duration * sr))

        # 3. Procedural ambient BGM + Emotion-Aware TTS Voiceover Mixing with Audio Ducking
        try:
            raw_bgm = synthesize_ambient_soundtrack(
                mood=audio_mood,
                duration_seconds=total_duration,
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

            ambient_clip = AudioArrayClip(mixed_audio, fps=sr)
            final_clip = final_clip.with_audio(ambient_clip)
        except Exception as e:
            logger.warning(f"Failed to attach synthesized soundtrack: {e}")

        # 4. Render to MP4
        cpu_threads = max(2, (os.cpu_count() or 4) - 1)
        final_clip.write_videofile(
            output_mp4_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            fps=self.fps,
            threads=cpu_threads,
            logger=None,
        )

        final_clip.close()
        for c in clips:
            if hasattr(c, "close"):
                try:
                    c.close()
                except Exception:
                    pass

        return {
            "output_path": output_mp4_path,
            "total_shots": len(clips) - 1,
            "duration_seconds": round(total_duration, 2),
            "audio_mood": audio_mood,
            "status": "completed",
        }
