"""
MemoryVerse - AspectRatioComposer
Adaptive Multimodal Composition Engine for Video Generation.
Guarantees source media is NEVER stretched, warped, or blindly cropped.
Composes portrait media into landscape (and vice-versa) using adaptive cinematic backgrounds.
"""
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class TargetAspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"

    @classmethod
    def get_dimensions(cls, ratio: str, high_res: bool = False) -> Tuple[int, int]:
        if ratio == cls.RATIO_9_16.value:
            return (1080, 1920) if high_res else (720, 1280)
        elif ratio == cls.RATIO_1_1.value:
            return (1080, 1080) if high_res else (720, 720)
        elif ratio == cls.RATIO_4_3.value:
            return (1440, 1080) if high_res else (960, 720)
        else:  # 16:9 default
            return (1920, 1080) if high_res else (1280, 720)


class BackgroundStrategy(str, Enum):
    BLURRED_DUPLICATE = "blurred_duplicate"
    COLOR_GRADIENT = "color_gradient"
    DARKENED_DUPLICATE = "darkened_duplicate"
    ADAPTIVE_CINEMATIC = "adaptive_cinematic"


class CompositionFitMode(str, Enum):
    FIT = "fit"
    CROP = "crop"
    PAD = "pad"
    ADAPTIVE_BACKGROUND = "adaptive_background"


class AspectRatioComposer:
    """
    Intelligent composition engine for photos and videos.
    Transforms arbitrary source dimensions into target canvas while strictly preserving source aspect ratio.
    """

    @staticmethod
    def calculate_layout(
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        margin_pct: float = 0.07,
        subject_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates foreground scaling and placement within the target canvas.
        Returns:
            fg_w, fg_h: dimensions of foreground
            fg_x, fg_y: top-left coordinates on target canvas
            scale: scale factor applied to source
            is_aspect_mismatch: whether background extension is needed
        """
        src_aspect = src_w / max(1, src_h)
        target_aspect = target_w / max(1, target_h)

        aspect_diff = abs(src_aspect - target_aspect)
        is_mismatch = aspect_diff > 0.05

        if not is_mismatch:
            # Matches aspect ratio closely; scale to fill
            scale = max(target_w / src_w, target_h / src_h)
            fg_w = int(src_w * scale)
            fg_h = int(src_h * scale)
            fg_x = (target_w - fg_w) // 2
            fg_y = (target_h - fg_h) // 2
            return {
                "fg_w": fg_w,
                "fg_h": fg_h,
                "fg_x": fg_x,
                "fg_y": fg_y,
                "scale": scale,
                "is_aspect_mismatch": False,
                "fit_mode": CompositionFitMode.FIT.value,
            }

        # Aspect mismatch (e.g. portrait 9:16 inside landscape 16:9 canvas)
        max_fw = int(target_w * (1.0 - 2 * margin_pct))
        max_fh = int(target_h * (1.0 - 2 * margin_pct))

        scale = min(max_fw / src_w, max_fh / src_h)
        fg_w = max(1, int(src_w * scale))
        fg_h = max(1, int(src_h * scale))

        # Default center placement
        fg_x = (target_w - fg_w) // 2
        fg_y = (target_h - fg_h) // 2

        # Subject-aware bias if subject box is provided
        if subject_box:
            sb_x, sb_y, sb_w, sb_h = subject_box
            sub_center_x = (sb_x + sb_w / 2) / src_w
            sub_center_y = (sb_y + sb_h / 2) / src_h
            bias_x = int((0.5 - sub_center_x) * 20)
            bias_y = int((0.5 - sub_center_y) * 20)
            fg_x = max(0, min(target_w - fg_w, fg_x + bias_x))
            fg_y = max(0, min(target_h - fg_h, fg_y + bias_y))

        return {
            "fg_w": fg_w,
            "fg_h": fg_h,
            "fg_x": fg_x,
            "fg_y": fg_y,
            "scale": scale,
            "is_aspect_mismatch": True,
            "fit_mode": CompositionFitMode.ADAPTIVE_BACKGROUND.value,
        }

    @staticmethod
    def generate_background(
        src_image: Image.Image,
        target_w: int,
        target_h: int,
        strategy: str = BackgroundStrategy.ADAPTIVE_CINEMATIC.value,
    ) -> Image.Image:
        """
        Creates an ambient, visually harmonious background canvas for letterbox/pillarbox areas.
        """
        # Downscale for performance during background generation
        small_w, small_h = max(16, target_w // 8), max(16, target_h // 8)
        src_small = src_image.resize((small_w, small_h), Image.Resampling.BILINEAR)

        if strategy == BackgroundStrategy.COLOR_GRADIENT.value:
            # Extract dominant tone and build vertical gradient
            np_src = np.array(src_small)
            avg_color = np_src.mean(axis=(0, 1)).astype(int)
            c1 = tuple(np.clip(avg_color * 0.4, 0, 255).astype(int))
            c2 = tuple(np.clip(avg_color * 0.15, 0, 255).astype(int))

            grad = Image.new("RGB", (target_w, target_h))
            draw = ImageDraw.Draw(grad)
            for y in range(target_h):
                p = y / max(1, target_h)
                r = int(c1[0] * (1 - p) + c2[0] * p)
                g = int(c1[1] * (1 - p) + c2[1] * p)
                b = int(c1[2] * (1 - p) + c2[2] * p)
                draw.line([(0, y), (target_w, y)], fill=(r, g, b))
            return grad

        elif strategy == BackgroundStrategy.DARKENED_DUPLICATE.value:
            bg = src_small.resize((target_w, target_h), Image.Resampling.BILINEAR)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=15))
            dark = Image.new("RGBA", (target_w, target_h), (10, 10, 15, 180))
            return Image.alpha_composite(bg.convert("RGBA"), dark).convert("RGB")

        else:
            # ADAPTIVE_CINEMATIC and BLURRED_DUPLICATE:
            # Soft dual-pass Gaussian blur with a subtle warm vignette overlay
            bg = src_small.resize((target_w, target_h), Image.Resampling.BILINEAR)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=25))

            vignette = Image.new("RGBA", (target_w, target_h), (12, 10, 18, 140))
            bg_composite = Image.alpha_composite(bg.convert("RGBA"), vignette).convert("RGB")
            return bg_composite

    @classmethod
    def compose_image(
        cls,
        src_image: Image.Image,
        target_size: Tuple[int, int] = (1280, 720),
        strategy: str = BackgroundStrategy.ADAPTIVE_CINEMATIC.value,
        add_subtle_border: bool = True,
    ) -> Image.Image:
        """
        Composes a single image into the target canvas size.
        Guarantees source is never distorted or stretched.
        """
        tw, th = target_size
        src_rgb = src_image.convert("RGB")
        ow, oh = src_rgb.size

        layout = cls.calculate_layout(ow, oh, tw, th)

        if not layout["is_aspect_mismatch"]:
            # Crop to fill without distortion
            scale = max(tw / ow, th / oh)
            cw, ch = int(ow * scale), int(oh * scale)
            resized = src_rgb.resize((cw, ch), Image.Resampling.LANCZOS)
            x_offset = (cw - tw) // 2
            y_offset = (ch - th) // 2
            return resized.crop((x_offset, y_offset, x_offset + tw, y_offset + th))

        # Adaptive background composition
        canvas = cls.generate_background(src_rgb, tw, th, strategy=strategy)
        fg_w, fg_h = layout["fg_w"], layout["fg_h"]
        fg_x, fg_y = layout["fg_x"], layout["fg_y"]

        fg_resized = src_rgb.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

        if add_subtle_border:
            # Subtle 2px translucent border around original photo
            border_box = Image.new("RGBA", (fg_w + 6, fg_h + 6), (0, 0, 0, 0))
            draw = ImageDraw.Draw(border_box)
            draw.rectangle([0, 0, fg_w + 5, fg_h + 5], outline=(255, 255, 255, 75), width=2)
            border_box.paste(fg_resized, (3, 3))
            canvas.paste(border_box, (fg_x - 3, fg_y - 3), border_box)
        else:
            canvas.paste(fg_resized, (fg_x, fg_y))

        return canvas
