"""
Typography & Subtitle Overlay Generator
Renders anti-aliased text overlays and translucent background pills onto video frames.
"""
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


def render_subtitle_frame(
    base_image: Image.Image,
    text: str,
    font_size: int = 42,
    pill_color: Tuple[int, int, int, int] = (0, 0, 0, 160),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bottom_margin: int = 120,
) -> Image.Image:
    """
    Overlays stylized narrative subtitles onto an image frame.
    """
    if not text:
        return base_image

    img = base_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Calculate text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center horizontally near bottom
    img_w, img_h = img.size
    x = (img_w - text_w) // 2
    y = img_h - bottom_margin - text_h

    # Draw rounded translucent pill background
    pad_x = 24
    pad_y = 12
    draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y],
        radius=16,
        fill=pill_color,
    )

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color)

    # Composite
    out = Image.alpha_composite(img, overlay)
    return out.convert("RGB")
