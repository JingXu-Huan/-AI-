"""
perception/utils/visualization.py

Draw bounding boxes and labels on an image for inspection and debugging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colour palette (BGR) per severity level
_SEVERITY_COLOURS: dict[str, tuple[int, int, int]] = {
    "high": (0, 0, 255),      # Red
    "medium": (0, 165, 255),  # Orange
    "low": (0, 255, 0),       # Green
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    """Save an image to *output_path*, including non-ASCII Windows paths.

    OpenCV's ``imwrite`` can fail on some Windows environments when the path
    contains Chinese characters. This helper falls back to ``imencode`` +
    ``tofile`` which handles such paths reliably.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix or ".jpg"
    if not ext.startswith("."):
        ext = f".{ext}"

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Failed to encode image for saving: {path}")
    encoded.tofile(str(path))


def _find_chinese_font() -> str | None:
    """Return a common Chinese font path on Windows if available."""
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        "msyh.ttc",  # Microsoft YaHei
        "msyh.ttf",
        "simhei.ttf",  # SimHei
        "simsun.ttc",  # SimSun
        "simsun.ttf",
        "mingliu.ttc",
    )
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return str(path)
    return None


def _load_font(size: int = 20, font_path: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font that can render Chinese labels; fall back gracefully."""
    path = font_path or _find_chinese_font()
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_text_with_pillow(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> np.ndarray:
    """Draw Unicode text onto a BGR image using Pillow."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(origin, text, font=font, fill=fill)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _measure_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    """Measure text size using Pillow so Chinese glyphs are handled correctly."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left), int(bottom - top)


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    font_path: str | None = None,
) -> np.ndarray:
    """Overlay bounding boxes and labels on *image* for each detection.

    Parameters
    ----------
    image : np.ndarray
        BGR image (as returned by ``cv2.imread``).
    detections : list[dict]
        Structured detection records produced by
        :func:`perception.utils.output.format_detections`.
        Each record must contain ``bounding_box`` (x1/y1/x2/y2),
        ``type``, ``confidence``, and ``severity``.
    font_path : str | None
        Optional path to a TrueType font that supports Chinese.

    Returns
    -------
    np.ndarray
        Annotated copy of *image* (original is not modified).
    """
    if image is None or image.size == 0:
        raise ValueError("draw_detections() received an empty image.")

    annotated = image.copy()
    font = _load_font(size=20, font_path=font_path)

    for det in detections:
        bbox = det["bounding_box"]
        x1, y1, x2, y2 = (
            int(bbox["x1"]),
            int(bbox["y1"]),
            int(bbox["x2"]),
            int(bbox["y2"]),
        )
        severity = det.get("severity", "low")
        colour = _SEVERITY_COLOURS.get(severity, _SEVERITY_COLOURS["low"])

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, _THICKNESS)

        # Label background + text
        label = f"{det['type']} {det['confidence']:.2f} [{severity}]"
        text_w, text_h = _measure_text(label, font)
        padding = 4
        label_y1 = max(y1 - text_h - padding, 0)
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (
                min(x1 + text_w + padding * 2, annotated.shape[1] - 1),
                min(label_y1 + text_h + padding, annotated.shape[0] - 1),
            ),
            colour,
            cv2.FILLED,
        )
        annotated = _draw_text_with_pillow(
            annotated,
            label,
            (x1 + padding, label_y1 + 1),
            font,
            (255, 255, 255),
        )

    return annotated
