"""
perception/utils/visualization.py

Draw bounding boxes and labels on an image for inspection and debugging.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Colour palette (BGR) per severity level
_SEVERITY_COLOURS: dict[str, tuple[int, int, int]] = {
    "high": (0, 0, 255),      # Red
    "medium": (0, 165, 255),  # Orange
    "low": (0, 255, 0),       # Green
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
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

    Returns
    -------
    np.ndarray
        Annotated copy of *image* (original is not modified).
    """
    if image is None or image.size == 0:
        raise ValueError("draw_detections() received an empty image.")

    annotated = image.copy()

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
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _THICKNESS
        )
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w, label_y1 + text_h + baseline + 4),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_y1 + text_h + 2),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
