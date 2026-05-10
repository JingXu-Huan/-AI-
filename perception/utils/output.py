"""
perception/utils/output.py

Converts raw YOLO detection results into the project's standard structured
JSON format:

  {
    "type": "pipe_leak",
    "confidence": 0.92,
    "location": "A区-3号楼",
    "severity": "high",
    "bounding_box": {"x1": 120, "y1": 80, "x2": 340, "y2": 200}
  }
"""

from __future__ import annotations

import json
import math
from typing import Any


def _get_severity(confidence: float, high_threshold: float = 0.75, medium_threshold: float = 0.50) -> str:
    """Map a confidence score to a severity level string."""
    if confidence >= high_threshold:
        return "high"
    if confidence >= medium_threshold:
        return "medium"
    return "low"


def format_detections(
    raw_results: list[dict[str, Any]],
    location: str = "未知区域",
    high_threshold: float = 0.75,
    medium_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Convert a list of raw detection dicts into structured output records.

    Parameters
    ----------
    raw_results : list[dict]
        Each entry must have keys: ``class_name``, ``confidence``,
        ``bounding_box`` (dict with x1/y1/x2/y2).
    location : str
        Human-readable location tag to embed in every record.
    high_threshold : float
        Confidence threshold for "high" severity.
    medium_threshold : float
        Confidence threshold for "medium" severity.

    Returns
    -------
    list[dict]
        Structured detection records.
    """
    structured: list[dict[str, Any]] = []
    for det in raw_results:
        bb = det["bounding_box"]
        # 计算对角线长度 (√(w² + h²)) 并记为道路缺陷目标长度
        w = bb["x2"] - bb["x1"]
        h = bb["y2"] - bb["y1"]
        diagonal_length = math.sqrt(w**2 + h**2)

        record: dict[str, Any] = {
            "type": det["class_name"],
            "confidence": round(float(det["confidence"]), 4),
            "location": location,
            "severity": _get_severity(
                float(det["confidence"]), high_threshold, medium_threshold
            ),
            "bounding_box": bb,
            "damage_length": round(diagonal_length, 2),
        }
        structured.append(record)
    return structured


def detection_to_json(
    detections: list[dict[str, Any]],
    indent: int = 2,
) -> str:
    """Serialise a list of structured detection records to a JSON string.

    Parameters
    ----------
    detections : list[dict]
        Output of :func:`format_detections`.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    return json.dumps(detections, ensure_ascii=False, indent=indent)
