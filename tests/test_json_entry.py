from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception import json_entry


class _FakeDetector:
    def __init__(self, config_path: str | Path):
        self.config_path = config_path

    def detect_image(self, image_path: Path, location: str | None = None):
        return [
            {
                "type": "裂缝",
                "confidence": 0.77,
                "location": location or "未知区域",
                "severity": "high",
                "bounding_box": {"x1": 1.0, "y1": 2.0, "x2": 100.0, "y2": 200.0},
            }
        ]


def _write_image(path: Path) -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    encoded.tofile(str(path))


def test_detect_image_to_json_returns_json_string(tmp_path, monkeypatch):
    image_path = tmp_path / "测试.png"
    _write_image(image_path)
    monkeypatch.setattr(json_entry, "YOLODetector", _FakeDetector)

    result = json_entry.detect_image_to_json(image_path, location="A区-3号楼")
    parsed = json.loads(result)

    assert isinstance(result, str)
    assert parsed[0]["type"] == "裂缝"
    assert parsed[0]["location"] == "A区-3号楼"


def test_detect_image_to_json_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        json_entry.detect_image_to_json(tmp_path / "missing.png")

