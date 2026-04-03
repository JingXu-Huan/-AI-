from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import main as entry


class _FakeDetector:
    def __init__(self, config_path: str | Path = "config/detection_config.yaml"):
        self.config_path = config_path

    def detect_image(self, image_path: Path):
        return [
            {
                "type": "裂缝",
                "confidence": 0.91,
                "location": "未知区域",
                "severity": "high",
                "bounding_box": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            }
        ]


def _write_image(path: Path) -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    encoded.tofile(str(path))


def test_detect_image_json_returns_json(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    _write_image(image_path)

    monkeypatch.setattr(entry, "YOLODetector", _FakeDetector)

    result = entry.detect_image_json(image_path)
    parsed = json.loads(result)

    assert isinstance(result, str)
    assert parsed[0]["type"] == "裂缝"


def test_main_prints_json_and_saves_outputs_in_same_dir(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "sample.png"
    _write_image(image_path)

    monkeypatch.setattr(entry, "YOLODetector", _FakeDetector)
    monkeypatch.chdir(tmp_path)

    rc = entry.main([str(image_path)])
    captured = capsys.readouterr()
    out = captured.out.strip()

    assert rc == 0
    assert json.loads(out)[0]["type"] == "裂缝"

    output_dir = tmp_path / "outputs" / "sample"
    json_path = output_dir / "sample.json"
    viz_path = output_dir / "sample_annotated.jpg"

    assert output_dir.is_dir()
    assert json_path.exists()
    assert viz_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["type"] == "裂缝"
    assert viz_path.stat().st_size > 0
