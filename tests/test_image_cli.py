from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from perception import image_cli


class _FakeDetector:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def detect_image(self, image_path: Path, location: str | None = None):
        return [
            {
                "type": "裂缝",
                "confidence": 0.8123,
                "location": location or "未知区域",
                "severity": "high",
                "bounding_box": {"x1": 10.0, "y1": 20.0, "x2": 120.0, "y2": 180.0},
            }
        ]


def _write_test_image(path: Path) -> None:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    encoded.tofile(str(path))


def test_image_cli_saves_json_and_visualization(tmp_path, monkeypatch):
    image_path = tmp_path / "测试图.png"
    _write_test_image(image_path)

    monkeypatch.setattr(image_cli, "YOLODetector", _FakeDetector)

    out_dir = tmp_path / "outputs"
    rc = image_cli.main([
        str(image_path),
        "--location",
        "A区-3号楼",
        "--out-dir",
        str(out_dir),
        "--name",
        "demo",
    ])

    assert rc == 0

    json_files = list(out_dir.glob("demo_*.json"))
    viz_files = list(out_dir.glob("demo_*_annotated.jpg"))
    assert len(json_files) == 1
    assert len(viz_files) == 1

    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert data[0]["type"] == "裂缝"
    assert data[0]["location"] == "A区-3号楼"


def test_image_cli_no_viz(tmp_path, monkeypatch):
    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    monkeypatch.setattr(image_cli, "YOLODetector", _FakeDetector)

    out_dir = tmp_path / "outputs"
    rc = image_cli.main([str(image_path), "--out-dir", str(out_dir), "--name", "demo", "--no-viz"])

    assert rc == 0
    assert len(list(out_dir.glob("demo_*.json"))) == 1
    assert len(list(out_dir.glob("demo_*_annotated.jpg"))) == 0

