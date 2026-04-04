from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import main as entry


class _FakeDetector:
    def __init__(self, config_path: str | Path = "config/detection_config.yaml"):
        self.config_path = config_path

    def detect_image(self, image_path: Path, location: str | None = None):
        return [
            {
                "type": "裂缝",
                "confidence": 0.91,
                "location": location or "未知区域",
                "severity": "high",
                "bounding_box": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            }
        ]

    def detect_frame(self, frame: np.ndarray, location: str | None = None):
        return [
            {
                "type": "坑洞",
                "confidence": 0.66,
                "location": location or "未知区域",
                "severity": "medium",
                "bounding_box": {"x1": 5.0, "y1": 6.0, "x2": 15.0, "y2": 16.0},
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


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._index = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop_id: int):
        if prop_id == cv2.CAP_PROP_FPS:
            return 10.0
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0])
        return 0.0

    def release(self) -> None:
        return None


class _FakeWriter:
    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"FAKE-MP4")


def test_main_video_source_outputs_frame_json_and_viz(tmp_path, monkeypatch, capsys):
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)]
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-real-video")

    monkeypatch.setattr(entry, "YOLODetector", _FakeDetector)
    monkeypatch.setattr(entry, "_open_capture", lambda source: _FakeCapture(frames))
    monkeypatch.setattr(entry.cv2, "VideoWriter", _FakeWriter)
    monkeypatch.setattr(entry.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.chdir(tmp_path)

    rc = entry.main([str(video_path), "--frame-interval", "2", "--location", "A区-3号楼"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert len(parsed) == 2
    assert parsed[0]["frame_index"] == 0
    assert parsed[1]["frame_index"] == 2
    assert parsed[0]["detections"][0]["type"] == "坑洞"
    assert parsed[0]["detections"][0]["location"] == "A区-3号楼"

    output_dir = tmp_path / "outputs" / "demo"
    assert (output_dir / "demo.json").exists()
    assert (output_dir / "demo_annotated.mp4").exists()

