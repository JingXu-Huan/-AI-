"""
tests/test_perception.py

Unit tests for the perception layer.

These tests exercise the preprocessor, output formatter, visualisation
helper, and the detector's helper methods without requiring a GPU or a
pre-downloaded YOLO model file.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ImagePreprocessor tests
# ---------------------------------------------------------------------------

from perception.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def _make_image(self, h: int = 64, w: int = 64) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_process_returns_same_shape(self):
        img = self._make_image()
        pp = ImagePreprocessor()
        result = pp.process(img)
        assert result.shape == img.shape

    def test_process_does_not_modify_original(self):
        img = self._make_image()
        original = img.copy()
        pp = ImagePreprocessor(denoise=True, clahe=True)
        pp.process(img)
        np.testing.assert_array_equal(img, original)

    def test_empty_image_raises(self):
        pp = ImagePreprocessor()
        with pytest.raises(ValueError, match="empty image"):
            pp.process(np.array([]))

    def test_all_steps_disabled(self):
        img = self._make_image()
        pp = ImagePreprocessor(denoise=False, clahe=False, edge_enhancement=False)
        result = pp.process(img)
        # With no steps, output should equal input (modulo copy)
        np.testing.assert_array_equal(result, img)

    def test_edge_enhancement_output_shape(self):
        img = self._make_image()
        pp = ImagePreprocessor(denoise=False, clahe=False, edge_enhancement=True)
        result = pp.process(img)
        assert result.shape == img.shape

    def test_denoise_smooths_noise(self):
        """Gaussian blur should reduce pixel variance."""
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        pp = ImagePreprocessor(denoise=True, clahe=False, edge_enhancement=False)
        smoothed = pp.process(noisy)
        assert smoothed.std() <= noisy.std()


# ---------------------------------------------------------------------------
# output.py tests
# ---------------------------------------------------------------------------

from perception.utils.output import detection_to_json, format_detections


class TestOutputFormatter:
    """Tests for format_detections and detection_to_json."""

    def _raw(self, cls: str = "pipe_leak", conf: float = 0.92) -> dict:
        return {
            "class_name": cls,
            "confidence": conf,
            "bounding_box": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0},
        }

    def test_format_single_detection(self):
        result = format_detections([self._raw()], location="A区-3号楼")
        assert len(result) == 1
        rec = result[0]
        assert rec["type"] == "pipe_leak"
        assert rec["confidence"] == pytest.approx(0.92, abs=1e-4)
        assert rec["location"] == "A区-3号楼"
        assert rec["severity"] == "high"
        assert "bounding_box" in rec

    def test_severity_high(self):
        result = format_detections([self._raw(conf=0.80)])
        assert result[0]["severity"] == "high"

    def test_severity_medium(self):
        result = format_detections([self._raw(conf=0.60)])
        assert result[0]["severity"] == "medium"

    def test_severity_low(self):
        result = format_detections([self._raw(conf=0.30)])
        assert result[0]["severity"] == "low"

    def test_empty_raw_results(self):
        assert format_detections([]) == []

    def test_default_location(self):
        result = format_detections([self._raw()])
        assert result[0]["location"] == "未知区域"

    def test_detection_to_json_is_valid_json(self):
        dets = format_detections([self._raw()])
        json_str = detection_to_json(dets)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "pipe_leak"

    def test_multiple_detections(self):
        raws = [
            self._raw("road_crack", 0.55),
            self._raw("pothole", 0.88),
            self._raw("water_accumulation", 0.42),
        ]
        results = format_detections(raws, location="B区")
        assert len(results) == 3
        assert results[0]["severity"] == "medium"
        assert results[1]["severity"] == "high"
        assert results[2]["severity"] == "low"


# ---------------------------------------------------------------------------
# visualization.py tests
# ---------------------------------------------------------------------------

from perception.utils.visualization import draw_detections


class TestDrawDetections:
    """Tests for draw_detections."""

    def _image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _detection(self, severity: str = "high") -> dict:
        return {
            "type": "road_crack",
            "confidence": 0.85,
            "severity": severity,
            "location": "A区",
            "bounding_box": {"x1": 50, "y1": 50, "x2": 200, "y2": 150},
        }

    def test_returns_same_shape(self):
        img = self._image()
        result = draw_detections(img, [self._detection()])
        assert result.shape == img.shape

    def test_does_not_modify_original(self):
        img = self._image()
        original = img.copy()
        draw_detections(img, [self._detection()])
        np.testing.assert_array_equal(img, original)

    def test_empty_detections(self):
        img = self._image()
        result = draw_detections(img, [])
        # No boxes drawn → image unchanged
        np.testing.assert_array_equal(result, img)

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="empty image"):
            draw_detections(np.array([]), [])

    def test_all_severity_colours(self):
        """Verify the function runs without error for all severity levels."""
        img = self._image()
        dets = [self._detection(s) for s in ("high", "medium", "low")]
        result = draw_detections(img, dets)
        assert result.shape == img.shape
        # At least one pixel should be non-zero (boxes are drawn)
        assert result.sum() > 0


# ---------------------------------------------------------------------------
# YOLODetector helper-method tests (no model download required)
# ---------------------------------------------------------------------------

from perception.detector import YOLODetector


class TestYOLODetectorHelpers:
    """Tests for YOLODetector that do not require loading an actual model."""

    CONFIG = "config/detection_config.yaml"

    def _make_detector_no_model(self) -> YOLODetector:
        """Patch _load_model so no weights file is needed."""
        with patch.object(YOLODetector, "_load_model", return_value=MagicMock()):
            return YOLODetector(self.CONFIG)

    def test_load_image_ndarray(self, tmp_path):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = YOLODetector._load_image(arr)
        np.testing.assert_array_equal(result, arr)

    def test_load_image_file(self, tmp_path):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        loaded = YOLODetector._load_image(path)
        assert loaded.shape == (50, 50, 3)

    def test_load_image_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            YOLODetector._load_image(tmp_path / "no_such_file.jpg")

    def test_config_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            YOLODetector(tmp_path / "missing.yaml")

    def test_detect_image_returns_list(self, tmp_path):
        """Full detect_image path with mocked model output."""
        detector = self._make_detector_no_model()

        # Build a fake YOLO result object
        fake_box = MagicMock()
        fake_box.cls = [MagicMock(item=lambda: 2)]      # class 2 → pipe_leak
        fake_box.conf = [MagicMock(item=lambda: 0.92)]
        fake_box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        detector._model.predict.return_value = [fake_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect_image(img, location="A区-3号楼")

        assert isinstance(dets, list)
        assert len(dets) == 1
        assert dets[0]["type"] == "pipe_leak"
        assert dets[0]["location"] == "A区-3号楼"
        assert dets[0]["severity"] == "high"

    def test_detect_image_uses_model_names_fallback(self):
        """When config map misses a class, use the model's own label."""
        detector = self._make_detector_no_model()
        detector._model_names = {15: "cat"}

        fake_box = MagicMock()
        fake_box.cls = [MagicMock(item=lambda: 15)]
        fake_box.conf = [MagicMock(item=lambda: 0.85)]
        fake_box.xyxy = [MagicMock(tolist=lambda: [3.8, 26.8, 447.5, 421.3])]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        detector._model.predict.return_value = [fake_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect_image(img, location="测试区")

        assert len(dets) == 1
        assert dets[0]["type"] == "cat"
        assert dets[0]["location"] == "测试区"

    def test_detect_image_no_detections(self):
        detector = self._make_detector_no_model()
        fake_result = MagicMock()
        fake_result.boxes = []
        detector._model.predict.return_value = [fake_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect_image(img)
        assert dets == []
