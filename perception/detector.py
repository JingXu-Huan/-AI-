"""
perception/detector.py

YOLO-based damage detector for campus infrastructure inspection.

Supports:
  - road_crack      (路面裂缝)
  - pothole         (坑洞)
  - pipe_leak       (管道泄漏)
  - water_accumulation (积水/漏水)
  - equipment_damage   (设备损坏)

Usage example
-------------
>>> from perception import YOLODetector
>>> detector = YOLODetector("config/detection_config.yaml")
>>> detections = detector.detect_image("path/to/image.jpg", location="A区-3号楼")
>>> print(detections)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .preprocessor import ImagePreprocessor
from .utils.output import format_detections


class YOLODetector:
    """Detect infrastructure damage in images using a YOLO model.

    Parameters
    ----------
    config_path : str | Path
        Path to ``detection_config.yaml``.
    """

    def __init__(self, config_path: str | Path = "config/detection_config.yaml") -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        model_cfg = cfg.get("model", {})
        self._weights: str = model_cfg.get("weights", "yolov8n.pt")
        self._conf_threshold: float = float(model_cfg.get("confidence_threshold", 0.4))
        self._iou_threshold: float = float(model_cfg.get("iou_threshold", 0.45))
        self._max_det: int = int(model_cfg.get("max_detections", 100))
        self._imgsz: int = int(model_cfg.get("image_size", 640))

        self._class_map: dict[int, str] = {
            int(k): str(v) for k, v in cfg.get("class_map", {}).items()
        }
        self._default_location: str = cfg.get("default_location", "未知区域")

        severity_cfg = cfg.get("severity", {})
        self._high_threshold: float = float(severity_cfg.get("high", 0.75))
        self._medium_threshold: float = float(severity_cfg.get("medium", 0.50))

        preproc_cfg = cfg.get("preprocessing", {})
        self._preprocessor = ImagePreprocessor(
            denoise=bool(preproc_cfg.get("denoise", True)),
            clahe=bool(preproc_cfg.get("clahe", True)),
            edge_enhancement=bool(preproc_cfg.get("edge_enhancement", False)),
        )

        self._model = self._load_model()
        self._model_names = self._extract_model_names(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_image(
        self,
        image_source: str | Path | np.ndarray,
        location: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run damage detection on a single image.

        Parameters
        ----------
        image_source : str | Path | np.ndarray
            A file path or an already-loaded BGR numpy array.
        location : str | None
            Location tag to embed in every detection record.  Falls back to
            the default defined in the config file.

        Returns
        -------
        list[dict]
            Structured detection records (see :mod:`perception.utils.output`).
        """
        image = self._load_image(image_source)
        preprocessed = self._preprocessor.process(image)
        raw = self._run_inference(preprocessed)
        return format_detections(
            raw,
            location=location or self._default_location,
            high_threshold=self._high_threshold,
            medium_threshold=self._medium_threshold,
        )

    def detect_frame(
        self,
        frame: np.ndarray,
        location: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience alias for detecting a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame as produced by ``cv2.VideoCapture.read()``.
        location : str | None
            Location tag for detections.

        Returns
        -------
        list[dict]
            Structured detection records.
        """
        return self.detect_image(frame, location=location)

    def detect_video(
        self,
        video_path: str | Path,
        location: str | None = None,
        frame_interval: int = 30,
    ) -> list[dict[str, Any]]:
        """Process a video file and return aggregated detection records.

        Parameters
        ----------
        video_path : str | Path
            Path to the video file.
        location : str | None
            Location tag for all detections.
        frame_interval : int
            Process every *n*-th frame (default: every 30 frames ≈ 1 fps at
            30 fps input).

        Returns
        -------
        list[dict]
            All detection records across the processed frames.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        all_detections: list[dict[str, Any]] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    dets = self.detect_frame(frame, location=location)
                    all_detections.extend(dets)
                frame_idx += 1
        finally:
            cap.release()

        return all_detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Import ultralytics and load the YOLO model.

        Ultralytics is imported lazily so that modules that only use
        :class:`ImagePreprocessor` or the output utilities do not require a
        GPU/CUDA environment.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for YOLODetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        model = YOLO(self._weights)

        return model

    @staticmethod
    def _extract_model_names(model: Any) -> dict[int, str]:
        """从加载的 YOLO 模型中提取人类可读的类名."""
        candidates = (
            getattr(model, "names", None),
            getattr(getattr(model, "model", None), "names", None),
        )

        for names in candidates:
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, (list, tuple)):
                return {int(i): str(v) for i, v in enumerate(names)}

        return {}

    def _run_inference(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO inference and convert results to raw detection dicts."""
        results = self._model.predict(
            source=image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            max_det=self._max_det,
            imgsz=self._imgsz,
            verbose=False,
        )

        raw_detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                class_name = self._class_map.get(
                    cls_idx,
                    self._model_names.get(cls_idx, f"class_{cls_idx}"),
                )
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        },
                    }
                )

        return raw_detections

    @staticmethod
    def _load_image(source: str | Path | np.ndarray) -> np.ndarray:
        """Load an image from a file path or return an ndarray as-is."""
        if isinstance(source, np.ndarray):
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2.imread failed to load: {path}")
        return img
