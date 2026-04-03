"""
perception/detector.py

基于YOLO的校园道路损伤检测器。

支持：
  - road_crack      (路面裂缝)
  - pothole         (坑洞)
  - Manhole         (井盖)
  - Patch-Crack     (修补区域上的裂缝)
  - Patch-Net       (修补区域的网裂)
  - Patch-Pothole   (修补区域上的坑洞)

用法示例
--------
>>> from perception import YOLODetector
>>> detector = YOLODetector("config/detection_config.yaml")
>>> detections = detector.detect_image("path/to/image.jpg", location="A区-3号楼")
>>> print(detections)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .class_map_sync import compare_class_maps, load_names
from .preprocessor import ImagePreprocessor
from .utils.output import format_detections


class YOLODetector:
    """使用YOLO模型检测图片中的基础设施损伤。

    参数
    ------
    config_path : str | Path
        ``detection_config.yaml`` 的路径。
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
        self._display_class_map: dict[str, str] = {
            str(k): str(v) for k, v in cfg.get("display_class_map", {}).items()
        }
        self._apply_class_map_sync(cfg)
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
    # 公共接口
    # ------------------------------------------------------------------

    def detect_image(
        self,
        image_source: str | Path | np.ndarray,
        location: str | None = None,
    ) -> list[dict[str, Any]]:
        """对单张图片进行损伤检测。

        参数
        ------
        image_source : str | Path | np.ndarray
            文件路径或已加载的BGR numpy数组。
        location : str | None
            每条检测记录中嵌入的位置标签。若未指定则使用配置文件中的默认值。

        返回
        ------
        list[dict]
            结构化检测结果（见 perception.utils.output）。
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
        """检测单帧视频帧。

        参数
        ------
        frame : np.ndarray
            BGR帧，通常由cv2.VideoCapture.read()获得。
        location : str | None
            检测结果中的位置标签。

        返回
        ------
        list[dict]
            结构化检测结果。
        """
        return self.detect_image(frame, location=location)

    def detect_video(
        self,
        video_path: str | Path,
        location: str | None = None,
        frame_interval: int = 30,
    ) -> list[dict[str, Any]]:
        """处理视频文件并返回所有检测结果。

        参数
        ------
        video_path : str | Path
            视频文件路径。
        location : str | None
            所有检测结果的统一位置标签。
        frame_interval : int
            每隔多少帧处理一次（默认30帧约1秒）。

        返回
        ------
        list[dict]
            所有帧的检测结果。
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
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _load_model(self):
        """导入ultralytics并加载YOLO模型。

        ultralytics为惰性导入，只有真正需要检测时才加载，
        这样只用预处理或输出工具时不需要GPU环境。
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YOLODetector 需要 'ultralytics' 包。"
                "请使用 pip install ultralytics 安装。"
            ) from exc

        model = YOLO(self._weights)

        return model

    @staticmethod
    def _extract_model_names(model: Any) -> dict[int, str]:
        """从加载的YOLO模型中提取类别名。"""
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
        """运行YOLO推理并将结果转为原始检测字典。"""
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
                class_name = self._display_class_map.get(class_name, class_name)
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
        """从文件路径加载图片，或直接返回ndarray。"""
        if isinstance(source, np.ndarray):
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            # 兼容Windows下cv2.imread无法读取中文路径的情况。
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                if data.size > 0:
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except OSError:
                img = None
        if img is None:
            raise ValueError(f"cv2.imread failed to load: {path}")
        return img

    def _apply_class_map_sync(self, cfg: dict[str, Any]) -> None:
        """可选：根据训练data.yaml自动校验/替换class_map。"""
        sync_cfg = cfg.get("class_map_sync") or {}
        data_yaml = sync_cfg.get("data_yaml")
        if not data_yaml:
            return

        mode = str(sync_cfg.get("mode", "warn")).lower()
        if mode not in {"warn", "strict", "overwrite"}:
            raise ValueError("class_map_sync.mode 必须是 warn|strict|overwrite 之一")

        try:
            expected = load_names(data_yaml)
        except (FileNotFoundError, ValueError) as exc:
            if mode == "strict":
                raise
            warnings.warn(
                f"class_map_sync 跳过: {exc} | action=warn",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        diffs = compare_class_maps(self._class_map, expected)
        if not diffs:
            return

        msg = "class_map 与训练data.yaml不一致: " + "; ".join(diffs)
        if mode == "overwrite":
            self._class_map = expected
            warnings.warn(msg + " | action=overwrite", RuntimeWarning, stacklevel=2)
            return
        if mode == "strict":
            raise ValueError(msg)

        warnings.warn(msg + " | action=warn", RuntimeWarning, stacklevel=2)
