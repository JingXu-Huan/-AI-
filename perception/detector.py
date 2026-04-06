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

增强功能：
  - 多种预处理模式
  - 批量图像处理
  - 视频流实时检测
  - 置信度过滤和NMS调优
  - 检测结果统计

用法示例
--------
>>> from perception import YOLODetector
>>> detector = YOLODetector("config/dection_config.yaml")
>>> detections = detector.detect_image("path/to/image.jpg", location="A区-3号楼")
>>> print(detections)

# 批量检测
>>> batch_results = detector.detect_batch(["img1.jpg", "img2.jpg"], locations=["A区", "B区"])

# 视频流检测
>>> for frame_results in detector.detect_video_stream("rtsp://camera"):
...     print(frame_results)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import yaml

from .class_map_sync import compare_class_maps, load_names
from .preprocessor import ImagePreprocessor
from .utils.output import format_detections


class DetectionStats:
    """检测统计信息"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_images = 0
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.avg_confidence = 0.0
        self.total_processing_time = 0.0
        self.high_severity_count = 0
        self.medium_severity_count = 0
        self.low_severity_count = 0
    
    def add(self, detections: list[dict], processing_time: float):
        self.total_images += 1
        self.total_detections += len(detections)
        self.total_processing_time += processing_time
        
        for det in detections:
            cls = det.get("type", "unknown")
            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
            
            sev = det.get("severity", "low")
            if sev == "high":
                self.high_severity_count += 1
            elif sev == "medium":
                self.medium_severity_count += 1
            else:
                self.low_severity_count += 1
            
            self.avg_confidence += det.get("confidence", 0)
        
        if self.total_detections > 0:
            self.avg_confidence /= self.total_detections
    
    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "total_detections": self.total_detections,
            "class_counts": self.class_counts,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_processing_time_ms": round(self.total_processing_time / max(1, self.total_images), 2),
            "severity": {
                "high": self.high_severity_count,
                "medium": self.medium_severity_count,
                "low": self.low_severity_count,
            }
        }


class YOLODetector:
    """使用YOLO模型检测图片中的基础设施损伤。

    参数
    ------
    config_path : str | Path
        ``detection_config.yaml`` 的路径。
    preprocessor_mode : str
        预处理模式: "default", "high_contrast", "edge_focused"
    """

    def __init__(
        self, 
        config_path: str | Path = "config/detection_config.yaml",
        preprocessor_mode: str = "default",
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = Path(config_path)
        if not config_path.is_absolute():
            project_candidate = project_root / config_path
            if project_candidate.exists():
                config_path = project_candidate

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        model_cfg = cfg.get("model", {})
        raw_weights = str(model_cfg.get("weights", "yolov8n.pt"))
        weights_path = Path(raw_weights)
        if not weights_path.is_absolute():
            from_config = config_path.parent / weights_path
            from_project = project_root / weights_path
            if from_config.exists():
                weights_path = from_config
            elif from_project.exists():
                weights_path = from_project

        self._weights: str = str(weights_path) if weights_path.exists() else raw_weights
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
        
        # 兼容旧配置并支持新模式
        self._preprocessor = ImagePreprocessor(
            denoise=bool(preproc_cfg.get("denoise", True)),
            clahe=bool(preproc_cfg.get("clahe", True)),
            edge_enhancement=bool(preproc_cfg.get("edge_enhancement", False)),
            sharpen=bool(preproc_cfg.get("sharpen", False)),
            mode=preprocessor_mode,
        )

        self._model = self._load_model()
        self._model_names = self._extract_model_names(self._model)
        
        # 统计信息
        self._stats = DetectionStats()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    @property
    def stats(self) -> DetectionStats:
        """获取检测统计信息"""
        return self._stats

    def reset_stats(self):
        """重置统计信息"""
        self._stats.reset()

    def detect_image(
        self,
        image_source: str | Path | np.ndarray,
        location: str | None = None,
        return_stats: bool = False,
    ) -> list[dict[str, Any]]:
        """对单张图片进行损伤检测。

        参数
        ------
        image_source : str | Path | np.ndarray
            文件路径或已加载的BGR numpy数组。
        location : str | None
            每条检测记录中嵌入的位置标签。若未指定则使用配置文件中的默认值。
        return_stats : bool
            是否返回处理统计信息

        返回
        ------
        list[dict]
            结构化检测结果（见 perception.utils.output）。
        """
        start_time = time.time()
        
        image = self._load_image(image_source)
        preprocessed = self._preprocessor.process(image)
        raw = self._run_inference(preprocessed)
        
        detections = format_detections(
            raw,
            location=location or self._default_location,
            high_threshold=self._high_threshold,
            medium_threshold=self._medium_threshold,
        )
        
        processing_time = (time.time() - start_time) * 1000  # ms
        self._stats.add(detections, processing_time)
        
        if return_stats:
            return detections, {
                "processing_time_ms": round(processing_time, 2),
                "image_shape": image.shape,
                "preprocessing_mode": self._preprocessor.mode,
            }
        
        return detections

    def detect_batch(
        self,
        image_sources: list[str | Path | np.ndarray],
        locations: list[str | None] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """批量检测多张图像。

        参数
        ------
        image_sources : list
            图像路径或数组的列表。
        locations : list[str | None] | None
            与图像列表对应的位置标签列表。若为None则使用默认值。

        返回
        ------
        list[list[dict]]
            每张图像的检测结果列表。
        """
        results = []
        locs = locations if locations else [None] * len(image_sources)
        
        for source, loc in zip(image_sources, locs):
            results.append(self.detect_image(source, location=loc))
        
        return results

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
        max_frames: int | None = None,
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
        max_frames : int | None
            最大处理帧数，用于防止处理过长视频。

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

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        all_detections: list[dict[str, Any]] = []
        frame_idx = 0

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    dets = self.detect_frame(frame, location=location)
                    timestamp_ms = round((frame_idx / fps) * 1000.0, 1)
                    all_detections.append({
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "detections": dets,
                    })
                frame_idx += 1
        finally:
            cap.release()

        return all_detections

    def detect_video_stream(
        self,
        source: str | int,
        location: str | None = None,
        frame_interval: int = 1,
        max_frames: int | None = None,
        callback: Callable[[list[dict], int], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """实时视频流检测（生成器模式）。

        参数
        ------
        source : str | int
            视频源：RTSP URL, 文件路径, 或摄像头索引。
        location : str | None
            位置标签。
        frame_interval : int
            每隔多少帧检测一次。
        max_frames : int | None
            最大处理帧数。
        callback : Callable | None
            每帧处理后的回调函数。

        生成
        ------
        list[dict]
            每帧的检测结果。
        """
        cap = cv2.VideoCapture(source) if isinstance(source, str) else cv2.VideoCapture(int(source))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        
        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    detections = self.detect_frame(frame, location=location)
                    
                    if callback:
                        callback(detections, frame_idx)
                    
                    yield {
                        "frame_index": frame_idx,
                        "timestamp_ms": round((frame_idx / fps) * 1000.0, 1),
                        "detections": detections,
                    }
                
                frame_idx += 1
        finally:
            cap.release()

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        severity: list[str] | None = None,
        classes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """过滤检测结果。

        参数
        ------
        detections : list[dict]
            检测结果列表。
        min_confidence : float | None
            最小置信度阈值。
        max_confidence : float | None
            最大置信度阈值。
        severity : list[str] | None
            保留的严重程度级别，如 ["high", "medium"]。
        classes : list[str] | None
            保留的类别列表。

        返回
        ------
        list[dict]
            过滤后的检测结果。
        """
        filtered = []
        
        for det in detections:
            # 置信度过滤
            conf = det.get("confidence", 0)
            if min_confidence is not None and conf < min_confidence:
                continue
            if max_confidence is not None and conf > max_confidence:
                continue
            
            # 严重程度过滤
            if severity and det.get("severity") not in severity:
                continue
            
            # 类别过滤
            if classes and det.get("type") not in classes:
                continue
            
            filtered.append(det)
        
        return filtered

    def export_results(
        self,
        detections: list[dict[str, Any]],
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """导出检测结果。

        参数
        ------
        detections : list[dict]
            检测结果列表。
        output_path : str | Path
            输出文件路径。
        format : str
            输出格式: "json", "csv", "txt"
        """
        output_path = Path(output_path)
        
        if format == "json":
            output_path.write_text(
                json.dumps(detections, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        elif format == "csv":
            import csv
            if not detections:
                return
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=detections[0].keys())
                writer.writeheader()
                writer.writerows(detections)
        elif format == "txt":
            lines = [json.dumps(d, ensure_ascii=False) for d in detections]
            output_path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _load_model(self):
        """导入ultralytics并加载YOLO模型。"""
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