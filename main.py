"""统一入口：接收一个路径，支持图片与视频流检测并输出JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception import YOLODetector
from perception.utils import detection_to_json, draw_detections, save_image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect image/video and print YOLO JSON.")
    parser.add_argument("source", metavar="SOURCE", help="Image path, video path, or stream URL")
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config/detection_config.yaml",
        help="Path to detection config (default: config/detection_config.yaml)",
    )
    parser.add_argument(
        "--location",
        metavar="NAME",
        default=None,
        help='Location label in output (e.g. "A区-3号楼")',
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Run inference every N frames for video/stream (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process for video/stream (prevent infinite streams)",
    )
    parser.add_argument(
        "--mjpeg",
        action="store_true",
        help="Output MJPEG stream to stdout (for live browser rendering) and suppress standard output",
    )
    return parser.parse_args(argv)


def _read_image(path: Path) -> np.ndarray | None:
    """Read image data with a fallback for Chinese/Unicode Windows paths."""
    image = cv2.imread(str(path))
    if image is not None:
        return image

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_image_json(image_path: str | Path) -> str:
    """Detect one image and return the YOLO JSON string."""
    detector = YOLODetector()
    return detection_to_json(_detect_detections(Path(image_path), detector=detector))


def _detect_detections(image_path: Path, detector: YOLODetector, location: str | None = None) -> list[dict]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return detector.detect_image(image_path, location=location)


def _output_dir_for(image_path: Path) -> Path:
    """输出目录名与输入文件名同名。"""
    return Path("outputs") / image_path.stem


def _output_paths_for(image_path: Path) -> tuple[Path, Path, Path]:
    """Return output dir, json path and annotated image path."""
    output_dir = _output_dir_for(image_path)
    return (
        output_dir,
        output_dir / f"{image_path.stem}.json",
        output_dir / f"{image_path.stem}_annotated.jpg",
    )


def _is_stream_source(raw_source: str) -> bool:
    lowered = raw_source.lower()
    return lowered.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def _is_video_source(path: Path, raw_source: str) -> bool:
    if _is_stream_source(raw_source):
        return True
    if raw_source.isdigit():
        return True
    return path.suffix.lower() in _VIDEO_SUFFIXES


def _video_output_paths(source: str) -> tuple[Path, Path, Path]:
    source_path = Path(source)
    if source.isdigit():
        name = f"camera_{source}"
    elif _is_stream_source(source):
        name = "stream"
    else:
        name = source_path.stem
    output_dir = Path("outputs") / name
    return (
        output_dir,
        output_dir / f"{name}.json",
        output_dir / f"{name}_annotated.mp4",
    )


def _open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _detect_video_mjpeg(
    detector: YOLODetector,
    source: str,
    location: str | None,
    frame_interval: int,
) -> None:
    output_dir, json_path, _ = _video_output_paths("stream")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_capture(source)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    records: list[dict[str, Any]] = []
    frame_idx = 0
    last_detections: list[dict[str, Any]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                last_detections = detector.detect_frame(frame, location=location)
                timestamp_ms = round((frame_idx / fps) * 1000.0, 1)
                records.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "detections": last_detections,
                    }
                )
                
            annotated = draw_detections(frame, last_detections)
            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                sys.stdout.buffer.write(b"--frame\r\n")
                sys.stdout.buffer.write(b"Content-Type: image/jpeg\r\n\r\n")
                sys.stdout.buffer.write(buffer.tobytes())
                sys.stdout.buffer.write(b"\r\n")
                sys.stdout.flush()

            # Incrementally write JSON to avoid data loss when killed
            if frame_idx % 10 == 0:
                json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_video_json(
    detector: YOLODetector,
    source: str,
    location: str | None,
    frame_interval: int,
    max_frames: int | None = None,
) -> tuple[str, Path]:
    if frame_interval <= 0:
        raise ValueError("--frame-interval must be a positive integer")

    output_dir, json_path, viz_path = _video_output_paths(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/stream: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer: cv2.VideoWriter | None = None
    if width > 0 and height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(viz_path), fourcc, fps, (width, height))

    records: list[dict[str, Any]] = []
    frame_idx = 0
    last_detections: list[dict[str, Any]] = []

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                last_detections = detector.detect_frame(frame, location=location)
                timestamp_ms = round((frame_idx / fps) * 1000.0, 1)
                records.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "detections": last_detections,
                    }
                )

            if writer is not None:
                writer.write(draw_detections(frame, last_detections))

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    json_str = json.dumps(records, ensure_ascii=False, indent=2)
    json_path.write_text(json_str, encoding="utf-8")
    if writer is not None:
        print(f"[INFO] Visualisation saved: {viz_path}", file=sys.stderr)
    print(f"[INFO] JSON saved: {json_path}", file=sys.stderr)
    return json_str, json_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = str(args.source)
    source_path = Path(source)

    try:
        detector = YOLODetector(config_path=args.config)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}", file=sys.stderr)
        return 1

    if args.mjpeg:
        _detect_video_mjpeg(
            detector=detector,
            source=source,
            location=args.location,
            frame_interval=args.frame_interval
        )
        return 0

    if _is_video_source(source_path, source):
        if source_path.suffix and not _is_stream_source(source) and not source.isdigit() and not source_path.exists():
            print(f"[ERROR] Video not found: {source_path}", file=sys.stderr)
            return 1
        try:
            json_str, _ = _detect_video_json(
                detector=detector,
                source=source,
                location=args.location,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(json_str)
        return 0

    if not source_path.exists():
        print(f"[ERROR] Image not found: {source_path}", file=sys.stderr)
        return 1

    try:
        detections = _detect_detections(source_path, detector=detector, location=args.location)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    json_str = detection_to_json(detections)
    print(json_str)

    output_dir, json_path, viz_path = _output_paths_for(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_str, encoding="utf-8")

    source_image = _read_image(source_path)
    if source_image is None:
        print(f"[WARN] Could not read image for visualisation: {source_path}", file=sys.stderr)
        return 0

    annotated = draw_detections(source_image, detections)
    save_image(annotated, viz_path)
    print(f"[INFO] JSON saved: {json_path}", file=sys.stderr)
    print(f"[INFO] Visualisation saved: {viz_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
