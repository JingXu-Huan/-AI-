"""
main.py

Single entry for image detection: pass one image path, print YOLO JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from perception import YOLODetector
from perception.utils import detection_to_json, draw_detections, save_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect one image and print YOLO JSON.")
    parser.add_argument("image", metavar="IMAGE", help="Path to an input image file")
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
    return detection_to_json(_detect_detections(Path(image_path)))


def _detect_detections(image_path: Path) -> list[dict]:
    detector = YOLODetector()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return detector.detect_image(image_path)


def _output_dir_for(image_path: Path) -> Path:
    """Build the output directory for one input image."""
    return Path("outputs") / image_path.stem


def _output_paths_for(image_path: Path) -> tuple[Path, Path, Path]:
    """Return output dir, json path and annotated image path."""
    output_dir = _output_dir_for(image_path)
    return (
        output_dir,
        output_dir / f"{image_path.stem}.json",
        output_dir / f"{image_path.stem}_annotated.jpg",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = Path(args.image)

    try:
        detections = _detect_detections(image_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}", file=sys.stderr)
        return 1

    json_str = detection_to_json(detections)
    print(json_str)

    output_dir, json_path, viz_path = _output_paths_for(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_str, encoding="utf-8")
    print(f"[INFO] JSON saved: {json_path}", file=sys.stderr)

    source_image = _read_image(image_path)
    if source_image is None:
        print(f"[WARN] Could not read image for visualisation: {image_path}", file=sys.stderr)
        return 0

    annotated = draw_detections(source_image, detections)
    save_image(annotated, viz_path)
    print(f"[INFO] Visualisation saved: {viz_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
