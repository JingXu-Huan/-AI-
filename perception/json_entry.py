"""
perception/json_entry.py

Minimal image detection entrypoint that returns JSON strings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from perception import YOLODetector
from perception.utils import detection_to_json


def detect_image_to_json(
    image_path: str | Path,
    config_path: str | Path = "config/detection_config.yaml",
    location: str | None = None,
) -> str:
    """Run detection on one image and return a JSON string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    detector = YOLODetector(config_path=config_path)
    detections = detector.detect_image(path, location=location)
    return detection_to_json(detections)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect one image and output JSON string")
    parser.add_argument("image", metavar="IMAGE", help="Input image path")
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
        help='Location label embedded in each detection (e.g. "A区-3号楼")',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        json_str = detect_image_to_json(
            image_path=args.image,
            config_path=args.config,
            location=args.location,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(json_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

