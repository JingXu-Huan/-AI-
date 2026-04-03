"""
perception/image_cli.py

Convenient CLI entry for single-image inspection.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from perception import YOLODetector
from perception.utils import detection_to_json, draw_detections, save_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect defects from one image and save outputs.")
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
        help='Location label in output (e.g. "A区-3号楼")',
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default="outputs",
        help="Directory for saved files (default: outputs)",
    )
    parser.add_argument(
        "--name",
        metavar="BASENAME",
        default=None,
        help="Base filename for outputs (default: image filename)",
    )
    parser.add_argument(
        "--font-path",
        metavar="PATH",
        default=None,
        help="Optional font file for Chinese labels",
    )
    parser.add_argument("--no-json", action="store_true", help="Do not save JSON result file")
    parser.add_argument("--no-viz", action="store_true", help="Do not save annotated image")
    parser.add_argument("--show", action="store_true", help="Show annotated image in a window")
    return parser.parse_args(argv)


def _read_image(path: Path) -> np.ndarray | None:
    """Read image with a fallback for non-ASCII Windows paths."""
    img = cv2.imread(str(path))
    if img is not None:
        return img

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        detector = YOLODetector(config_path=args.config)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}", file=sys.stderr)
        return 1

    detections = detector.detect_image(image_path, location=args.location)
    json_str = detection_to_json(detections)
    print(json_str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or image_path.stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_json:
        json_path = out_dir / f"{base}_{stamp}.json"
        json_path.write_text(json_str, encoding="utf-8")
        print(f"[INFO] JSON saved: {json_path}", file=sys.stderr)

    need_annotated = (not args.no_viz) or args.show
    if need_annotated:
        source_image = _read_image(image_path)
        if source_image is None:
            print(f"[WARN] Could not read image for visualisation: {image_path}", file=sys.stderr)
        else:
            annotated = draw_detections(source_image, detections, font_path=args.font_path)
            if not args.no_viz:
                viz_path = out_dir / f"{base}_{stamp}_annotated.jpg"
                save_image(annotated, viz_path)
                print(f"[INFO] Visualisation saved: {viz_path}", file=sys.stderr)
            if args.show:
                cv2.imshow("Detections", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

