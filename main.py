"""
main.py

Entry point for the YOLO-based infrastructure perception layer.

Usage
-----
  python main.py --image path/to/image.jpg --location "A区-3号楼"
  python main.py --video path/to/video.mp4 --location "B区" --frame-interval 15
  python main.py --image path/to/image.jpg --output result.json --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from perception import YOLODetector
from perception.utils import detection_to_json, draw_detections


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Campus infrastructure damage detection (YOLO perception layer)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", metavar="PATH", help="Path to an input image file")
    source.add_argument("--video", metavar="PATH", help="Path to an input video file")

    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config/detection_config.yaml",
        help="Path to detection_config.yaml (default: config/detection_config.yaml)",
    )
    parser.add_argument(
        "--location",
        metavar="NAME",
        default=None,
        help='Location label embedded in every detection (e.g. "A区-3号楼")',
    )
    parser.add_argument(
        "--frame-interval",
        metavar="N",
        type=int,
        default=30,
        help="For video input: process every N-th frame (default: 30)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSON results to this file (optional)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show annotated image/first-frame in a window (requires display)",
    )
    parser.add_argument(
        "--save-viz",
        metavar="PATH",
        default=None,
        help="Save annotated image to this path instead of displaying it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # Load detector
    # -----------------------------------------------------------------
    try:
        detector = YOLODetector(config_path=args.config)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------
    # Run detection
    # -----------------------------------------------------------------
    detections: list[dict]
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
            return 1
        detections = detector.detect_image(image_path, location=args.location)
        source_image = cv2.imread(str(image_path))
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
            return 1
        detections = detector.detect_video(
            video_path,
            location=args.location,
            frame_interval=args.frame_interval,
        )
        source_image = None  # visualisation not supported for video in CLI

    # -----------------------------------------------------------------
    # Output JSON
    # -----------------------------------------------------------------
    json_str = detection_to_json(detections)
    print(json_str)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"\n[INFO] Results written to {out_path}", file=sys.stderr)

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------
    if (args.visualize or args.save_viz) and source_image is not None:
        annotated = draw_detections(source_image, detections)
        if args.save_viz:
            cv2.imwrite(args.save_viz, annotated)
            print(f"[INFO] Annotated image saved to {args.save_viz}", file=sys.stderr)
        if args.visualize:
            cv2.imshow("Detections", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
