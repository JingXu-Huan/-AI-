"""Utilities to generate/validate class_map from a YOLO training data YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def resolve_data_yaml(data_yaml_or_dir: str | Path) -> Path:
    """Resolve a data yaml path from a file or a dataset directory."""
    candidate = Path(data_yaml_or_dir)
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        preferred = ("data.yaml", "dataset.yaml", "lumian.yaml")
        for name in preferred:
            p = candidate / name
            if p.is_file():
                return p

        yamls = sorted(candidate.glob("*.y*ml"))
        if len(yamls) == 1:
            return yamls[0]
        if len(yamls) > 1:
            raise FileNotFoundError(
                f"Multiple yaml files found in {candidate}; pass an explicit yaml path."
            )

    raise FileNotFoundError(f"Training data yaml not found: {candidate}")


def load_names(data_yaml_or_dir: str | Path) -> dict[int, str]:
    """Load class names from YOLO data yaml and normalize to {int: str}."""
    data_yaml = resolve_data_yaml(data_yaml_or_dir)
    with data_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    names = data.get("names")
    if isinstance(names, list):
        class_map = {int(i): str(v) for i, v in enumerate(names)}
    elif isinstance(names, dict):
        class_map = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError(f"Invalid or missing 'names' in data yaml: {data_yaml}")

    nc = data.get("nc")
    if nc is not None and int(nc) != len(class_map):
        raise ValueError(
            f"'nc' ({nc}) does not match number of names ({len(class_map)}) in {data_yaml}"
        )

    return dict(sorted(class_map.items(), key=lambda item: item[0]))


def load_config_class_map(config_path: str | Path) -> dict[int, str]:
    """Load class_map from detector config and normalize keys to ints."""
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    class_map = cfg.get("class_map") or {}
    return {int(k): str(v) for k, v in class_map.items()}


def compare_class_maps(current: dict[int, str], expected: dict[int, str]) -> list[str]:
    """Return human-readable diffs between current and expected class maps."""
    diffs: list[str] = []
    all_keys = sorted(set(current) | set(expected))
    for idx in all_keys:
        cur = current.get(idx)
        exp = expected.get(idx)
        if cur != exp:
            diffs.append(f"index {idx}: config={cur!r}, data_yaml={exp!r}")
    return diffs


def write_class_map(config_path: str | Path, new_class_map: dict[int, str]) -> None:
    """Write normalized class_map back into detection config yaml."""
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg["class_map"] = {int(k): str(v) for k, v in sorted(new_class_map.items())}

    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)


def sync_class_map(
    config_path: str | Path,
    data_yaml_or_dir: str | Path,
    write: bool = False,
) -> tuple[bool, list[str], dict[int, str]]:
    """Validate class_map against training data yaml, optionally writing updates."""
    current = load_config_class_map(config_path)
    expected = load_names(data_yaml_or_dir)
    diffs = compare_class_maps(current, expected)

    if write:
        write_class_map(config_path, expected)

    return len(diffs) == 0, diffs, expected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/validate class_map from YOLO training data yaml"
    )
    parser.add_argument(
        "--config",
        default="config/detection_config.yaml",
        help="Path to detection_config.yaml",
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        help="Path to data.yaml, or a dataset directory that contains it",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite class_map in config with names from data yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        is_match, diffs, expected = sync_class_map(
            config_path=args.config,
            data_yaml_or_dir=args.data_yaml,
            write=args.write,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("[INFO] expected class_map from data yaml:")
    for idx, name in expected.items():
        print(f"  {idx}: {name}")

    if diffs:
        print("[WARN] class_map mismatch detected:")
        for line in diffs:
            print(f"  - {line}")
    else:
        print("[INFO] class_map matches data yaml")

    if args.write:
        print(f"[INFO] class_map written to {args.config}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

