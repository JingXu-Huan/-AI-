from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from perception.class_map_sync import (
    compare_class_maps,
    load_names,
    resolve_data_yaml,
    sync_class_map,
)


def test_resolve_data_yaml_from_dir(tmp_path: Path):
    data = tmp_path / "data.yaml"
    data.write_text("names: [a, b]\n", encoding="utf-8")
    assert resolve_data_yaml(tmp_path) == data


def test_load_names_with_list_and_nc(tmp_path: Path):
    data = tmp_path / "data.yaml"
    data.write_text("nc: 3\nnames: [Crack, Manhole, Net]\n", encoding="utf-8")
    assert load_names(data) == {0: "Crack", 1: "Manhole", 2: "Net"}


def test_load_names_nc_mismatch_raises(tmp_path: Path):
    data = tmp_path / "data.yaml"
    data.write_text("nc: 2\nnames: [Crack, Manhole, Net]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_names(data)


def test_compare_class_maps_reports_diff():
    diffs = compare_class_maps({0: "A", 1: "B"}, {0: "A", 1: "X", 2: "C"})
    assert len(diffs) == 2
    assert "index 1" in diffs[0]
    assert "index 2" in diffs[1]


def test_sync_class_map_write_updates_config(tmp_path: Path):
    config = tmp_path / "detection_config.yaml"
    yaml.safe_dump({"class_map": {0: "old"}}, config.open("w", encoding="utf-8"), sort_keys=False)

    data = tmp_path / "data.yaml"
    data.write_text("names: [Crack, Manhole]\n", encoding="utf-8")

    is_match, diffs, expected = sync_class_map(config, data, write=True)

    assert not is_match
    assert diffs
    assert expected == {0: "Crack", 1: "Manhole"}

    cfg_after = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert cfg_after["class_map"] == {0: "Crack", 1: "Manhole"}

