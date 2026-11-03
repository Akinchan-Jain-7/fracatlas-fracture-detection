from pathlib import Path

import pytest
import yaml

from fracdet.config import apply_overrides, deep_merge, load_config, save_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_deep_merge_merges_nested_and_replaces_leaves():
    base = {"train": {"lr": 0.1, "epochs": 10}, "tags": ["a"]}
    override = {"train": {"lr": 0.01}, "tags": ["b"]}
    merged = deep_merge(base, override)
    assert merged == {"train": {"lr": 0.01, "epochs": 10}, "tags": ["b"]}
    assert base["train"]["lr"] == 0.1  # input not mutated


def test_base_inheritance_resolves_relative_to_file(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "base.yaml", {"seed": 1, "train": {"lr": 0.1, "epochs": 5}})
    child = _write(tmp_path / "sub" / "exp.yaml", {"_base_": "../base.yaml", "train": {"lr": 0.5}})
    cfg = load_config(child)
    assert cfg == {"seed": 1, "train": {"lr": 0.5, "epochs": 5}}
    assert "_base_" not in cfg


def test_circular_base_raises(tmp_path):
    _write(tmp_path / "a.yaml", {"_base_": "b.yaml"})
    _write(tmp_path / "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("override", "key_path", "expected"),
    [
        ("train.lr=3e-4", ("train", "lr"), 3e-4),  # PyYAML alone returns the str "3e-4"
        ("train.lr=3.0e-4", ("train", "lr"), 3e-4),
        ("train.wd=1E5", ("train", "wd"), 1e5),
        ("model.pretrained=false", ("model", "pretrained"), False),
        ("model.name=resnet50", ("model", "name"), "resnet50"),
        ("data.sizes=[224, 384]", ("data", "sizes"), [224, 384]),
        ("new.deep.key=7", ("new", "deep", "key"), 7),
    ],
)
def test_overrides_parse_yaml_types(override, key_path, expected):
    cfg = apply_overrides({"train": {"lr": 0.1}}, [override])
    node = cfg
    for k in key_path:
        node = node[k]
    assert node == expected


def test_override_without_equals_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.lr"])


def test_override_into_non_mapping_raises():
    with pytest.raises(KeyError):
        apply_overrides({"seed": 42}, ["seed.value=1"])


def test_save_and_reload_roundtrip(tmp_path):
    cfg = {"seed": 42, "train": {"lr": 0.001}}
    save_config(cfg, tmp_path / "run" / "config.yaml")
    assert load_config(tmp_path / "run" / "config.yaml") == cfg


def test_repo_base_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert cfg["seed"] == 42
    assert cfg["data"]["md5"] and len(cfg["data"]["md5"]) == 32
