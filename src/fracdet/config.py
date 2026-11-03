"""YAML config loading with inheritance and command-line overrides.

Why not just ``yaml.safe_load``?
  1. Inheritance: an experiment file says ``_base_: base.yaml`` and lists only
     what it changes. Shared settings live in one place.
  2. Overrides: ``--set train.lr=3e-4`` changes one value for a quick test
     without editing (or forgetting to revert) a file.
  3. The *resolved* config gets saved with every run, so each reported number
     traces back to the exact settings that produced it.

Libraries like Hydra/OmegaConf do this and more; this ~80-line version keeps the
dependency list short and every line explainable.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

BASE_KEY = "_base_"


def deep_merge(base: Config, override: Config) -> Config:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dicts are merged key by key; any other value (list, number, str)
    in ``override`` replaces the base value entirely.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_bases(path: Path, seen: set[Path]) -> Config:
    path = path.resolve()
    if path in seen:  # a.yaml -> b.yaml -> a.yaml would recurse forever
        raise ValueError(f"Circular _base_ chain detected at {path}")
    seen.add(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(data).__name__}")

    base_ref = data.pop(BASE_KEY, None)
    if base_ref is None:
        return data
    # _base_ paths are relative to the file that references them, not to the CWD,
    # so configs work no matter where the script is launched from.
    base_cfg = _load_with_bases(path.parent / base_ref, seen)
    return deep_merge(base_cfg, data)


def _set_dotted(cfg: Config, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise KeyError(f"Cannot set '{dotted_key}': '{key}' is not a mapping")
        node = child
    node[keys[-1]] = value


# Scientific notation without a dot, e.g. "3e-4" or "1E5".
_SCI_NOTATION = re.compile(r"^[+-]?\d+[eE][+-]?\d+$")


def parse_value(raw: str) -> Any:
    """Parse an override value as YAML, fixing one PyYAML trap.

    PyYAML implements YAML 1.1, where ``3e-4`` is NOT a float (it needs a dot:
    ``3.0e-4``) and silently stays the string ``"3e-4"``. A learning rate that is
    a string would crash the optimizer at best, or be compared wrongly at worst.
    The same trap applies inside config files: write ``lr: 3.0e-4`` there.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _SCI_NOTATION.match(value.strip()):
        return float(value)
    return value


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``key.sub=value`` overrides. Values are parsed as YAML.

    So ``lr=3e-4`` becomes a float, ``pretrained=false`` a bool,
    ``sizes=[224,384]`` a list, and ``name=resnet50`` stays a string.
    """
    result = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override has an empty key: '{item}'")
        _set_dotted(result, key, parse_value(raw))
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a config file, resolve its ``_base_`` chain, then apply overrides."""
    cfg = _load_with_bases(Path(path), seen=set())
    return apply_overrides(cfg, overrides)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a resolved config next to a run's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
