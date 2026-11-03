"""Locating the dataset across machines (laptop, Kaggle, Colab, Docker).

The same code must find the data whether it lives in ``data/raw`` locally or in
``/kaggle/input/<your-dataset>/`` on Kaggle, without editing source files.
Resolution order:

1. ``FRACDET_DATA_DIR`` environment variable (set per platform)
2. ``paths.data_dir`` in the config
"""

from __future__ import annotations

import os
from pathlib import Path

from fracdet.config import Config

DATA_DIR_ENV = "FRACDET_DATA_DIR"
# The one file every FracAtlas copy has at its root; used as the landmark.
MARKER_FILE = "dataset.csv"


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def resolve_data_dir(cfg: Config) -> Path:
    """Return the configured base data folder (may not contain data yet)."""
    env_value = os.environ.get(DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return Path(cfg["paths"]["data_dir"]).expanduser()


def find_dataset_root(base: str | Path, max_depth: int = 4) -> Path | None:
    """Find the folder containing ``dataset.csv`` at or below ``base``.

    Needed because the extracted layout differs by source: unzipping locally
    gives ``base/FracAtlas/dataset.csv``, while Kaggle may add another folder
    level when it unpacks an uploaded zip. Searching breadth-first and
    depth-limited returns the shallowest match quickly.
    """
    base = Path(base)
    if not base.is_dir():
        return None
    frontier = [base]
    for _ in range(max_depth + 1):
        next_frontier: list[Path] = []
        for folder in sorted(frontier):
            if (folder / MARKER_FILE).is_file():
                return folder
            next_frontier.extend(p for p in folder.iterdir() if p.is_dir())
        frontier = next_frontier
    return None
