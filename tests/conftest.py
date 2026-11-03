"""Shared test fixtures: a miniature, realistic FracAtlas on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fracdet.data.schema import BODY_PARTS, VIEW_COLUMNS


def write_xray(path: Path, seed: int, size: tuple[int, int] = (96, 128)) -> None:
    """A random but smooth grayscale JPEG, so perceptual hashes are meaningful."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(6, 8), dtype=np.uint8)
    img = Image.fromarray(coarse, mode="L").resize(size, Image.Resampling.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def make_fake_dataset(root: Path, n: int = 40, frac_rate: float = 0.25, seed: int = 0) -> Path:
    """FracAtlas-shaped folder with real JPEGs, CSV flags and YOLO files."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.random() < frac_rate)
        image_id = f"IMG{i:07d}.jpg"
        folder = "Fractured" if label else "Non_fractured"
        write_xray(root / "images" / folder / image_id, seed=1000 + i)
        row = {"image_id": image_id, "fractured": label, "mixed": 0, "hardware": 0, "multiscan": 0}
        row.update({p: 0 for p in BODY_PARTS})
        row[BODY_PARTS[i % len(BODY_PARTS)]] = 1
        row.update({v: 0 for v in VIEW_COLUMNS})
        row[VIEW_COLUMNS[i % 2]] = 1
        row["fracture_count"] = label
        rows.append(row)
        yolo = root / "Annotations" / "YOLO" / f"IMG{i:07d}.txt"
        yolo.parent.mkdir(parents=True, exist_ok=True)
        yolo.write_text("0 0.5 0.5 0.2 0.2\n" if label else "")
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    return root


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    return make_fake_dataset(tmp_path / "FracAtlas")
