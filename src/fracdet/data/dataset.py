"""PyTorch Dataset over the committed split.

``load_split_frames`` joins image metadata with ``splits/<name>.csv`` and
returns one DataFrame per requested split. The training script asks only for
train and val: the test set is not even loaded until Phase 3 evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fracdet.data.metadata import load_metadata
from fracdet.data.schema import ID_COLUMN

VALID_SPLITS = ("train", "val", "test")


def load_split_frames(
    root: str | Path, split_csv: str | Path, splits: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Metadata rows for each requested split, in a stable order."""
    unknown = set(splits) - set(VALID_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits {unknown}; valid: {VALID_SPLITS}")
    meta = load_metadata(root)
    split = pd.read_csv(split_csv, keep_default_na=False)
    # split_group (similar images / likely same patient) travels with each row so
    # evaluation can resample whole groups (Phase 3 cluster bootstrap).
    split = split[[c for c in (ID_COLUMN, "split", "split_group") if c in split]]
    df = meta.merge(split, on=ID_COLUMN, how="inner", validate="one_to_one")
    if len(df) != len(meta):
        raise ValueError(f"Split file covers {len(df)} of {len(meta)} images; wrong dataset?")
    return {s: df[df["split"] == s].reset_index(drop=True) for s in splits}


class FracAtlasDataset(Dataset):
    """Returns ``(image_tensor, label_float, index)``.

    The index lets evaluation map each prediction back to its row (image id,
    region, hardware flag) for per-group analysis later.
    """

    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].astype(float).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        with Image.open(self.paths[i]) as img:
            x = self.transform(img)
        return x, torch.tensor(self.labels[i], dtype=torch.float32), i
