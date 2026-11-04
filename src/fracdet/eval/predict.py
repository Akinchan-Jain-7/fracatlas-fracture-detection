"""Score one split with a saved checkpoint.

Everything needed (architecture, image size, normalisation) comes from the
checkpoint itself, the same path the web app will use in Phase 8. If
evaluation and serving rebuilt preprocessing separately, they could drift
apart silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fracdet.data.dataset import FracAtlasDataset, load_split_frames
from fracdet.data.transforms import build_transforms
from fracdet.train.checkpoint import load_checkpoint
from fracdet.train.engine import predict

PREDICTION_COLUMNS = ["image_id", "label", "region", "view", "hardware", "multiscan", "split_group"]


def score_split(
    ckpt_path: str | Path,
    data_root: str | Path,
    split_csv: str | Path,
    split: str,
    device: str | torch.device | None = None,
    num_workers: int = 4,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Probabilities for every image of ``split``, with its metadata.

    Returns ``(predictions, checkpoint)``; predictions are in split order with
    the columns of ``PREDICTION_COLUMNS`` that exist, plus ``prob``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    channels_last = device.type == "cuda"

    model, ckpt = load_checkpoint(ckpt_path, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    frame = load_split_frames(data_root, split_csv, [split])[split]
    loader = DataLoader(
        FracAtlasDataset(frame, build_transforms(ckpt.get("preprocess", {}), train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    out = predict(model, loader, nn.BCEWithLogitsLoss(), device, amp, channels_last)
    preds = frame.loc[out["index"], [c for c in PREDICTION_COLUMNS if c in frame]].copy()
    preds["prob"] = out["prob"]
    return preds.reset_index(drop=True), ckpt
