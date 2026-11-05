"""Images and fracture boxes for the detector.

Differences from the classifier pipeline:

- No letterbox: torchvision's detector resizes each image itself (short side
  -> ``min_size``) and maps its boxes back to the input size.
- Pixels stay in 0..1. The detector normalises with ImageNet statistics
  internally, so normalising here would do it twice.
- Grayscale, repeated to 3 channels, with the same fast draft decode as the
  classifier: identical pixels, and the colour-mode shortcut stays removed.
- Boxes are kept in normalised coordinates (0..1) until the image size is
  known, so the draft decode's reduced size never shifts them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

from fracdet.data.boxes import Box, load_boxes
from fracdet.data.transforms import DraftGrayscale


def boxes_to_pixels(boxes: Sequence[Box], width: int, height: int) -> torch.Tensor:
    """(N, 4) float tensor in pixel x0, y0, x1, y1 (empty -> shape (0, 4))."""
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(
        [[b.x0 * width, b.y0 * height, b.x1 * width, b.y1 * height] for b in boxes],
        dtype=torch.float32,
    )


def hflip(image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the image left-right and its boxes with it."""
    w = image.shape[-1]
    flipped = boxes.clone()
    if len(boxes):
        flipped[:, 0] = w - boxes[:, 2]
        flipped[:, 2] = w - boxes[:, 0]
    return image.flip(-1), flipped


class DetectionDataset(Dataset):
    """Returns ``(image (3,H,W) in 0..1, target dict, index)``.

    ``target`` = ``{"boxes": (N,4) pixels, "labels": (N,) all 1}``; a normal
    image has N = 0, which torchvision treats as "everything is background".
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: str,
        min_side: int,
        augment: dict[str, Any] | None = None,
        train: bool = False,
    ) -> None:
        self.paths = frame["path"].tolist()
        self.ids = frame["image_id"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.boxes = [load_boxes(data_root, i) if lab else [] for i, lab in zip(
            self.ids, self.labels, strict=True)]  # fmt: skip
        self.decode = DraftGrayscale(min_side)
        self.train = train
        a = augment or {}
        self.hflip_p = float(a.get("hflip_p", 0.0)) if train else 0.0
        self.jitter = (
            v2.ColorJitter(brightness=a.get("brightness", 0), contrast=a.get("contrast", 0))
            if train and (a.get("brightness") or a.get("contrast"))
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as img:
            gray = self.decode(img)
            w, h = gray.size
            x = torch.from_numpy(np.asarray(gray, dtype=np.uint8).copy()).float().div(255)
        return x[None].expand(3, -1, -1).contiguous(), boxes_to_pixels(self.boxes[i], w, h)

    def __getitem__(self, i: int):
        x, boxes = self.load(i)
        if self.train:
            if self.hflip_p and random.random() < self.hflip_p:
                x, boxes = hflip(x, boxes)
            if self.jitter is not None:
                x = self.jitter(x).clamp(0, 1)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return x, target, i


def collate(batch):
    """Detection batches are lists: images differ in size."""
    images, targets, idx = zip(*batch, strict=True)
    return list(images), list(targets), list(idx)


class PositivesPlusNegatives(Sampler[int]):
    """Each epoch: every fractured image + a fresh random draw of normal ones.

    ``ratio`` normals per fractured image. Seeded per epoch (``set_epoch``),
    so a run is reproducible but every epoch sees different normals.
    """

    def __init__(self, labels: Sequence[int], ratio: float, seed: int) -> None:
        labels = np.asarray(labels)
        self.pos = np.flatnonzero(labels == 1)
        self.neg = np.flatnonzero(labels == 0)
        self.n_neg = min(len(self.neg), round(ratio * len(self.pos)))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pos) + self.n_neg

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        neg = rng.choice(self.neg, size=self.n_neg, replace=False)
        order = np.concatenate([self.pos, neg])
        rng.shuffle(order)
        return iter(order.tolist())
