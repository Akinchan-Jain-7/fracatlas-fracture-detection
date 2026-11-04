"""Ways to handle the class imbalance (about 17.6% fractured in train).

Phase 2 used plain BCE: every image counts the same, so the 82% normals
dominate the gradient. Phase 4 compares three standard remedies against it.
All three change *training only*; evaluation is identical for every variant.

1. ``pos_weight`` (weighted BCE): multiply the loss of every fractured image
   by w. With w = n_normal / n_fractured (about 4.7), both classes contribute
   equally to the total loss. Cheapest option, no change to the data.

2. Balanced sampler: draw training images so each batch is about half
   fractured. Each fractured image is seen ~2.8x per epoch, each normal ~0.6x.
   Same effect on the class balance as weighting, but through repetition,
   which also repeats augmentation, and risks overfitting the 483 positives.

3. Focal loss (Lin et al., 2017): ``FL = -(1 - p_t)^gamma * log(p_t)``. It
   down-weights *easy* examples of either class (p_t close to 1) and focuses
   on hard ones. Our Phase 3 finding fits it: the hardest fractures scored
   near 0. With ``gamma = 0`` it is exactly BCE.

What to expect, honestly: weighting and sampling mostly *shift* the scores
upward for fractures (calibration), which a tuned threshold already
compensates for. Gains in *ranking* (PR-AUC, specificity at a fixed recall)
are usually small. Measuring whether there is any, beyond seed-to-seed noise,
is the point of this phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

    ``alpha`` (optional) weights positives by alpha and negatives by 1 - alpha;
    ``None`` leaves classes unweighted so only the focusing effect is tested.
    Computed from ``binary_cross_entropy_with_logits`` for numerical stability.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)  # probability assigned to the true class
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            loss = loss * (self.alpha * target + (1 - self.alpha) * (1 - target))
        return loss.mean()


def balance_weight(labels) -> float:
    """n_normal / n_fractured: the pos_weight that equalises both classes' total loss."""
    y = np.asarray(labels).astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive labels: cannot balance")
    return (len(y) - n_pos) / n_pos


def build_criterion(loss_cfg: dict[str, Any] | None, train_labels) -> tuple[nn.Module, dict]:
    """Loss module from ``train.loss`` config, plus a record of what was used.

    ``{name: bce}`` (default), ``{name: bce, pos_weight: auto | <float>}``,
    ``{name: focal, gamma: 2.0, alpha: null}``.
    """
    cfg = {"name": "bce", **(loss_cfg or {})}
    name = cfg["name"]
    if name == "bce":
        pw = cfg.get("pos_weight")
        if pw is None:
            return nn.BCEWithLogitsLoss(), {"name": "bce", "pos_weight": None}
        value = balance_weight(train_labels) if pw == "auto" else float(pw)
        return (
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor(value)),
            {"name": "bce", "pos_weight": round(value, 4)},
        )
    if name == "focal":
        gamma, alpha = float(cfg.get("gamma", 2.0)), cfg.get("alpha")
        return FocalLoss(gamma, alpha), {"name": "focal", "gamma": gamma, "alpha": alpha}
    raise ValueError(f"Unknown loss '{name}' (expected bce or focal)")


def balanced_sampler(labels, seed: int) -> WeightedRandomSampler:
    """Sample each class with equal total probability, with replacement.

    ``num_samples = len(labels)`` keeps one epoch the same number of steps as
    without the sampler, so schedules and early stopping stay comparable.
    """
    y = np.asarray(labels).astype(int)
    counts = np.bincount(y, minlength=2)
    weights = 1.0 / counts[y]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(y), replacement=True,
        generator=g,
    )  # fmt: skip
