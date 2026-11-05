"""Sanity check for the heatmaps: do they depend on what the model learned?

Adebayo et al. (2018, "Sanity Checks for Saliency Maps") showed that several
popular saliency methods produce almost the same maps for a trained network
and for the same network with random weights. Those maps were mostly edge
detectors: they showed the image, not the model's reasoning.

The test: re-initialise weights, recompute the maps on the same images and
measure how similar they still are.

- ``head_and_last_stage``: trained weights everywhere except the classifier
  head and the last stage (the layer Grad-CAM reads), which get a fresh random
  initialisation.
- ``full``: a freshly initialised network, no trained weights at all.

If Grad-CAM reflects the trained model, the rank correlation with the trained
maps should fall towards 0, and the random models' pointing-game hit rate
towards chance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from fracdet.models.classifier import build_model

LEVELS = ("head_and_last_stage", "full")


def randomized_copy(
    model: nn.Module, model_cfg: dict[str, Any], level: str, seed: int = 0
) -> nn.Module:
    """A copy of ``model`` with some or all weights re-initialised (timm's own init)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    torch.manual_seed(seed)
    fresh = build_model({**model_cfg, "pretrained": False})
    if level == "head_and_last_stage":
        last_stage = model.feature_info[-1]["module"]
        head_root = model.default_cfg["classifier"].split(".")[0]  # "head" or "fc"
        prefixes = (f"{last_stage}.", f"{head_root}.")
        trained = model.state_dict()
        state = {
            k: (v if k.startswith(prefixes) else trained[k]) for k, v in fresh.state_dict().items()
        }
        fresh.load_state_dict(state)
    device = next(model.parameters()).device
    return fresh.to(device).eval()


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two maps over the same pixels; NaN if either is flat."""
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
