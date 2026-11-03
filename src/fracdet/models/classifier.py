"""Image classifier: a pretrained backbone with a single-logit head.

One output (not two with softmax) because the task is binary: the logit z
gives P(fracture) = sigmoid(z), trained with BCEWithLogitsLoss, which fuses
sigmoid + log for numerical stability.

``timm`` provides ResNet, EfficientNet, DenseNet, ... behind one API, which
Phase 5 uses to compare backbones under identical settings.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """``cfg`` = the ``model`` section: name, pretrained, drop_rate."""
    return timm.create_model(
        cfg["name"],
        pretrained=cfg.get("pretrained", True),
        num_classes=1,  # replaces the 1000-class ImageNet head with one logit
        drop_rate=cfg.get("drop_rate", 0.0),  # dropout before the final layer
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """P(fracture) for a batch, shape (N,)."""
    model.eval()
    return torch.sigmoid(model(x).squeeze(1))
