"""One training epoch and one prediction pass.

Mixed precision (AMP): matrix multiplies run in float16 on the GPU's tensor
cores (~2x faster on a T4, half the activation memory), while weights stay
float32. ``GradScaler`` multiplies the loss before backward so tiny float16
gradients don't underflow to zero, then un-scales before the optimizer step.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float | None = None,
    channels_last: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)  # loss in float32 for stability
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite loss {loss.item()}; lower the LR or check data")

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(y)
        n += len(y)
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    channels_last: bool = False,
) -> dict[str, np.ndarray | float]:
    """Probabilities for every sample, in dataset order (via the returned index)."""
    model.eval()
    idx, labels, probs = [], [], []
    total_loss, n = 0.0, 0
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x).squeeze(1).float()
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        idx.append(i.numpy())
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
    order = np.argsort(np.concatenate(idx))
    return {
        "index": np.concatenate(idx)[order],
        "label": np.concatenate(labels)[order],
        "prob": np.concatenate(probs)[order],
        "loss": total_loss / max(n, 1),
    }
