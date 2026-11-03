"""Checkpoints that carry everything needed to reuse the model correctly.

A bare ``state_dict`` is not enough: to get the same predictions later you
also need the architecture name, the exact preprocessing (image size,
normalisation), and you want provenance (which code, which split, which epoch).
All of it goes in the file, so the web app (Phase 8) loads one file and
rebuilds model + preprocessing with no copied constants.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

import fracdet
from fracdet.models.classifier import build_model

FORMAT_VERSION = 1


def git_info() -> dict[str, Any]:
    """Commit hash and whether there were uncommitted changes (best effort)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    provenance: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "model_cfg": cfg["model"],
        "preprocess": cfg.get("preprocess", {}),
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
        "provenance": {
            "fracdet_version": fracdet.__version__,
            "torch_version": torch.__version__,
            **(provenance or {}),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt "best.pt"


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from the checkpoint's own config and load its weights."""
    # weights_only=False: our own trusted file containing plain dicts/config.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format {ckpt.get('format_version')}")
    model = build_model({**ckpt["model_cfg"], "pretrained": False})
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
