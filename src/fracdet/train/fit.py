"""The training driver: data -> model -> epochs -> best checkpoint + run record.

Every run writes to its own folder:
    config.yaml           resolved config (what actually ran)
    train.log             console log
    metrics.csv           one row per epoch (train loss, val loss/metrics, lr, time)
    best.pt / last.pt     checkpoints (best = highest val PR-AUC)
    val_predictions.csv   best model's val probabilities + metadata (Phase 3 tunes
                          the threshold on these)
    summary.json          headline numbers + provenance (git commit, split hash)

The TEST split is never loaded here. It is reserved for one final evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fracdet.data.dataset import FracAtlasDataset, load_split_frames
from fracdet.data.download import md5_of_file
from fracdet.data.transforms import build_transforms
from fracdet.models.classifier import build_model, count_parameters
from fracdet.seed import seed_everything, worker_init_fn
from fracdet.train.checkpoint import git_info, load_checkpoint, save_checkpoint
from fracdet.train.engine import predict, train_one_epoch
from fracdet.train.metrics import binary_metrics
from fracdet.train.schedule import warmup_cosine

TRAIN_SPLITS = ("train", "val")  # never "test"
PREDICTION_COLUMNS = ["image_id", "label", "region", "view", "hardware", "multiscan", "split_group"]


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """No weight decay on biases and normalisation weights (1-D params).

    Weight decay shrinks weights toward 0 to regularise. For BatchNorm scales
    and biases that only shifts/rescales activations and tends to hurt, so the
    standard practice is to exempt them.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _maybe_subsample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Debug only: keep n rows, stratified by label, for fast smoke runs."""
    if not n or n >= len(frame):
        return frame
    parts = [
        g.sample(max(1, round(n * len(g) / len(frame))), random_state=seed)
        for _, g in frame.groupby("label")
    ]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _loader(ds, batch_size, shuffle, num_workers, seed, drop_last=False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=g,  # reproducible shuffle order
        drop_last=drop_last,  # avoid a final batch of 1, which breaks BatchNorm statistics
    )


def fit(cfg: dict[str, Any], data_root: str | Path, run_dir: str | Path, log: logging.Logger):
    run_dir = Path(run_dir)
    t_cfg, debug = cfg["train"], cfg.get("debug", {})
    seed_everything(cfg["seed"], deterministic=t_cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    channels_last = device.type == "cuda"  # faster convolutions on tensor cores

    split_csv = Path(cfg["paths"]["splits_dir"]) / f"{cfg['split']['name']}.csv"
    frames = load_split_frames(data_root, split_csv, TRAIN_SPLITS)
    train_df = _maybe_subsample(frames["train"], debug.get("n_train"), cfg["seed"])
    val_df = _maybe_subsample(frames["val"], debug.get("n_val"), cfg["seed"])
    log.info(
        "train %d (%d fractured) | val %d (%d fractured) | device %s | amp %s",
        len(train_df), train_df.label.sum(), len(val_df), val_df.label.sum(), device, amp,
    )  # fmt: skip

    pre, aug = cfg.get("preprocess", {}), cfg.get("augment", {})
    workers = t_cfg.get("num_workers", 4)
    train_loader = _loader(
        FracAtlasDataset(train_df, build_transforms(pre, aug, train=True)),
        t_cfg["batch_size"], True, workers, cfg["seed"], drop_last=True,
    )  # fmt: skip
    val_loader = _loader(
        FracAtlasDataset(val_df, build_transforms(pre, train=False)),
        t_cfg.get("eval_batch_size", t_cfg["batch_size"] * 2), False, workers, cfg["seed"],
    )  # fmt: skip

    model = build_model(cfg["model"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    params = count_parameters(model)
    log.info("model %s: %s parameters", cfg["model"]["name"], f"{params['total']:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        _param_groups(model, t_cfg["weight_decay"]), lr=t_cfg["lr"], betas=(0.9, 0.999)
    )
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(
        optimizer,
        warmup_steps=round(t_cfg.get("warmup_epochs", 1) * steps_per_epoch),
        total_steps=t_cfg["epochs"] * steps_per_epoch,
        min_lr_ratio=t_cfg.get("min_lr_ratio", 0.01),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    select = t_cfg.get("select_metric", "pr_auc")
    patience = t_cfg.get("early_stop_patience", 0) or None
    provenance = {
        "git": git_info(),
        "split_name": cfg["split"]["name"],
        "split_csv_md5": md5_of_file(split_csv),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }

    best_score, best_epoch, epochs_without_gain = -np.inf, -1, 0
    metrics_path = run_dir / "metrics.csv"
    start = time.time()
    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, amp,
            t_cfg.get("grad_clip"), channels_last,
        )  # fmt: skip
        pv = predict(model, val_loader, criterion, device, amp, channels_last)
        m = binary_metrics(pv["label"], pv["prob"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "val_loss": pv["loss"],
            **{f"val_{k}": v for k, v in m.items()},
            "seconds": time.time() - t0,
        }
        with metrics_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                w.writeheader()
            w.writerow(row)

        improved = m[select] > best_score
        log.info(
            "epoch %2d | train_loss %.4f | val_loss %.4f | val PR-AUC %.4f | ROC-AUC %.4f "
            "| recall@0.5 %.3f | precision@0.5 %.3f | %.0fs%s",
            epoch, tr["loss"], pv["loss"], m["pr_auc"], m["roc_auc"], m["recall"],
            m["precision"], row["seconds"], "  *best*" if improved else "",
        )  # fmt: skip
        save_checkpoint(run_dir / "last.pt", model, cfg, epoch, m, provenance)
        if improved:
            best_score, best_epoch, epochs_without_gain = m[select], epoch, 0
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, m, provenance)
        else:
            epochs_without_gain += 1
            if patience and epochs_without_gain >= patience:
                log.info("early stop: no %s gain for %d epochs", select, patience)
                break

    log.info("training done in %.1f min; best epoch %d", (time.time() - start) / 60, best_epoch)
    return finalize_run(cfg, data_root, run_dir, log)


def finalize_run(
    cfg: dict[str, Any], data_root: str | Path, run_dir: str | Path, log: logging.Logger
) -> dict[str, Any]:
    """Re-score validation with the saved best.pt and write the run record.

    Separate from the epoch loop so a finished run can be (re)finalised from its
    files alone, without retraining. ``scripts/finalize_run.py`` does that.
    Loading from disk (not reusing the in-memory model) proves the checkpoint
    file itself reproduces the reported score.
    """
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_cfg, debug = cfg["train"], cfg.get("debug", {})
    amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    channels_last = device.type == "cuda"

    split_csv = Path(cfg["paths"]["splits_dir"]) / f"{cfg['split']['name']}.csv"
    frames = load_split_frames(data_root, split_csv, TRAIN_SPLITS)
    n_train = len(_maybe_subsample(frames["train"], debug.get("n_train"), cfg["seed"]))
    val_df = _maybe_subsample(frames["val"], debug.get("n_val"), cfg["seed"])
    val_loader = _loader(
        FracAtlasDataset(val_df, build_transforms(cfg.get("preprocess", {}), train=False)),
        t_cfg.get("eval_batch_size", t_cfg["batch_size"] * 2), False,
        t_cfg.get("num_workers", 4), cfg["seed"],
    )  # fmt: skip

    model, ckpt = load_checkpoint(run_dir / "best.pt", device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    pv = predict(model, val_loader, nn.BCEWithLogitsLoss(), device, amp, channels_last)
    best_metrics = binary_metrics(pv["label"], pv["prob"])
    saved = ckpt["metrics"][t_cfg.get("select_metric", "pr_auc")]
    rescored = best_metrics[t_cfg.get("select_metric", "pr_auc")]
    if abs(saved - rescored) > 1e-3:
        log.warning("re-scored %.4f differs from saved %.4f: non-reproducible?", rescored, saved)

    preds = val_df.loc[pv["index"], [c for c in PREDICTION_COLUMNS if c in val_df]].copy()
    preds["prob"] = pv["prob"]
    preds.to_csv(run_dir / "val_predictions.csv", index=False)

    epochs = pd.read_csv(run_dir / "metrics.csv")
    summary = {
        "run_name": cfg.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "best_epoch": int(ckpt["epoch"]),
        "epochs_run": int(epochs["epoch"].max()),
        "select_metric": t_cfg.get("select_metric", "pr_auc"),
        # NaN (e.g. precision with no positive predictions) -> null: valid JSON
        "val": {k: (None if math.isnan(v) else round(v, 4)) for k, v in best_metrics.items()},
        "n_train": n_train,
        "n_val": len(val_df),
        "minutes": round(float(epochs["seconds"].sum()) / 60, 2),
        "parameters": count_parameters(model)["total"],
        **ckpt["provenance"],
        "finalized_by": git_info(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    log.info("val with best.pt: %s", summary["val"])
    return summary
