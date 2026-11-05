"""Train the detector on the training split, choosing the epoch on validation.

A run folder gets:
    config.yaml, train.log
    metrics.csv          per epoch: losses, lr, validation AP50 / FROC / top-1, time
    best.pt              highest validation AP50 (all images), with config + provenance
    val_detections.csv   the best epoch's validation boxes (normalised) and scores
    summary.json         headline numbers + provenance

Like the classifier, this never loads the test split.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import fracdet
from fracdet.data.dataset import load_split_frames
from fracdet.data.download import md5_of_file
from fracdet.detect.data import DetectionDataset, PositivesPlusNegatives, collate
from fracdet.detect.metrics import ImageResult, split_metrics
from fracdet.detect.model import build_detector, detect
from fracdet.seed import seed_everything, worker_init_fn
from fracdet.train.checkpoint import git_info
from fracdet.train.schedule import warmup_cosine

TRAIN_SPLITS = ("train", "val")  # never "test"
FORMAT_VERSION = 1


def _subsample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Debug only: n rows, keeping the label mix."""
    if not n or n >= len(frame):
        return frame
    parts = [g.sample(max(1, round(n * len(g) / len(frame))), random_state=seed)
             for _, g in frame.groupby("label")]  # fmt: skip
    return pd.concat(parts).sort_index().reset_index(drop=True)


def predict_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    data_root: str | Path,
    det_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 2,
) -> list[ImageResult]:
    """Detector output for every image of ``frame``, boxes normalised to 0..1."""
    ds = DetectionDataset(frame, str(data_root), det_cfg.get("min_size", 800))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate)  # fmt: skip
    amp = device.type == "cuda"
    out: list[ImageResult] = []
    for images, targets, idx in loader:
        with torch.autocast(device.type, enabled=amp):
            dets = detect(model, [x.to(device, non_blocking=True) for x in images])
        for x, t, i, d in zip(images, targets, idx, dets, strict=True):
            h, w = x.shape[-2:]
            scale = np.array([w, h, w, h], dtype=float)
            out.append(
                ImageResult(
                    image_id=ds.ids[i],
                    label=ds.labels[i],
                    gt=t["boxes"].numpy() / scale,
                    boxes=d["boxes"].numpy() / scale,
                    scores=d["scores"].numpy(),
                )
            )
    return out


def detections_frame(results: list[ImageResult]) -> pd.DataFrame:
    """One row per predicted box (images with no box get one row with NaNs)."""
    rows = []
    for r in results:
        if len(r.scores) == 0:
            rows.append({"image_id": r.image_id, "label": r.label, "rank": -1})
        for k, j in enumerate(np.argsort(-r.scores, kind="stable")):
            x0, y0, x1, y1 = r.boxes[j]
            rows.append({"image_id": r.image_id, "label": r.label, "rank": k, "score": r.scores[j],
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1})  # fmt: skip
    return pd.DataFrame(rows)


def save_detector(path: Path, model, cfg, epoch, metrics, provenance) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "detector",
        "model_state": model.state_dict(),
        "detector_cfg": cfg["detector"],
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
        "provenance": {
            "fracdet_version": fracdet.__version__,
            "torch_version": torch.__version__,
            **provenance,
        },  # fmt: skip
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_detector(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("kind") != "detector" or ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a detector checkpoint of format {FORMAT_VERSION}")
    model = build_detector({**ckpt["detector_cfg"], "pretrained": False})
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def fit_detector(cfg: dict[str, Any], data_root: str | Path, run_dir: str | Path,
                 log: logging.Logger) -> dict[str, Any]:  # fmt: skip
    run_dir = Path(run_dir)
    t_cfg, d_cfg, debug = cfg["train"], cfg["detector"], cfg.get("debug", {})
    seed = cfg["seed"]
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"

    split_csv = Path(cfg["paths"]["splits_dir"]) / f"{cfg['split']['name']}.csv"
    frames = load_split_frames(data_root, split_csv, TRAIN_SPLITS)
    train_df = _subsample(frames["train"], debug.get("n_train"), seed)
    val_df = _subsample(frames["val"], debug.get("n_val"), seed)
    log.info("train %d (%d fractured) | val %d (%d fractured) | device %s | amp %s",
             len(train_df), train_df.label.sum(), len(val_df), val_df.label.sum(),
             device, amp)  # fmt: skip

    train_ds = DetectionDataset(train_df, str(data_root), d_cfg.get("min_size", 800),
                                cfg.get("augment"), train=True)  # fmt: skip
    sampler = PositivesPlusNegatives(train_ds.labels, cfg["data"]["negatives_per_positive"], seed)
    g = torch.Generator()
    g.manual_seed(seed)
    workers = t_cfg.get("num_workers", 2)
    loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], sampler=sampler, num_workers=workers,
        collate_fn=collate, worker_init_fn=worker_init_fn, generator=g, drop_last=True,
        pin_memory=device.type == "cuda",
    )  # fmt: skip

    model = build_detector(d_cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    log.info("detector %s (pretrained=%s): %s trainable parameters", d_cfg["name"],
             d_cfg.get("pretrained", True), f"{sum(p.numel() for p in params):,}")  # fmt: skip
    optimizer = torch.optim.SGD(params, lr=t_cfg["lr"], momentum=t_cfg.get("momentum", 0.9),
                                weight_decay=t_cfg.get("weight_decay", 1e-4))  # fmt: skip
    steps = len(loader)
    scheduler = warmup_cosine(
        optimizer, round(t_cfg.get("warmup_epochs", 1) * steps), t_cfg["epochs"] * steps,
        t_cfg.get("min_lr_ratio", 0.01),
    )  # fmt: skip
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    provenance = {
        "git": git_info(),
        "split_csv_md5": md5_of_file(split_csv),
        "seed": seed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }

    select = t_cfg.get("select_metric", "ap50")
    select_key = {"ap50": "ap50_all"}.get(select, select)
    patience = t_cfg.get("early_stop_patience", 0)
    best, best_epoch, best_val = -1.0, 0, None
    history = []
    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()
        sampler.set_epoch(epoch)
        model.train()
        sums: dict[str, float] = {}
        n = 0
        for images, targets, _ in loader:
            images = [x.to(device, non_blocking=True) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        val = predict_frame(model, val_df, data_root, d_cfg, device,
                            t_cfg.get("eval_batch_size", 4), workers)  # fmt: skip
        m = split_metrics(val, 0.5, cfg.get("evaluate", {}).get("froc_fp_per_image", (0.5, 1, 2)))
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": sum(sums.values()) / max(n, 1),
               **{f"train_{k}": v / max(n, 1) for k, v in sums.items()},
               **{f"val_{k}": v for k, v in m.items() if not k.startswith("n_")},
               "seconds": round(time.time() - t0, 1)}  # fmt: skip
        history.append(row)
        improved = m[select_key] > best
        if improved:
            best, best_epoch, best_val = m[select_key], epoch, val
            save_detector(run_dir / "best.pt", model, cfg, epoch, m, provenance)
        log.info("epoch %2d | loss %.4f | val AP50 %.4f | AP50 fract %.4f | top1 pointing %.3f"
                 " | sens@1fp %.3f | %.0fs%s", epoch, row["train_loss"], m["ap50_all"],
                 m["ap50_fractured"], m.get("top1_pointing", float("nan")),
                 m.get("sens_at_1_fp", float("nan")), row["seconds"],
                 "  *best*" if improved else "")  # fmt: skip
        if patience and epoch - best_epoch >= patience:
            log.info("early stop: no %s gain for %d epochs", select_key, patience)
            break

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in history for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    val_frame = detections_frame(best_val or [])
    val_frame.to_csv(run_dir / "val_detections.csv", index=False)
    summary = {"run_name": cfg.get("run_name"), "seed": seed, "best_epoch": best_epoch,
               "epochs_run": len(history), f"val_{select_key}": best,
               "val_best": history[best_epoch - 1] if best_epoch else None,
               "n_train": len(train_df), "n_val": len(val_df), **provenance}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log.info("done: best epoch %d, val %s %.4f", best_epoch, select_key, best)
    return summary
