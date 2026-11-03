import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fracdet.logging_utils import get_logger
from fracdet.models.classifier import build_model
from fracdet.train.checkpoint import load_checkpoint, save_checkpoint
from fracdet.train.metrics import binary_metrics
from fracdet.train.schedule import warmup_cosine


def test_binary_metrics_known_values():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1, 0.2]
    m = binary_metrics(y, p, threshold=0.5)
    assert m["recall"] == 0.5  # 1 of 2 positives above 0.5
    assert m["precision"] == 0.5  # 1 of 2 predicted positives is real
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(5 / 6)  # 5 of 6 pos/neg pairs ranked right
    assert m["accuracy"] == 0.6


def test_metrics_single_class_is_nan_not_crash():
    m = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(m["roc_auc"]) and math.isnan(m["recall"])


def test_warmup_then_cosine():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=4, total_steps=20, min_lr_ratio=0.1)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.25) and lrs[3] == pytest.approx(1.0)  # linear warmup
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:], strict=False))  # then non-increasing
    assert lrs[-1] == pytest.approx(0.1, abs=0.02)  # ends near min_lr_ratio


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path):
    cfg = {"model": {"name": "resnet18", "pretrained": False}, "preprocess": {"image_size": 32}}
    model = build_model(cfg["model"]).eval()
    x = torch.randn(2, 3, 32, 32)
    save_checkpoint(tmp_path / "m.pt", model, cfg, epoch=3, metrics={"pr_auc": 0.5})
    loaded, ckpt = load_checkpoint(tmp_path / "m.pt")
    assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert ckpt["preprocess"] == {"image_size": 32} and ckpt["epoch"] == 3
    assert not (tmp_path / "m.pt.tmp").exists()


def test_model_can_overfit_a_tiny_batch():
    """Sanity check every training loop should pass: memorise 8 samples."""
    torch.manual_seed(0)
    model = build_model({"name": "resnet18", "pretrained": False})
    x, y = torch.randn(8, 3, 32, 32), torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def _fit_cfg(tmp_path, fake_root):
    meta = pd.read_csv(fake_root / "dataset.csv")
    splits = np.array(["train", "train", "val", "test"])[np.arange(len(meta)) % 4]
    (tmp_path / "splits").mkdir()
    pd.DataFrame({"image_id": meta.image_id, "split": splits}).to_csv(
        tmp_path / "splits" / "s.csv", index=False
    )
    return {
        "seed": 0,
        "run_name": "test",
        "paths": {"splits_dir": str(tmp_path / "splits")},
        "split": {"name": "s"},
        "model": {"name": "resnet18", "pretrained": False},
        "preprocess": {"image_size": 32},
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "num_workers": 0,
            "early_stop_patience": 0,
        },  # fmt: skip
    }


def test_fit_end_to_end_writes_run_artifacts(tmp_path, fake_root):
    from fracdet.train.fit import fit

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = fit(_fit_cfg(tmp_path, fake_root), fake_root, run_dir, get_logger("t"))
    for name in ["metrics.csv", "best.pt", "last.pt", "val_predictions.csv", "summary.json"]:
        assert (run_dir / name).is_file(), name
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics.epoch.tolist() == [1, 2]
    preds = pd.read_csv(run_dir / "val_predictions.csv")
    assert len(preds) == summary["n_val"] == 10
    assert preds.prob.between(0, 1).all()
    assert json.loads((run_dir / "summary.json").read_text())["best_epoch"] in (1, 2)


def test_fit_never_requests_the_test_split(tmp_path, fake_root, monkeypatch):
    import fracdet.train.fit as fit_module

    real = fit_module.load_split_frames
    requested = []

    def spy(root, csv, splits):
        requested.extend(splits)
        return real(root, csv, splits)

    monkeypatch.setattr(fit_module, "load_split_frames", spy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fit_module.fit(_fit_cfg(tmp_path, fake_root), fake_root, run_dir, get_logger("t"))
    assert "test" not in requested and set(requested) == {"train", "val"}
