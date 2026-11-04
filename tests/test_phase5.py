"""Phase 5: the model/resolution variants build, run and are compared correctly."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import timm
import torch
import yaml

from fracdet.config import load_config
from fracdet.eval.compare import aggregate, markdown, run_record
from fracdet.models.classifier import build_model

SWEEP = yaml.safe_load(Path("configs/sweep_phase5.yaml").read_text())


@pytest.mark.parametrize("name", SWEEP["configs"])
def test_every_variant_builds_a_one_logit_model(name):
    cfg = load_config(f"configs/{name}")
    assert cfg["model"]["name"] in timm.list_pretrained(cfg["model"]["name"])  # weights exist
    model = build_model({**cfg["model"], "pretrained": False}).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1)


def test_variants_change_only_model_or_resolution():
    """Same recipe for all: only model.name or preprocess.image_size may differ."""
    base = load_config("configs/baseline_resnet50.yaml")
    for name in SWEEP["configs"][1:]:
        cfg = load_config(f"configs/{name}")
        for section in ("train", "augment", "split", "seed"):
            assert cfg[section] == base[section], (name, section)
        changed = {
            "model": cfg["model"]["name"] != base["model"]["name"],
            "size": cfg["preprocess"]["image_size"] != base["preprocess"]["image_size"],
        }
        assert sum(changed.values()) == 1, name


def test_run_record_reports_pr_auc_without_hardware(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"run_name": "x_s1", "experiment": "x", "seed": 1})
    )
    (run / "summary.json").write_text(
        '{"best_epoch": 3, "epochs_run": 5, "minutes": 1.0, "val": {"pr_auc": 0.5, "roc_auc": 0.5}}'
    )
    # Hardware images are all fractured and scored highest: a pure "metal detector".
    preds = pd.DataFrame(
        {"label": [1, 1, 1, 0, 0, 0, 1, 0], "hardware": [1, 1, 0, 0, 0, 0, 0, 0],
         "prob": [0.99, 0.98, 0.2, 0.5, 0.4, 0.3, 0.1, 0.6]}
    )  # fmt: skip
    preds.to_csv(run / "val_predictions.csv", index=False)
    r = run_record(run, 0.9)
    assert r["experiment"] == "x" and r["pr_auc_no_hw"] < 0.5  # weak once metal is removed


def test_markdown_adds_no_hardware_column_only_when_available():
    rows = [
        {"experiment": e, "seed": s, "run_dir": f"{e}{s}", "best_epoch": 5, "minutes": 1.0,
         "pr_auc": 0.7, "roc_auc": 0.8, "spec_at_recall": 0.5, "precision_at_recall": 0.3,
         "alarms_per_100": 50.0}
        for e in ("b", "v") for s in (1, 2)
    ]  # fmt: skip
    old = pd.DataFrame(rows)
    assert "no hardware" not in markdown(aggregate(old, "b"), "b")
    new = old.assign(pr_auc_no_hw=np.linspace(0.6, 0.7, len(old)))
    assert "PR-AUC no hardware" in markdown(aggregate(new, "b"), "b")
