"""Phase 4: imbalance options, the sweep runner and the selection rule."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from fracdet.eval.compare import aggregate, decide
from fracdet.train.imbalance import FocalLoss, balance_weight, balanced_sampler, build_criterion

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- losses
def test_focal_with_gamma_zero_is_bce():
    torch.manual_seed(0)
    logits, y = torch.randn(64), (torch.rand(64) < 0.2).float()
    assert torch.allclose(FocalLoss(gamma=0)(logits, y), nn.BCEWithLogitsLoss()(logits, y))


def test_focal_downweights_easy_examples_more_than_hard_ones():
    y = torch.tensor([1.0, 1.0])
    logits = torch.tensor([4.0, -2.0])  # an easy (confident, right) and a hard positive
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    focal = torch.stack([FocalLoss(2.0)(logits[i : i + 1], y[i : i + 1]) for i in range(2)])
    ratio = focal / bce
    assert ratio[0] < 0.01 and ratio[1] > 0.7  # easy one almost ignored, hard one kept


def test_pos_weight_auto_balances_the_classes():
    labels = np.r_[np.ones(20), np.zeros(80)]
    assert balance_weight(labels) == 4.0
    crit, rec = build_criterion({"name": "bce", "pos_weight": "auto"}, labels)
    assert rec["pos_weight"] == 4.0 and float(crit.pos_weight) == 4.0
    # Weighted total loss: positives and negatives now contribute equally at p = 0.5.
    logits, y = torch.zeros(100), torch.as_tensor(labels, dtype=torch.float32)
    per = nn.functional.binary_cross_entropy_with_logits(
        logits, y, pos_weight=crit.pos_weight, reduction="none"
    )
    assert per[y == 1].sum() == pytest.approx(per[y == 0].sum())


def test_default_and_unknown_losses():
    crit, rec = build_criterion(None, [0, 1])
    assert isinstance(crit, nn.BCEWithLogitsLoss) and rec == {"name": "bce", "pos_weight": None}
    with pytest.raises(ValueError, match="Unknown loss"):
        build_criterion({"name": "dice"}, [0, 1])


def test_balanced_sampler_gives_about_half_positives():
    labels = np.r_[np.ones(100), np.zeros(900)].astype(int)
    idx = np.array(list(balanced_sampler(labels, seed=0)))
    assert len(idx) == len(labels)  # same epoch length
    assert 0.45 < labels[idx].mean() < 0.55
    assert np.array_equal(idx, np.array(list(balanced_sampler(labels, seed=0))))  # seeded


# ------------------------------------------------------- fit integration
@pytest.mark.parametrize(
    "train_extra",
    [{"sampler": "balanced"}, {"loss": {"name": "focal", "gamma": 2.0}},
     {"loss": {"name": "bce", "pos_weight": "auto"}}],
)  # fmt: skip
def test_fit_runs_with_each_imbalance_option(tmp_path, fake_root, train_extra):
    from fracdet.logging_utils import get_logger
    from fracdet.train.fit import fit
    from tests.test_training import _fit_cfg

    cfg = _fit_cfg(tmp_path, fake_root)
    cfg["train"].update(train_extra, epochs=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = fit(cfg, fake_root, run_dir, get_logger("t"))
    rec = summary["imbalance"]
    assert rec["sampler"] == train_extra.get("sampler")
    assert rec["loss"]["name"] == train_extra.get("loss", {}).get("name", "bce")
    # val loss is always plain BCE, so it is comparable across variants
    assert pd.read_csv(run_dir / "metrics.csv")["val_loss"].notna().all()


# ------------------------------------------------------------ decision
def _runs(spec_gain, spec_noise, pr_gain=0.0, seeds=(42, 43, 44)):
    rng = np.random.default_rng(0)
    rows = []
    for s in seeds:
        shared = rng.normal(0, spec_noise)  # what a seed does to both variants
        for exp, g, pg in [("baseline", 0.0, 0.0), ("variant", spec_gain, pr_gain)]:
            rows.append(
                {"experiment": exp, "seed": s, "run_dir": f"{exp}_{s}", "best_epoch": 10,
                 "minutes": 10.0, "pr_auc": 0.71 + pg + rng.normal(0, 0.01),
                 "roc_auc": 0.87, "spec_at_recall": 0.56 + g + shared + rng.normal(0, spec_noise),
                 "precision_at_recall": 0.3, "alarms_per_100": 52.0}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_clear_gain_wins_and_picks_the_median_seed():
    runs = _runs(spec_gain=0.10, spec_noise=0.01)
    d = decide(aggregate(runs, "baseline"), runs, "baseline", "spec_at_recall", "pr_auc")
    assert d["winner"] == "variant" and d["changed"]
    v = runs[runs.experiment == "variant"].sort_values("spec_at_recall")
    assert d["representative_run"] == v.iloc[1]["run_dir"]  # the median, not the best


def test_gain_within_noise_keeps_the_baseline():
    runs = _runs(spec_gain=0.01, spec_noise=0.04)
    d = decide(aggregate(runs, "baseline"), runs, "baseline", "spec_at_recall", "pr_auc")
    assert d["winner"] == "baseline" and not d["changed"]
    assert not d["verdicts"]["variant"]["wins"]


def test_guard_blocks_a_variant_that_loses_ranking():
    runs = _runs(spec_gain=0.10, spec_noise=0.01, pr_gain=-0.10)
    d = decide(aggregate(runs, "baseline"), runs, "baseline", "spec_at_recall", "pr_auc")
    assert d["winner"] == "baseline"


# ------------------------------------------------- sweep + compare scripts
def test_sweep_and_compare_end_to_end(tmp_path, fake_root):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name in ["base.yaml", "baseline_resnet50.yaml", "imb_focal.yaml"]:
        shutil.copy(REPO / "configs" / name, cfg_dir)
    meta = pd.read_csv(fake_root / "dataset.csv")
    splits = np.array(["train", "train", "val", "test"])[np.arange(len(meta)) % 4]
    (tmp_path / "splits").mkdir()
    pd.DataFrame({"image_id": meta.image_id, "split": splits}).to_csv(
        tmp_path / "splits" / "s.csv", index=False
    )
    sweep = {
        "name": "t", "configs": ["baseline_resnet50.yaml", "imb_focal.yaml"], "seeds": [1, 2],
        "baseline": "baseline_resnet50",
        "selection": {"recall_target": 0.9, "primary": "spec_at_recall", "guard": "pr_auc"},
    }  # fmt: skip
    (cfg_dir / "sweep.yaml").write_text(yaml.safe_dump(sweep))
    overrides = [
        "model.name=resnet18", "model.pretrained=false", "preprocess.image_size=32",
        "train.epochs=1", "train.batch_size=4", "train.num_workers=0",
        f"paths.splits_dir={tmp_path / 'splits'}", "split.name=s",
        f"paths.output_dir={tmp_path / 'outputs'}", f"paths.data_dir={fake_root}",
    ]  # fmt: skip
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    run = subprocess.run(
        [sys.executable, "scripts/sweep.py", str(cfg_dir / "sweep.yaml"),
         "--out-root", str(tmp_path / "sweeps"), "--set", *overrides],
        cwd=REPO, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads((tmp_path / "sweeps" / "t" / "manifest.json").read_text())
    assert [j["run_name"] for j in manifest["jobs"]] == [
        "baseline_resnet50_s1", "imb_focal_s1", "baseline_resnet50_s2", "imb_focal_s2",
    ]  # seed-major order  # fmt: skip
    assert all(j["status"] == "done" for j in manifest["jobs"])

    cmp_run = subprocess.run(
        [sys.executable, "scripts/compare_runs.py", str(cfg_dir / "sweep.yaml"),
         "--out", str(tmp_path / "sweeps" / "t")],
        cwd=REPO, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert cmp_run.returncode == 0, cmp_run.stdout + cmp_run.stderr
    result = json.loads((tmp_path / "sweeps" / "t" / "compare.json").read_text())
    assert result["n_runs"] == 4 and result["decision"]["winner"] in {
        "baseline_resnet50",
        "imb_focal",
    }
    runs = pd.read_csv(tmp_path / "sweeps" / "t" / "runs.csv")
    assert sorted(runs.seed.unique()) == [1, 2]
    assert (tmp_path / "sweeps" / "t" / "compare.png").stat().st_size > 0
