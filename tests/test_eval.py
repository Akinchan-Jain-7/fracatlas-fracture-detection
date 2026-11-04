"""Phase 3: threshold rule, bootstrap, subgroup report, protocol guards, end to end."""

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from fracdet.eval.bootstrap import cluster_bootstrap, metric_fn_at
from fracdet.eval.metrics import confusion_counts, eval_metrics
from fracdet.eval.protocol import (
    ProtocolError,
    check_not_already_evaluated,
    check_preregistered,
    check_same_model,
    check_val_reproduced,
    committed_and_unchanged,
)
from fracdet.eval.report import evaluate_frame
from fracdet.eval.threshold import operating_points, recall_table, threshold_for_recall


def _scores(n=300, pos_rate=0.2, seed=0):
    """Imperfect but informative scores, with deliberate ties."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.18, n), 0, 1).round(2)  # rounding -> ties
    return y, p


# ------------------------------------------------------------------- metrics
def test_eval_metrics_known_values():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1, 0.2]
    m = eval_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 2, 1)
    assert m["npv"] == pytest.approx(2 / 3)  # 2 of 3 cleared images are really normal
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.36, 0.36, 0.01, 0.04]))


def test_operating_points_match_brute_force_including_ties():
    y, p = _scores()
    ops = operating_points(y, p)
    assert ops["threshold"].is_monotonic_decreasing
    for _, row in ops.sample(25, random_state=0).iterrows():
        c = confusion_counts(y, p, row["threshold"])
        assert (row.tp, row.fp, row.tn, row.fn) == (c["tp"], c["fp"], c["tn"], c["fn"])


# ----------------------------------------------------------------- threshold
def test_threshold_for_recall_small_example():
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.6, 0.3, 0.7, 0.5, 0.2, 0.1]
    r = threshold_for_recall(y, p, 0.75)  # need 3 of 4 positives
    assert r["positives_required"] == 3
    assert r["gap"] == [0.5, 0.6]  # 3rd positive scores 0.6; next lower score is 0.5
    assert r["threshold"] == pytest.approx(0.55)
    assert r["metrics"]["recall"] == 0.75 and r["metrics"]["fp"] == 1


@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.95, 1.0])
def test_threshold_is_the_highest_meeting_the_target(target):
    y, p = _scores(seed=1)
    r = threshold_for_recall(y, p, target)
    assert r["metrics"]["recall"] >= target
    s_next, s_k = r["gap"]
    assert s_next < r["threshold"] < s_k or (s_next == 0 and r["threshold"] < s_k)
    # Anything above the k-th positive's score misses the target.
    assert eval_metrics(y, p, s_k + 1e-9)["recall"] < target
    # Anywhere in the gap gives the same predictions as the threshold.
    assert confusion_counts(y, p, s_k) == confusion_counts(y, p, r["threshold"])


def test_float_error_does_not_demand_an_extra_positive():
    y = np.r_[np.ones(10, int), np.zeros(10, int)]
    p = np.linspace(1, 0.05, 20)
    assert threshold_for_recall(y, p, 0.9)["positives_required"] == 9  # 0.9 * 10, not 10


def test_recall_table_is_monotone():
    y, p = _scores(seed=2)
    t = recall_table(y, p)
    assert t["threshold"].is_monotonic_decreasing  # higher recall -> lower threshold
    assert (t["recall"] >= t["target_recall"]).all()


# ----------------------------------------------------------------- bootstrap
def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    y, p = _scores(seed=3)
    groups = np.arange(len(y))
    a = cluster_bootstrap(y, p, groups, metric_fn_at(0.5), n_boot=300, seed=7)
    b = cluster_bootstrap(y, p, groups, metric_fn_at(0.5), n_boot=300, seed=7)
    assert a == b
    for k in ("recall", "precision", "roc_auc", "pr_auc"):
        assert a[k]["low"] <= a[k]["value"] <= a[k]["high"]
    assert "tp" not in a  # counts get no interval


def test_cluster_bootstrap_is_wider_when_images_are_copies():
    """5 near-copies per patient are not 5 patients: the interval must reflect that."""
    y0, p0 = _scores(n=120, seed=4)
    y, p = np.repeat(y0, 5), np.repeat(p0, 5)
    groups = np.repeat(np.arange(len(y0)), 5)
    per_image = cluster_bootstrap(y, p, np.arange(len(y)), metric_fn_at(0.5), 400, seed=0)
    clustered = cluster_bootstrap(y, p, groups, metric_fn_at(0.5), 400, seed=0)
    width = lambda r: r["recall"]["high"] - r["recall"]["low"]  # noqa: E731
    assert width(clustered) > 1.7 * width(per_image)  # ~sqrt(5) = 2.2x in theory


def test_bootstrap_skips_undefined_resamples():
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5])
    r = cluster_bootstrap(y, p, np.arange(6), metric_fn_at(0.5), n_boot=200, seed=0)
    assert 0 < r["roc_auc"]["n_valid"] < 200  # resamples without the positive are skipped


# -------------------------------------------------------------------- report
def test_evaluate_frame_subgroups_and_valid_json():
    y, p = _scores(n=400, seed=5)
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "label": y,
            "prob": p,
            "hardware": (rng.random(len(y)) < 0.03).astype(int),
            "region": rng.choice(["hand", "leg", "hip"], len(y), p=[0.6, 0.38, 0.02]),
            "split_group": np.arange(len(y)) // 3,
        }
    )
    rep = evaluate_frame(df, 0.5, n_boot=100, min_positives=10)
    assert {"all", "no_hardware", "hardware", "region=hand", "region=hip"} <= set(rep)
    assert rep["all"]["n"] == 400 and rep["all"]["ci95"] is not None
    assert rep["region=hip"]["ci95"] is None  # too few fractures for an interval
    assert rep["all"]["n_groups"] == df["split_group"].nunique()
    json.dumps(rep, allow_nan=False)


# ------------------------------------------------------------------ protocol
def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
        capture_output=True,
    )  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    return d


def test_preregistration_requires_a_clean_commit(repo):
    f = repo / "threshold.json"
    f.write_text('{"threshold": 0.1}')
    assert not committed_and_unchanged(f)[0]  # untracked
    with pytest.raises(ProtocolError, match="not pre-registered"):
        check_preregistered(f)
    _git(repo, "add", "threshold.json")
    _git(repo, "commit", "-q", "-m", "pre-register")
    info = check_preregistered(f)
    assert len(info["threshold_commit"]) == 40
    f.write_text('{"threshold": 0.05}')  # edited after committing
    with pytest.raises(ProtocolError, match="uncommitted changes"):
        check_preregistered(f)


def test_test_set_is_scored_once_unless_a_reason_is_given(tmp_path):
    check_not_already_evaluated(tmp_path, None)  # nothing yet: fine
    (tmp_path / "test_metrics.json").write_text("{}")
    with pytest.raises(ProtocolError, match="already"):
        check_not_already_evaluated(tmp_path, None)
    check_not_already_evaluated(tmp_path, "crashed before writing figures")


def test_same_model_check():
    record = {"model": {"best_epoch": 16, "trained_commit": "abc", "split_csv_md5": "m"}}
    ckpt = {"epoch": 16, "provenance": {"git": {"commit": "abc"}, "split_csv_md5": "m"}}
    check_same_model(ckpt, record, "m")
    with pytest.raises(ProtocolError, match="epoch"):
        check_same_model({**ckpt, "epoch": 22}, record, "m")
    with pytest.raises(ProtocolError, match="split"):
        check_same_model(ckpt, record, "changed")


def test_val_reproduction_tolerance():
    record = {"val": {"pr_auc": 0.7136, "at_threshold": {"tp": 92, "fp": 150}}}
    check_val_reproduced({"pr_auc": 0.7140, "tp": 92, "fp": 151}, record)  # GPU noise
    with pytest.raises(ProtocolError):
        check_val_reproduced({"pr_auc": 0.69, "tp": 92, "fp": 150}, record)
    with pytest.raises(ProtocolError):
        check_val_reproduced({"pr_auc": 0.7136, "tp": 88, "fp": 140}, record)


# ---------------------------------------------------------------- end to end
def test_tune_then_evaluate_end_to_end(tmp_path, fake_root):
    from fracdet.config import save_config
    from fracdet.eval.pipeline import evaluate_test, find_run, results_dir_for, tune
    from fracdet.logging_utils import get_logger
    from fracdet.train.fit import fit
    from tests.test_training import _fit_cfg

    log = get_logger("t")
    cfg = _fit_cfg(tmp_path, fake_root)
    run_dir = tmp_path / "outputs" / "runs" / "20260101-000000_test"
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    fit(cfg, fake_root, run_dir, log)
    assert find_run("test", [tmp_path / "outputs"]) == run_dir

    eval_cfg = {
        "eval": {
            "target_recall": 0.9,
            "bootstrap": {"n": 50, "seed": 0, "alpha": 0.05},
            "group_column": "split_group",
            "min_positives_for_ci": 1,
            "val_check": {"pr_auc_tol": 0.002, "max_flips": 0},
        }
    }
    record = tune(run_dir, eval_cfg, tmp_path / "tune", log)
    assert record["val"]["at_threshold"]["recall"] >= 0.9
    assert (tmp_path / "tune" / "val_threshold_sweep.png").stat().st_size > 0

    # Pre-register: commit threshold.json into a git repo's results folder.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    dest = results_dir_for("test", repo / "results")
    dest.mkdir(parents=True)
    shutil.copy(tmp_path / "tune" / "threshold.json", dest)

    kwargs = dict(
        run_dir=run_dir, data_root=fake_root, threshold_path=dest / "threshold.json",
        eval_cfg=eval_cfg, log=log, results_root=repo / "results", num_workers=0,
    )  # fmt: skip
    with pytest.raises(ProtocolError, match="not pre-registered"):
        evaluate_test(out_dir=tmp_path / "never", **kwargs)
    assert not (tmp_path / "never").exists()  # refused before scoring anything

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pre-register threshold")
    result = evaluate_test(out_dir=tmp_path / "test_out", **kwargs)

    out = tmp_path / "test_out"
    preds = pd.read_csv(out / "test_predictions.csv")
    meta_test = (pd.read_csv(tmp_path / "splits" / "s.csv").split == "test").sum()
    assert len(preds) == meta_test and preds.prob.between(0, 1).all()
    assert result["protocol"]["preregistered"] and result["protocol"]["val_check"]["flips"] == 0
    assert result["threshold"] == record["threshold"]
    saved = json.loads((out / "test_metrics.json").read_text())
    assert saved["test"]["all"]["n"] == meta_test
    for fig in ["pr_roc_val_vs_test.png", "confusion_test.png", "report.md"]:
        assert (out / fig).is_file(), fig

    # Results committed -> a second run is refused without a reason.
    shutil.copy(out / "test_metrics.json", dest)
    with pytest.raises(ProtocolError, match="already"):
        evaluate_test(out_dir=tmp_path / "again", **kwargs)
