"""The two Phase 3 steps. The scripts in ``scripts/`` are thin wrappers around these.

Step 1, ``tune``: validation only. Reads the run's ``val_predictions.csv``,
applies the rule from ``configs/eval.yaml`` and writes ``threshold.json`` plus
the validation trade-off table. No images are loaded and no test data is
touched.

    -> commit threshold.json to results/phase3_<run>/ (the pre-registration)

Step 2, ``evaluate_test``: runs the protocol checks, re-scores validation to
prove it is the same model, then scores the test split once and writes
predictions, metrics with intervals and figures.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fracdet.config import load_config
from fracdet.data.download import md5_of_file
from fracdet.eval import plots
from fracdet.eval.metrics import eval_metrics
from fracdet.eval.predict import score_split
from fracdet.eval.protocol import (
    check_not_already_evaluated,
    check_preregistered,
    check_same_model,
    check_val_reproduced,
)
from fracdet.eval.report import evaluate_frame, markdown_table
from fracdet.eval.threshold import (
    RULE_MAX_THRESHOLD_AT_RECALL,
    operating_points,
    recall_table,
    reference_points,
    threshold_for_recall,
)
from fracdet.train.checkpoint import git_info


def results_dir_for(run_name: str, root: str | Path = "results") -> Path:
    """Where a run's committed Phase 3 records live."""
    return Path(root) / f"phase3_{run_name}"


def find_run(name: str, search: list[str | Path]) -> Path:
    """The newest run folder ``*_<name>`` containing best.pt, under any search root.

    On Kaggle the Phase 2 run arrives as a read-only notebook-output input
    (``/kaggle/input/<notebook>/...``), so its exact path is not known in advance.
    """
    found = []
    for root in map(Path, search):
        if root.is_dir():
            found += [p.parent for p in root.rglob("best.pt") if p.parent.name.endswith(f"_{name}")]
    if not found:
        raise FileNotFoundError(f"No run '*_{name}' with best.pt under {[str(s) for s in search]}")
    return sorted(found, key=lambda p: p.name)[-1]


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _split_csv(run_cfg: dict[str, Any]) -> Path:
    return Path(run_cfg["paths"]["splits_dir"]) / f"{run_cfg['split']['name']}.csv"


def _with_groups(preds: pd.DataFrame, split_csv: Path) -> pd.DataFrame:
    """Add split_group from the split file (Phase 2 val_predictions.csv predates it)."""
    split = pd.read_csv(split_csv, keep_default_na=False)
    if "split_group" in preds or "split_group" not in split:
        return preds  # already there, or no groups at all (-> per-image bootstrap)
    groups = split[["image_id", "split_group"]]
    out = preds.merge(groups, on="image_id", how="left", validate="one_to_one")
    if out["split_group"].isna().any():
        raise ValueError("Some predictions have no split_group: wrong split file?")
    return out


# --------------------------------------------------------------------- step 1
def tune(run_dir: str | Path, eval_cfg: dict[str, Any], out_dir: str | Path,
         log: logging.Logger) -> dict[str, Any]:  # fmt: skip
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    e = eval_cfg["eval"]
    run_cfg = load_config(run_dir / "config.yaml")
    summary = json.loads((run_dir / "summary.json").read_text())
    split_csv = _split_csv(run_cfg)
    split_md5 = md5_of_file(split_csv)
    if split_md5 != summary["split_csv_md5"]:
        raise ValueError(f"Split file {split_csv} changed since training ({split_md5})")

    preds_path = run_dir / "val_predictions.csv"
    val = _with_groups(pd.read_csv(preds_path), split_csv)
    y, p = val["label"].to_numpy(), val["prob"].to_numpy()
    log.info("validation: %d images, %d fractured (run %s)", len(val), y.sum(), run_dir.name)

    chosen = threshold_for_recall(y, p, e["target_recall"])
    t = chosen["threshold"]
    val_all = eval_metrics(y, p, t)
    table = recall_table(y, p)
    record = {
        "phase": 3,
        "purpose": "Operating threshold chosen on validation only, committed before the test run.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "created_by": git_info(),
        "rule": {
            "name": RULE_MAX_THRESHOLD_AT_RECALL,
            "target_recall": e["target_recall"],
            "placement": "midpoint between the k-th highest fracture score and the next lower one",
        },
        "threshold": t,
        "gap": chosen["gap"],
        "model": {
            "run_name": summary["run_name"],
            "best_epoch": summary["best_epoch"],
            "trained_commit": summary["git"]["commit"],
            "split_name": summary["split_name"],
            "split_csv_md5": summary["split_csv_md5"],
        },
        "val": {
            "n": int(len(val)),
            "n_fractured": int(y.sum()),
            "pr_auc": val_all["pr_auc"],
            "roc_auc": val_all["roc_auc"],
            "at_threshold": val_all,
            "val_predictions_md5": md5_of_file(preds_path),
        },
        "reference_points": reference_points(y, p),
        "recall_table": table.to_dict(orient="records"),
    }
    record = _clean(record)
    _write_json(record, out_dir / "threshold.json")
    val.to_csv(out_dir / "val_predictions.csv", index=False)
    operating_points(y, p).to_csv(out_dir / "val_operating_points.csv", index=False)

    b = e["bootstrap"]
    val_report = evaluate_frame(
        val, t, b["n"], b["seed"], e["min_positives_for_ci"], e["group_column"]
    )
    _write_json(val_report, out_dir / "val_metrics.json")
    plots.plot_threshold_sweep(
        operating_points(y, p), t, out_dir / "val_threshold_sweep.png",
        f"Validation trade-off — chosen threshold for recall ≥ {e['target_recall']:.0%}",
    )  # fmt: skip

    log.info("threshold %.4f (val gap %.4f-%.4f)", t, *chosen["gap"])
    log.info("what each recall target would cost on validation:\n%s",
             table.round(3).to_string(index=False))  # fmt: skip
    log.info("validation at the chosen threshold (95%% cluster-bootstrap CI):\n%s",
             markdown_table(val_report))  # fmt: skip
    return record


# --------------------------------------------------------------------- step 2
def evaluate_test(
    run_dir: str | Path,
    data_root: str | Path,
    threshold_path: str | Path,
    eval_cfg: dict[str, Any],
    out_dir: str | Path,
    log: logging.Logger,
    results_root: str | Path = "results",
    rerun_reason: str | None = None,
    require_preregistration: bool = True,
    num_workers: int = 4,
) -> dict[str, Any]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    e = eval_cfg["eval"]
    record = json.loads(Path(threshold_path).read_text())
    t = float(record["threshold"])

    # ---- protocol checks, all before any test image is read
    prereg = check_preregistered(threshold_path) if require_preregistration else None
    if prereg is None:
        log.warning("PRE-REGISTRATION CHECK SKIPPED: results are marked as not pre-registered")
    check_not_already_evaluated(
        results_dir_for(record["model"]["run_name"], results_root), rerun_reason
    )

    run_cfg = load_config(run_dir / "config.yaml")
    split_csv = _split_csv(run_cfg)
    split_md5 = md5_of_file(split_csv)
    ckpt_path = run_dir / "best.pt"
    log.info("re-scoring validation with %s to confirm it is the tuned model", ckpt_path)
    val, ckpt = score_split(ckpt_path, data_root, split_csv, "val", num_workers=num_workers)
    check_same_model(ckpt, record, split_md5)
    val_now = eval_metrics(val["label"], val["prob"], t)
    val_check = check_val_reproduced(val_now, record, **e["val_check"])
    log.info("validation reproduced: %s", val_check)

    # ---- the one test pass
    log.info("scoring the TEST split (once) at threshold %.4f", t)
    test, _ = score_split(ckpt_path, data_root, split_csv, "test", num_workers=num_workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_dir / "test_predictions.csv", index=False)  # saved before any analysis
    val.to_csv(out_dir / "val_predictions_rescored.csv", index=False)

    b = e["bootstrap"]
    args = (t, b["n"], b["seed"], e["min_positives_for_ci"], e["group_column"])
    test_report = evaluate_frame(test, *args)
    val_report = evaluate_frame(val, *args)
    result = _clean(
        {
            "phase": 3,
            "threshold": t,
            "rule": record["rule"],
            "model": record["model"],
            "protocol": {
                "preregistered": prereg is not None,
                **(prereg or {}),
                "rerun_reason": rerun_reason,
                "val_check": val_check,
                "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "evaluated_by": git_info(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "bootstrap": {**b, "unit": e["group_column"]},
            },
            "test": test_report,
            "test_at_0.5": eval_metrics(test["label"], test["prob"], 0.5),
            "val": val_report,
        }
    )
    _write_json(result, out_dir / "test_metrics.json")

    plots.plot_pr_roc(
        {"validation": (val["label"], val["prob"]), "test": (test["label"], test["prob"])},
        t, out_dir / "pr_roc_val_vs_test.png", "Baseline ResNet-50: validation vs test",
    )  # fmt: skip
    plots.plot_confusion(
        test_report["all"]["metrics"], out_dir / "confusion_test.png",
        f"Test, threshold {t:.3f}",
    )  # fmt: skip
    if "no_hardware" in test_report:
        plots.plot_confusion(
            test_report["no_hardware"]["metrics"], out_dir / "confusion_test_no_hardware.png",
            f"Test without hardware, threshold {t:.3f}",
        )  # fmt: skip

    tables = (
        f"## Test at threshold {t:.4f} (95% cluster-bootstrap CI)\n\n"
        f"{markdown_table(test_report)}\n\n"
        f"## Validation at the same threshold\n\n{markdown_table(val_report)}\n"
    )
    (out_dir / "report.md").write_text(tables)
    log.info("\n%s", tables)
    return result
