"""Print a run's per-epoch metrics and summary in a compact, readable form.

Usage:
    python scripts/show_run.py outputs/runs/20261103-160000_baseline_resnet50
    python scripts/show_run.py --latest baseline_resnet50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = {
    "epoch": "ep",
    "train_loss": "train_loss",
    "val_loss": "val_loss",
    "val_pr_auc": "PR-AUC",
    "val_roc_auc": "ROC-AUC",
    "val_recall": "rec@.5",
    "val_precision": "prec@.5",
    "seconds": "sec",
}


def latest_run(name: str, runs_dir: Path = Path("outputs/runs")) -> Path:
    matches = sorted(runs_dir.glob(f"*_{name}"))
    if not matches:
        raise SystemExit(f"No runs named {name!r} in {runs_dir}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--latest", metavar="RUN_NAME", help="most recent run with this name")
    args = parser.parse_args()
    run = args.run_dir or latest_run(args.latest)

    metrics = pd.read_csv(run / "metrics.csv")[list(COLUMNS)].rename(columns=COLUMNS)
    print(f"run: {run}\n")
    print(metrics.round({"train_loss": 4, "val_loss": 4, "sec": 0}).round(3).to_string(index=False))
    summary_path = run / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        print(f"\nbest epoch {s['best_epoch']} of {s['epochs_run']}", end=" ")
        print(f"({s['minutes']} min, {s['device']})")
        print("best val:", ", ".join(f"{k}={v}" for k, v in s["val"].items()))
        print(f"commit {s['git']['commit'][:7]} dirty={s['git']['dirty']} split={s['split_name']}")
    else:
        print("\n(no summary.json yet: run still in progress or crashed)")


if __name__ == "__main__":
    main()
