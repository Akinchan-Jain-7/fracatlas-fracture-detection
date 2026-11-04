"""Phase 3, step 1: choose the operating threshold on VALIDATION only.

Reads the run's val_predictions.csv (no images, no test data), applies the rule
in configs/eval.yaml, and writes to --out:
    threshold.json            the threshold + how it was chosen (commit this)
    val_predictions.csv       the validation scores it was chosen on (+ split_group)
    val_operating_points.csv  every possible threshold with its counts and rates
    val_metrics.json          validation metrics at the threshold, with 95% CIs
    val_threshold_sweep.png   recall / precision / specificity vs threshold

Then copy those files into results/phase3_<run_name>/, commit and push. That
commit is the pre-registration: it happens before the test set is scored.

Usage:
    python scripts/tune_threshold.py --find-run baseline_resnet50
    python scripts/tune_threshold.py --run-dir outputs/runs/20261103-153336_baseline_resnet50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracdet.config import load_config
from fracdet.eval.pipeline import find_run, results_dir_for, tune
from fracdet.logging_utils import get_logger

DEFAULT_SEARCH = ["outputs/runs", "/kaggle/input"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--run-dir", type=Path)
    where.add_argument(
        "--find-run", metavar="RUN_NAME", help="search outputs/runs and /kaggle/input"
    )
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--out", type=Path, help="default outputs/eval/<run_name>/tune")
    args = parser.parse_args()

    run_dir = args.run_dir or find_run(args.find_run, DEFAULT_SEARCH)
    run_name = load_config(run_dir / "config.yaml").get("run_name", run_dir.name)
    out = args.out or Path("outputs/eval") / run_name / "tune"
    log = get_logger("tune", log_file=out / "tune.log")
    log.info("run %s -> %s", run_dir, out)
    record = tune(run_dir, load_config(args.config), out, log)

    dest = results_dir_for(run_name)
    print(
        f"\nTHRESHOLD {record['threshold']:.6f}\n\nNext (pre-registration), from the repo root:\n"
        f"  mkdir -p {dest}\n  cp {out}/*.json {out}/*.csv {out}/*.png {dest}/\n"
        f"  git add {dest} && git commit -m 'phase3: pre-register threshold' && git push\n"
        "Only then run scripts/evaluate_test.py."
    )


if __name__ == "__main__":
    main()
