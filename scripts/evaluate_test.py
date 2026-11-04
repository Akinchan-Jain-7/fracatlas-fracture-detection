"""Phase 3, step 2: score the TEST split once, with the pre-registered threshold.

Refuses to run unless:
  - results/phase3_<run>/threshold.json is committed to git and unmodified;
  - the checkpoint is the one that threshold was tuned for (epoch, commit, split);
  - re-scoring validation with it reproduces the validation numbers;
  - this model has no committed test results yet (or --rerun-reason is given).

Writes to --out (default outputs/eval/<run_name>/test):
    test_predictions.csv          every test image: label, prob, region, hardware, group
    test_metrics.json             all / no_hardware / hardware / per region, 95% CIs,
                                  metrics at 0.5 for comparison, protocol record
    report.md                     the tables, ready to paste
    pr_roc_val_vs_test.png, confusion_test.png, confusion_test_no_hardware.png

Usage (Kaggle, GPU):
    python scripts/evaluate_test.py --find-run baseline_resnet50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracdet.config import load_config
from fracdet.eval.pipeline import evaluate_test, find_run, results_dir_for
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir

DEFAULT_SEARCH = ["outputs/runs", "/kaggle/input"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--run-dir", type=Path)
    where.add_argument(
        "--find-run", metavar="RUN_NAME", help="search outputs/runs and /kaggle/input"
    )
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--threshold", type=Path, help="default results/phase3_<run>/threshold.json"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, help="default outputs/eval/<run_name>/test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rerun-reason", help="required if test results are already committed")
    parser.add_argument(
        "--skip-preregistration-check",
        action="store_true",
        help="dry runs only: results are marked preregistered=false",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_run(args.find_run, DEFAULT_SEARCH)
    run_cfg = load_config(run_dir / "config.yaml")
    run_name = run_cfg.get("run_name", run_dir.name)
    threshold = args.threshold or results_dir_for(run_name) / "threshold.json"
    root = find_dataset_root(args.data_dir or resolve_data_dir(run_cfg))
    if root is None:
        raise SystemExit("Dataset not found")
    out = args.out or Path("outputs/eval") / run_name / "test"
    log = get_logger("evaluate", log_file=out / "evaluate.log")
    log.info("run %s | threshold file %s | data %s", run_dir, threshold, root)

    evaluate_test(
        run_dir, root, threshold, load_config(args.config), out, log,
        rerun_reason=args.rerun_reason,
        require_preregistration=not args.skip_preregistration_check,
        num_workers=args.num_workers,
    )  # fmt: skip
    dest = results_dir_for(run_name)
    print(
        f"\nDone. Copy the results next to the threshold and commit:\n"
        f"  cp {out}/* {dest}/\n"
        f"  git add {dest} && git commit -m 'phase3: test results' && git push"
    )


if __name__ == "__main__":
    main()
