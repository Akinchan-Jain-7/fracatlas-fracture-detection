"""Phase 7: evaluate a trained detector on one split, with the Grad-CAM box baseline.

The test split runs only for the pre-registered seed (configs/detect.yaml
reported_seed), only with the plan committed unchanged, and only once.

Writes to --out (default outputs/detect_eval/<run_name>/<split>):
    summary.json, report.md         detector and baseline metrics with 95% CIs, protocol
    <split>_per_image.csv           per image: top box hit / IoU for both methods, classifier
    <split>_detections.csv          every detector box (normalised) with its score
    figures/                        galleries (test only)

Usage (Kaggle, GPU):
    python scripts/evaluate_detector.py --seed 42 --split val
    python scripts/evaluate_detector.py --seed 42 --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracdet.config import load_config
from fracdet.detect.evaluate import SEARCH, evaluate_detector
from fracdet.eval.pipeline import find_run
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/detect.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rerun-reason")
    parser.add_argument("--skip-preregistration-check", action="store_true",
                        help="dry runs only")  # fmt: skip
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = f"{cfg['run_name']}_s{args.seed}"
    run_dir = args.run_dir or find_run(name, SEARCH)
    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        raise SystemExit("Dataset not found")
    out = args.out or Path("outputs/detect_eval") / name / args.split
    log = get_logger("detect_eval", log_file=out / "evaluate.log")
    log.info("run %s | split %s | data %s", run_dir, args.split, root)
    evaluate_detector(
        run_dir, root, cfg, args.split, out, log, config_path=args.config,
        rerun_reason=args.rerun_reason,
        require_preregistration=not args.skip_preregistration_check,
        num_workers=args.num_workers,
    )  # fmt: skip
    print(f"\nDone: {out}/report.md")


if __name__ == "__main__":
    main()
