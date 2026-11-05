"""Phase 6: Grad-CAM heatmaps and localisation scores for one split.

Run validation first (it also runs the randomised-weights sanity check), look
at it, then test. The test run refuses to start unless configs/explain.yaml is
committed unchanged, so the analysis cannot be adjusted after seeing test maps.

Writes to --out (default outputs/explain/<run_name>/<split>):
    summary.json                    hit rates with 95% CIs, baselines, sanity check, protocol
    report.md                       the same as tables
    localization_per_image.csv      every fractured image: hit, energy share, box area share
    cams.npz                        raw Grad-CAM maps (all images, every layer) + geometry
    figures/<split>_*.png           pre-declared galleries and the pointing-game chart

Usage (Kaggle, GPU):
    python scripts/explain.py --find-run p5_convnext_tiny_s44 --split val
    python scripts/explain.py --find-run p5_convnext_tiny_s44 --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracdet.config import load_config
from fracdet.eval.pipeline import find_run
from fracdet.explain.pipeline import explain_split
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir

DEFAULT_SEARCH = ["outputs/runs", "/kaggle/input"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--run-dir", type=Path)
    where.add_argument("--find-run", metavar="RUN_NAME", help="default: run_name in the config")
    parser.add_argument("--config", default="configs/explain.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--skip-preregistration-check",
        action="store_true",
        help="dry runs only: lets the test split run with an uncommitted plan",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = args.run_dir or find_run(args.find_run or cfg["run_name"], DEFAULT_SEARCH)
    run_cfg = load_config(run_dir / "config.yaml")
    run_name = run_cfg.get("run_name", run_dir.name)
    root = find_dataset_root(args.data_dir or resolve_data_dir(run_cfg))
    if root is None:
        raise SystemExit("Dataset not found")
    out = args.out or Path("outputs/explain") / run_name / args.split
    log = get_logger("explain", log_file=out / "explain.log")
    log.info("run %s | split %s | data %s", run_dir, args.split, root)

    explain_split(
        run_dir, root, cfg, args.split, out, log,
        config_path=args.config,
        require_preregistration=not args.skip_preregistration_check,
        num_workers=args.num_workers,
    )  # fmt: skip
    print(f"\nDone: {out}/report.md")


if __name__ == "__main__":
    main()
