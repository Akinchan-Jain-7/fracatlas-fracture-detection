"""(Re)finalise a training run from its saved files, without retraining.

Re-scores validation with the run's best.pt and writes val_predictions.csv and
summary.json. Use it when a run's last step failed after training finished,
or to regenerate those files with newer evaluation code.

Usage:
    python scripts/finalize_run.py outputs/runs/20261103-153336_baseline_resnet50
    python scripts/finalize_run.py --latest baseline_resnet50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fracdet.config import load_config
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir
from fracdet.train.fit import finalize_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--latest", metavar="RUN_NAME")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    run = args.run_dir
    if run is None:
        matches = sorted(Path("outputs/runs").glob(f"*_{args.latest}"))
        if not matches:
            raise SystemExit(f"No runs named {args.latest!r}")
        run = matches[-1]
    cfg = load_config(run / "config.yaml")  # the exact config that trained this run
    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        raise SystemExit("Dataset not found")
    log = get_logger("finalize", log_file=run / "train.log")
    summary = finalize_run(cfg, root, run, log)
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
