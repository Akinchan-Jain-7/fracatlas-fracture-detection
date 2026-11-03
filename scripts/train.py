"""Train a fracture classifier from a config file.

Usage:
    python scripts/train.py --config configs/baseline_resnet50.yaml
    python scripts/train.py --config configs/baseline_resnet50.yaml --set train.epochs=2

Creates outputs/runs/<timestamp>_<run_name>/ (see src/fracdet/train/fit.py for
its contents) and prints the summary as JSON on the last line.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fracdet.config import load_config, save_config
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir
from fracdet.train.fit import fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        raise SystemExit("Dataset not found; run scripts/download_data.py first")

    name = cfg.get("run_name", Path(args.config).stem)
    run_dir = Path(cfg["paths"]["output_dir"]) / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    log = get_logger("train", log_file=run_dir / "train.log")
    log.info("run dir %s", run_dir)

    summary = fit(cfg, root, run_dir, log)
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
