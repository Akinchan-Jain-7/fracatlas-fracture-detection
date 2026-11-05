"""Phase 7: train the fracture detector (configs/detect.yaml) for one seed.

Usage:
    python scripts/train_detector.py --seed 42
    python scripts/train_detector.py --seed 43 --set train.epochs=2   # quick check

Creates outputs/runs/<timestamp>_<run_name>_s<seed>/ (see src/fracdet/detect/train.py).
On a 2-GPU machine, run one seed per GPU: CUDA_VISIBLE_DEVICES=0 ... and =1 ....
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fracdet.config import load_config, save_config
from fracdet.detect.train import fit_detector
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/detect.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, [*args.set, f"seed={args.seed}"])
    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        raise SystemExit("Dataset not found; run scripts/download_data.py first")
    name = f"{cfg['run_name']}_s{args.seed}"
    run_dir = Path(cfg["paths"]["output_dir"]) / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    run_dir.mkdir(parents=True)
    cfg["run_name"] = name
    save_config(cfg, run_dir / "config.yaml")
    log = get_logger(f"detect_s{args.seed}", log_file=run_dir / "train.log")
    log.info("run dir %s", run_dir)
    summary = fit_detector(cfg, root, run_dir, log)
    print("SUMMARY_JSON " + json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
