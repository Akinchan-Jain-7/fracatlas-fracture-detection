"""Download + extract FracAtlas into the configured data folder.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-dir /kaggle/working/data

Not needed on Kaggle if FracAtlas was added as a Kaggle Dataset (read-only input).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fracdet.config import load_config
from fracdet.data.download import prepare_dataset
from fracdet.paths import resolve_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", type=Path, help="Override the configured data folder")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    data_dir = args.data_dir or resolve_data_dir(cfg)
    root = prepare_dataset(cfg, data_dir)
    print(f"\nDATASET_ROOT={root}")


if __name__ == "__main__":
    main()
