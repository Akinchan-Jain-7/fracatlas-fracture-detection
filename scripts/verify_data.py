"""Verify an extracted FracAtlas copy: layout, CSV <-> image matching, counts.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --data-dir /kaggle/input/fracatlas

Exits with code 1 if any error is found, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fracdet.config import load_config
from fracdet.data.verify import verify_dataset
from fracdet.paths import resolve_data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", type=Path, help="Override the configured data folder")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    base = args.data_dir or resolve_data_dir(cfg)
    report = verify_dataset(
        base,
        expected_num_images=cfg["data"].get("expected_num_images"),
        expected_num_fractured=cfg["data"].get("expected_num_fractured"),
    )

    print("=" * 60)
    print(f"Searched: {base}")
    print(f"Dataset root: {report.root}")
    for key, value in report.stats.items():
        print(f"  {key}: {value}")
    for w in report.warnings:
        print(f"WARNING: {w}")
    for e in report.errors:
        print(f"ERROR:   {e}")
    print("=" * 60)
    print("RESULT:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
