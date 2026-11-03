"""Create the leakage-safe train/val/test split and write it to ``splits/``.

Usage:
    python scripts/make_split.py
    python scripts/make_split.py --set split.seed=7 split.name=split_seed7

Writes (committed to git, so every machine uses the identical split):
    splits/<name>.csv           one row per image: split, exclude_reason, groups
    splits/<name>_summary.json  config used, counts, per-split balance, data hash

Refuses to overwrite an existing split unless --force: a test set must not
change silently once results have been reported on it.

    python scripts/make_split.py --verify
rebuilds the split from the raw data and checks it matches the committed file
exactly (run it on any new machine, e.g. Kaggle, to prove reproducibility).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fracdet.config import load_config
from fracdet.data.download import md5_of_file
from fracdet.data.duplicates import compute_hashes, compute_md5s
from fracdet.data.integrity import find_corrupt
from fracdet.data.metadata import load_metadata
from fracdet.data.schema import CSV_NAME
from fracdet.data.split import SplitConfig, check_split, make_split, summarize
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root, resolve_data_dir
from fracdet.seed import seed_everything

logger = get_logger("make_split")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    parser.add_argument(
        "--verify", action="store_true", help="rebuild and compare with the committed split"
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    split_cfg = SplitConfig.from_dict(cfg["split"])
    out_dir = Path(cfg["paths"]["splits_dir"])
    csv_path = out_dir / f"{cfg['split']['name']}.csv"
    if args.verify and not csv_path.exists():
        logger.error("Nothing to verify: %s does not exist", csv_path)
        return 1
    if csv_path.exists() and not (args.force or args.verify):
        logger.error("%s exists. Use --force only if no results depend on it.", csv_path)
        return 1

    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        logger.error("Dataset not found; run scripts/download_data.py first")
        return 1
    meta = load_metadata(root)
    paths = meta["path"].tolist()

    logger.info("Checking %d images decode fully", len(paths))
    corrupt = meta["path"].isin(find_corrupt(paths)).to_numpy()
    logger.info("Hashing images (dHash %dx%d) and computing MD5s", *[cfg["split"]["hash_size"]] * 2)
    hashes = compute_hashes(paths, hash_size=cfg["split"]["hash_size"])
    md5s = compute_md5s(paths)

    df = make_split(meta, hashes, md5s, corrupt, split_cfg)
    check_split(df)
    summary = summarize(df)
    summary["config"] = cfg["split"]
    summary["dataset_csv_md5"] = md5_of_file(root / CSV_NAME)

    if args.verify:
        committed = pd.read_csv(csv_path, keep_default_na=False)
        cols = ["image_id", "split", "exclude_reason"]
        same = committed[cols].equals(df[cols].astype(committed[cols].dtypes.to_dict()))
        print("VERIFY:", "MATCH" if same else "MISMATCH", f"({csv_path})")
        if not same:
            diff = (committed["split"] != df["split"]).sum()
            print(f"{diff} images assigned differently")
        return 0 if same else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_path = out_dir / f"{cfg['split']['name']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("n_total", "n_used", "excluded")}, indent=2))
    for name, s in summary["splits"].items():
        print(
            f"{name:5}  n={s['n']:5}  ({s['fraction']:.1%})  "
            f"fractured={s['fractured']:4} ({s['fracture_rate']:.1%})  {s['fractured_by_region']}"
        )
    print(f"\nWrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
