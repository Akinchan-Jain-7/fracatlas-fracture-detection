"""Exploratory data analysis: summary JSON + figures.

Usage:
    python scripts/eda.py                               # -> outputs/eda/
    python scripts/eda.py --out docs/figures/phase-1    # figures used in the docs

Figures:
    region_fracture_rate.png   positives per body region (and how few hip/shoulder has)
    shortcut_attributes.png    non-fracture attributes that predict the label
    fracture_rate_by_id.png    labels cluster by image ID (same-patient evidence)
    duplicate_distances.png    nearest-neighbour hash distances and the two thresholds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on Kaggle/CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fracdet.config import load_config  # noqa: E402
from fracdet.data.duplicates import compute_hashes, nearest_neighbor_distance  # noqa: E402
from fracdet.data.eda import (  # noqa: E402
    add_image_properties,
    fracture_rate_by_id,
    rate_table,
    summarize_dataset,
    yolo_box_counts,
)
from fracdet.data.integrity import find_corrupt  # noqa: E402
from fracdet.data.metadata import load_metadata  # noqa: E402
from fracdet.paths import find_dataset_root, resolve_data_dir  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode) + ink tokens.
SERIES = "#2a78d6"
ACCENT = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 10,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _reference_line(ax, rate: float, label: str) -> None:
    """Dashed overall-rate line, named in a legend so it never collides with labels."""
    ax.axvline(rate, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label=label)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2, fontsize=9)


def _rate_bars(ax, labels, rates, counts, totals) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rates, height=0.6, color=SERIES)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    # Room for the value labels, but no ticks beyond 100%.
    ax.set_xlim(0, min(max(rates) + 0.3, 1.3))
    ax.set_xticks([t for t in np.arange(0, 1.01, 0.2) if t <= ax.get_xlim()[1]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for yi, r, c, t in zip(y, rates, counts, totals, strict=True):
        ax.text(r + 0.01, yi, f"{r:.0%}  ({c:,} of {t:,})", va="center", color=INK, fontsize=9)


def plot_region(meta: pd.DataFrame, overall: float, out: Path) -> None:
    t = rate_table(meta, "region")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _rate_bars(ax, t.index.tolist(), t.fracture_rate, t.fractured, t.n)
    _reference_line(ax, overall, f"overall {overall:.1%}")
    ax.set_xlabel("Fracture rate")
    ax.set_title("Hip and shoulder have only 10 fractures each: too few to evaluate per region")
    fig.savefig(out)
    plt.close(fig)


def plot_shortcuts(meta: pd.DataFrame, overall: float, out: Path) -> None:
    rows = [
        ("Surgical hardware", meta.hardware == 1),
        ("No hardware", meta.hardware == 0),
        ("Large image (>=1000 px)", meta.size_group == "large"),
        ("Small image", meta.size_group == "small"),
        ("Grayscale file (mode L)", meta["mode"] == "L"),
        ("RGB file", meta["mode"] == "RGB"),
        ("Truncated file", meta.corrupt),
        ("Intact file", ~meta.corrupt),
    ]
    labels = [r[0] for r in rows]
    counts = [int(meta.label[m].sum()) for _, m in rows]
    totals = [int(m.sum()) for _, m in rows]
    rates = [c / t for c, t in zip(counts, totals, strict=True)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _rate_bars(ax, labels, rates, counts, totals)
    _reference_line(ax, overall, f"overall {overall:.1%}")
    ax.set_xlabel("Fracture rate")
    ax.set_title("Attributes that predict the label without being a fracture")
    fig.savefig(out)
    plt.close(fig)


def plot_rate_by_id(meta: pd.DataFrame, overall: float, out: Path, window: int = 101) -> None:
    df = fracture_rate_by_id(meta, window=window)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(df.id_num, df.rolling_rate, color=SERIES, linewidth=2)
    ax.axhline(
        overall, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label=f"overall {overall:.1%}"
    )
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Image ID number")
    ax.set_ylabel(f"Fracture rate ({window}-image window)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Labels cluster by image ID: neighbouring IDs are not independent")
    fig.savefig(out)
    plt.close(fig)


def plot_duplicate_distances(nn: np.ndarray, dup_t: int, group_t: int, out: Path) -> None:
    bins = np.arange(0, 81)
    counts = np.bincount(np.clip(nn, 0, 80), minlength=81)[:80]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(bins[:-1], counts, width=0.85, color=SERIES)
    for t, label, height in [(dup_t, "same picture", 0.92), (group_t, "keep in one split", 0.76)]:
        ax.axvline(t + 0.5, color=ACCENT, linewidth=1.5)
        ax.text(t + 1.2, counts.max() * height, f"<= {t}: {label}", color=INK, fontsize=9)
    ax.set_xlabel("Hamming distance to nearest other image (256-bit dHash; > 80 not shown)")
    ax.set_ylabel("Images")
    ax.grid(axis="x", visible=False)
    ax.set_title("A cluster of near-identical images sits far left of the bulk")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    root = find_dataset_root(args.data_dir or resolve_data_dir(cfg))
    if root is None:
        raise SystemExit("Dataset not found; run scripts/download_data.py first")
    meta = add_image_properties(load_metadata(root))
    meta["yolo_boxes"] = yolo_box_counts(root, meta["image_id"])
    meta["corrupt"] = meta["path"].isin(find_corrupt(meta["path"].tolist()))

    summary = summarize_dataset(meta)
    hashes = compute_hashes(meta["path"].tolist(), hash_size=cfg["split"]["hash_size"])
    nn = nearest_neighbor_distance(hashes)
    summary["nearest_neighbor_distance_counts"] = {
        int(k): int(v) for k, v in enumerate(np.bincount(nn)) if v
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _style()
    overall = summary["fracture_rate"]
    plot_region(meta, overall, args.out / "region_fracture_rate.png")
    plot_shortcuts(meta, overall, args.out / "shortcut_attributes.png")
    plot_rate_by_id(meta, overall, args.out / "fracture_rate_by_id.png")
    s = cfg["split"]
    plot_duplicate_distances(
        nn, s["duplicate_threshold"], s["group_threshold"], args.out / "duplicate_distances.png"
    )

    print(
        f"images={summary['n_images']} fractured={summary['n_fractured']} "
        f"rate={summary['fracture_rate']:.2%} majority-class accuracy="
        f"{summary['majority_baseline_accuracy']:.2%}"
    )
    print("yolo:", summary["yolo"])
    print(f"wrote {args.out}/summary.json and 4 figures")


if __name__ == "__main__":
    main()
