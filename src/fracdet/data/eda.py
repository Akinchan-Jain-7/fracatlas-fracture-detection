"""Exploratory statistics for FracAtlas, as plain data (dict / DataFrame).

Kept separate from plotting so the numbers are unit-tested and the same
numbers feed the figures, the docs and the summary JSON.

The focus is on what can mislead a model or a metric:
- class balance (accuracy is meaningless at ~18% positives),
- per-region sample sizes (some are too small to evaluate),
- attributes that predict the label without being a fracture ("shortcuts"):
  surgical hardware, image size/colour mode, corrupt files,
- annotation consistency (YOLO boxes vs the CSV's fracture_count).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from fracdet.data.schema import ID_COLUMN, YOLO_DIR

LARGE_SIDE = 1000  # px; FracAtlas sizes are bimodal: ~454 px vs >= 2140 px


def add_image_properties(meta: pd.DataFrame) -> pd.DataFrame:
    """Width, height, colour mode and size group, read from file headers only."""
    out = meta.copy()
    widths, heights, modes = [], [], []
    for path in out["path"]:
        with Image.open(path) as im:  # header only; pixels are not decoded
            widths.append(im.size[0])
            heights.append(im.size[1])
            modes.append(im.mode)
    out["width"], out["height"], out["mode"] = widths, heights, modes
    out["size_group"] = (out[["width", "height"]].max(axis=1) >= LARGE_SIDE).map(
        {True: "large", False: "small"}
    )
    return out


def yolo_box_counts(root: str | Path, image_ids: pd.Series) -> pd.Series:
    """Number of boxes (non-empty lines) in each image's YOLO file; -1 if no file."""
    yolo_dir = Path(root) / YOLO_DIR
    counts = []
    for image_id in image_ids:
        f = yolo_dir / f"{Path(image_id).stem}.txt"
        if not f.is_file():
            counts.append(-1)
        else:
            counts.append(sum(1 for line in f.read_text().splitlines() if line.strip()))
    return pd.Series(counts, index=image_ids.index, name="yolo_boxes")


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """n, fractured count and fracture rate per value of ``by``."""
    g = df.groupby(by)["label"]
    t = pd.DataFrame({"n": g.size(), "fractured": g.sum()})
    t["fracture_rate"] = (t["fractured"] / t["n"]).round(4)
    return t.sort_values("n", ascending=False)


def summarize_dataset(meta: pd.DataFrame) -> dict:
    """Headline numbers. ``meta`` needs add_image_properties + yolo_boxes + corrupt."""
    frac = meta["label"] == 1
    boxes = meta["yolo_boxes"]
    summary: dict = {
        "n_images": len(meta),
        "n_fractured": int(frac.sum()),
        "fracture_rate": round(float(frac.mean()), 4),
        "majority_baseline_accuracy": round(float(1 - frac.mean()), 4),
        "by_region": rate_table(meta, "region").to_dict("index"),
        "by_view": rate_table(meta, "view").to_dict("index"),
        "fracture_count_distribution": meta.loc[frac, "fracture_count"]
        .value_counts()
        .sort_index()
        .to_dict(),
        # "WxH" string keys, largest group first
        "image_sizes": {
            f"{w}x{h}": int(n)
            for (w, h), n in meta.groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
            .items()
        },
        "yolo": {
            "fractured_with_boxes": int((frac & (boxes > 0)).sum()),
            "fractured_without_boxes": int((frac & (boxes <= 0)).sum()),
            "non_fractured_with_boxes": int((~frac & (boxes > 0)).sum()),
            "non_fractured_empty_file": int((~frac & (boxes == 0)).sum()),
            "box_count_matches_fracture_count": int(
                (frac & (boxes == meta["fracture_count"])).sum()
            ),
        },
        "shortcut_candidates": {},
    }
    for col in ["hardware", "multiscan", "size_group", "mode", "corrupt"]:
        summary["shortcut_candidates"][col] = rate_table(meta, col).to_dict("index")
    return _jsonable(summary)


def fracture_rate_by_id(meta: pd.DataFrame, window: int = 101) -> pd.DataFrame:
    """Centred rolling fracture rate over images sorted by ID number."""
    df = meta[[ID_COLUMN, "label"]].copy()
    df["id_num"] = df[ID_COLUMN].str.extract(r"(\d+)", expand=False).astype(int)
    df = df.sort_values("id_num").reset_index(drop=True)
    df["rolling_rate"] = df["label"].rolling(window, center=True, min_periods=window // 2).mean()
    return df


def _jsonable(obj):
    """Convert numpy scalars / non-str keys so json.dumps works."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
