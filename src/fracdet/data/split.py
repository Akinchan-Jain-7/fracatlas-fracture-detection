"""Leakage-safe train/val/test split for FracAtlas.

Data leakage = information about a test image reaching training, so the test
score overstates real-world performance. In FracAtlas there are three routes:

1. The same picture twice (byte copies and re-encoded near-copies, some with
   a marker added). One copy in train, one in test = memorised, not detected.
2. The same patient in different views. FracAtlas has no patient IDs, but
   neighbouring image IDs behave like one study: after a fractured image the
   next ID is fractured 74% of the time (base rate 18%). Frontal in train +
   lateral in test of the same fracture is leakage.
3. Visual shortcuts: truncated files decode with grey bars and are all
   negatives, so the bar itself would "predict" the label.

What this module does:
- Exclude: corrupt files; every extra copy of a duplicated image (keep one);
  every duplicate group whose copies disagree on the label (unknowable truth).
- Group: near-duplicates (looser threshold) + runs of consecutive IDs with
  the same label and region. A group always lands in one split.
- Split: StratifiedGroupKFold into ``n_folds`` equal folds, stratified on
  label x region, then whole folds are assigned to train/val/test.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fracdet.data.duplicates import exact_duplicate_pairs, group_pairs, near_duplicate_pairs
from fracdet.data.schema import ID_COLUMN

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    n_folds: int = 20
    seed: int = 42
    # Hamming distances on 256-bit dHash, chosen by inspecting real pairs (Phase 1 doc):
    duplicate_threshold: int = 11  # <= this: same picture
    group_threshold: int = 20  # <= this: similar enough to keep in one split
    group_consecutive_ids: bool = True

    def fold_counts(self) -> dict[str, int]:
        counts = {s: round(getattr(self, s) * self.n_folds) for s in SPLITS}
        for s in SPLITS:
            if abs(counts[s] / self.n_folds - getattr(self, s)) > 1e-9:
                raise ValueError(f"{s}={getattr(self, s)} is not a multiple of 1/{self.n_folds}")
        if sum(counts.values()) != self.n_folds:
            raise ValueError("train + val + test must equal 1")
        return counts

    @classmethod
    def from_dict(cls, d: dict) -> SplitConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def image_number(image_ids: pd.Series) -> np.ndarray:
    """'IMG0003375.jpg' -> 3375."""
    return image_ids.str.extract(r"(\d+)", expand=False).astype(int).to_numpy()


def consecutive_run_pairs(meta: pd.DataFrame) -> list[tuple[int, int]]:
    """Pairs of row indices with consecutive IDs and the same label and region."""
    num = image_number(meta[ID_COLUMN])
    order = np.argsort(num, kind="stable")
    pairs = []
    for a, b in zip(order[:-1], order[1:], strict=True):
        if (
            num[b] - num[a] == 1
            and meta.at[a, "label"] == meta.at[b, "label"]
            and meta.at[a, "region"] == meta.at[b, "region"]
        ):
            pairs.append((int(a), int(b)))
    return pairs


def find_exclusions(meta: pd.DataFrame, dup_group: np.ndarray, corrupt: np.ndarray) -> pd.Series:
    """Reason each image is excluded ('' = kept). First matching reason wins."""
    reason = pd.Series("", index=meta.index, dtype=object)
    df = pd.DataFrame({"g": dup_group, "label": meta["label"], "corrupt": corrupt})
    conflict = df.groupby("g")["label"].transform("nunique") > 1
    reason[conflict] = "label_conflict"
    reason[(reason == "") & corrupt] = "corrupt"

    # Keep one copy per duplicate group: the first intact one by ID number.
    num = image_number(meta[ID_COLUMN])
    candidates = df[(reason == "") & (df.groupby("g")["g"].transform("size") > 1)]
    for _, members in candidates.groupby("g"):
        keep = members.index[np.argmin(num[members.index])]
        for idx in members.index:
            if idx != keep:
                reason[idx] = f"duplicate_of:{meta.at[keep, ID_COLUMN]}"
    return reason


def assign_folds(strata: Sequence[str], groups: Sequence[int], cfg: SplitConfig) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.full(len(strata), -1)
    with warnings.catch_warnings():
        # Fractured hip/shoulder have fewer members than folds; sklearn warns,
        # and we report the achieved per-split balance instead.
        warnings.filterwarnings("ignore", message="The least populated class")
        for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(strata)), strata, groups)):
            fold[idx] = k
    return fold


def make_split(
    meta: pd.DataFrame,
    hashes: np.ndarray,
    md5s: Sequence[str],
    corrupt: np.ndarray,
    cfg: SplitConfig,
) -> pd.DataFrame:
    """Return one row per image: split ('train'/'val'/'test'/'excluded') and why."""
    meta = meta.reset_index(drop=True)
    n = len(meta)
    corrupt = np.asarray(corrupt, dtype=bool)
    counts = cfg.fold_counts()

    near = near_duplicate_pairs(hashes, cfg.group_threshold)
    exact = exact_duplicate_pairs(md5s)
    dup_pairs = [(i, j) for i, j, d in near if d <= cfg.duplicate_threshold] + exact
    dup_group = group_pairs(n, dup_pairs)

    group_edges = [(i, j) for i, j, _ in near] + exact
    if cfg.group_consecutive_ids:
        group_edges += consecutive_run_pairs(meta)
    split_group = group_pairs(n, group_edges)

    reason = find_exclusions(meta, dup_group, corrupt)
    kept = (reason == "").to_numpy()

    strata = (meta["label"].astype(str) + "_" + meta["region"]).to_numpy()
    fold = np.full(n, -1)
    fold[kept] = assign_folds(strata[kept], split_group[kept], cfg)

    # Whole folds -> splits, in a seeded random order.
    fold_order = np.random.default_rng(cfg.seed).permutation(cfg.n_folds)
    fold_to_split = {}
    start = 0
    for s in ("test", "val", "train"):
        for f in fold_order[start : start + counts[s]]:
            fold_to_split[int(f)] = s
        start += counts[s]
    split = np.where(kept, [fold_to_split.get(int(f), "excluded") for f in fold], "excluded")

    return pd.DataFrame(
        {
            ID_COLUMN: meta[ID_COLUMN],
            "label": meta["label"],
            "region": meta["region"],
            "split": split,
            "exclude_reason": reason.to_numpy(),
            "dup_group": dup_group,
            "split_group": split_group,
        }
    )


def check_split(df: pd.DataFrame) -> None:
    """Raise if any leakage invariant is violated."""
    used = df[df["split"] != "excluded"]
    straddling = used.groupby("split_group")["split"].nunique()
    if (straddling > 1).any():
        raise AssertionError(f"{int((straddling > 1).sum())} groups span more than one split")
    dup = used.groupby("dup_group").size()
    if (dup > 1).any():
        raise AssertionError("a duplicate group still has more than one used image")
    if df[ID_COLUMN].duplicated().any():
        raise AssertionError("duplicate image ids in split table")


def summarize(df: pd.DataFrame) -> dict:
    used = df[df["split"] != "excluded"]
    out: dict = {
        "n_total": len(df),
        "n_used": len(used),
        "excluded": df.loc[df.split == "excluded", "exclude_reason"]
        .str.split(":")
        .str[0]
        .value_counts()
        .to_dict(),
        "splits": {},
    }
    for s in SPLITS:
        part = used[used["split"] == s]
        out["splits"][s] = {
            "n": len(part),
            "fraction": round(len(part) / max(len(used), 1), 4),
            "fractured": int(part["label"].sum()),
            "fracture_rate": round(float(part["label"].mean()), 4) if len(part) else None,
            "fractured_by_region": part[part.label == 1]["region"].value_counts().to_dict(),
        }
    return out
