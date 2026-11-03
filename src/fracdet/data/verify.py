"""Sanity-check an extracted FracAtlas copy before any experiment touches it.

Catches the boring failures that otherwise surface hours later as weird metrics:
a partial upload, a CSV row whose image is missing, images with no label, the
wrong dataset version. It discovers the layout instead of hard-coding folder
names, and reports what it found so the Phase 1 code can be written against
the real structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fracdet.paths import MARKER_FILE, find_dataset_root

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ID_COLUMN = "image_id"
LABEL_COLUMN = "fractured"
REGION_COLUMNS = ["hand", "leg", "hip", "shoulder", "mixed"]
# Columns described in the FracAtlas paper. Missing ones are reported as
# warnings, not errors, until the real CSV confirms the exact names.
EXPECTED_COLUMNS = [
    ID_COLUMN,
    *REGION_COLUMNS,
    "hardware",
    "multiscan",
    LABEL_COLUMN,
    "fracture_count",
    "frontal",
    "lateral",
    "oblique",
]


@dataclass
class VerifyReport:
    root: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_dataset(
    base_dir: str | Path,
    expected_num_images: int | None = None,
    expected_num_fractured: int | None = None,
) -> VerifyReport:
    report = VerifyReport()
    root = find_dataset_root(base_dir)
    if root is None:
        report.errors.append(f"No {MARKER_FILE} found under {base_dir} (searched 4 levels deep)")
        return report
    report.root = root

    df = pd.read_csv(root / MARKER_FILE)
    report.stats["csv_rows"] = len(df)
    report.stats["csv_columns"] = list(df.columns)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.warnings.append(f"CSV lacks expected columns: {missing_cols}")

    # Map file name -> paths. Names, not full paths, because the CSV stores bare
    # file names and the images sit in label-named subfolders.
    images: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault(p.name, []).append(p)
    report.stats["image_files"] = sum(len(v) for v in images.values())
    folders = sorted({str(p.parent.relative_to(root)) for ps in images.values() for p in ps})
    report.stats["image_folders"] = folders

    dup_names = [n for n, ps in images.items() if len(ps) > 1]
    if dup_names:
        report.warnings.append(
            f"{len(dup_names)} file names appear in more than one folder, e.g. {dup_names[:3]}"
        )

    if ID_COLUMN not in df.columns:
        report.errors.append(f"CSV has no '{ID_COLUMN}' column; cannot match rows to images")
        return report

    ids = set(df[ID_COLUMN].astype(str))
    if df[ID_COLUMN].duplicated().any():
        report.errors.append(f"{int(df[ID_COLUMN].duplicated().sum())} duplicate image_id rows")
    missing_files = sorted(ids - images.keys())
    unlabeled_files = sorted(images.keys() - ids)
    if missing_files:
        report.errors.append(
            f"{len(missing_files)} CSV rows have no image file, e.g. {missing_files[:3]}"
        )
    if unlabeled_files:
        report.warnings.append(
            f"{len(unlabeled_files)} image files not listed in CSV, e.g. {unlabeled_files[:3]}"
        )

    if expected_num_images is not None and len(df) != expected_num_images:
        report.errors.append(f"Expected {expected_num_images} rows in CSV, found {len(df)}")

    if LABEL_COLUMN in df.columns:
        n_frac = int(df[LABEL_COLUMN].sum())
        report.stats["fractured"] = n_frac
        report.stats["fractured_fraction"] = round(n_frac / max(len(df), 1), 4)
        if expected_num_fractured is not None and n_frac != expected_num_fractured:
            report.errors.append(f"Expected {expected_num_fractured} fractured, found {n_frac}")

    regions = {c: int(df[c].sum()) for c in REGION_COLUMNS if c in df.columns}
    if regions:
        report.stats["region_counts"] = regions
    return report
