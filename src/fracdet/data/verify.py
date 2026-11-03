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

from fracdet.data.download import md5_of_file
from fracdet.data.schema import (
    ID_COLUMN,
    LABEL_COLUMN,
    LABEL_FOLDERS,
    REGION_COLUMNS,
    VIEW_COLUMNS,
    YOLO_DIR,
)
from fracdet.paths import MARKER_FILE, find_dataset_root

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Columns described in the FracAtlas paper. Missing ones are reported as
# warnings, not errors, until the real CSV confirms the exact names.
EXPECTED_COLUMNS = [
    ID_COLUMN,
    *REGION_COLUMNS,
    "hardware",
    "multiscan",
    LABEL_COLUMN,
    "fracture_count",
    *VIEW_COLUMNS,
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

    dup_names = {n for n, ps in images.items() if len(ps) > 1}

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
        dup_names -= _check_label_folders(df, images, report)
        _annotation_coverage(root, df, report)

    if dup_names:
        report.warnings.append(
            f"{len(dup_names)} file names appear in more than one folder, "
            f"e.g. {sorted(dup_names)[:3]}"
        )

    regions = {c: int(df[c].sum()) for c in REGION_COLUMNS if c in df.columns}
    if regions:
        report.stats["region_counts"] = regions
    return report


def _check_label_folders(
    df: pd.DataFrame, images: dict[str, list[Path]], report: VerifyReport
) -> set[str]:
    """Check every image exists in the folder that matches its CSV label.

    The CSV is the source of truth for labels; folders are a convenience copy.
    FracAtlas v7 ships 2 images (IMG0003375/6) labelled fractured, with boxes,
    that ALSO sit in Non_fractured/ as byte-identical leftovers. Those are
    "stale copies": harmless as long as loaders pick the copy in the label's
    folder. A row with NO copy in its label folder, or a stale copy whose bytes
    differ, is a real conflict and an error.

    Returns the names explained as stale copies.
    """
    stale: set[str] = set()
    wrong_folder: list[str] = []
    conflicting: list[str] = []
    for image_id, label in zip(df[ID_COLUMN].astype(str), df[LABEL_COLUMN], strict=True):
        paths = images.get(image_id, [])
        if not paths:
            continue  # already reported as a missing file
        expected = LABEL_FOLDERS.get(int(label))
        good = [p for p in paths if p.parent.name == expected]
        if not good:
            wrong_folder.append(image_id)
        elif len(paths) > 1:
            if len({md5_of_file(p) for p in paths}) == 1:
                stale.add(image_id)
            else:
                conflicting.append(image_id)

    report.stats["stale_copies"] = sorted(stale)
    if stale:
        report.warnings.append(
            f"{len(stale)} images also have identical stale copies outside their label folder "
            f"(use the copy in the label folder): {sorted(stale)[:5]}"
        )
    if wrong_folder:
        report.errors.append(
            f"{len(wrong_folder)} images are not in the folder matching their CSV label, "
            f"e.g. {wrong_folder[:3]}"
        )
    if conflicting:
        report.errors.append(
            f"{len(conflicting)} images have copies with DIFFERENT content, e.g. {conflicting[:3]}"
        )
    return stale


def _annotation_coverage(root: Path, df: pd.DataFrame, report: VerifyReport) -> None:
    """Report how many fractured / non-fractured images have a YOLO box file.

    Expected: every fractured image has one; non-fractured images have none.
    Reported as stats (not errors) because Phase 7 decides how to use them.
    """
    yolo_dir = root / YOLO_DIR
    if not yolo_dir.is_dir():
        report.warnings.append(f"No {YOLO_DIR} folder found; skipping annotation check")
        return
    with_box = {p.stem for p in yolo_dir.glob("*.txt")}
    stems = df[ID_COLUMN].astype(str).map(lambda s: Path(s).stem)
    is_frac = df[LABEL_COLUMN] == 1
    report.stats["yolo_files_fractured"] = int(stems[is_frac].isin(with_box).sum())
    report.stats["yolo_files_non_fractured"] = int(stems[~is_frac].isin(with_box).sum())
