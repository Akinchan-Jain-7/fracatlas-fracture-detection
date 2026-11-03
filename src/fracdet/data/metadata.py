"""Load FracAtlas into one clean table: one row per image, with a resolved path.

Every later phase (EDA, splitting, training, evaluation) starts from
``load_metadata`` instead of parsing the CSV itself. That gives one place
to apply the dataset's rules:

- the label comes from the CSV, and the file from the folder matching that
  label (this is what makes the 2 stale copies in Non_fractured/ harmless);
- the 0/1 region and view flags are summarised into readable keys
  (``region``, ``view``) used for stratifying the split and reporting per-region
  metrics later.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fracdet.data.schema import (
    BODY_PARTS,
    CSV_NAME,
    ID_COLUMN,
    IMAGES_DIR,
    LABEL_COLUMN,
    LABEL_FOLDERS,
    REGION_COLUMNS,
    VIEW_COLUMNS,
)

MIXED = "mixed"


def region_key(row: pd.Series) -> str:
    """One region name per image, for stratification and per-region reporting.

    Exactly one body part flagged (and not "mixed") -> that part's name.
    Several parts or the "mixed" flag -> "mixed". Nothing flagged -> "unknown".
    """
    parts = [p for p in BODY_PARTS if row[p] == 1]
    if row.get(MIXED, 0) == 1 or len(parts) > 1:
        return MIXED
    return parts[0] if parts else "unknown"


def view_key(row: pd.Series) -> str:
    views = [v for v in VIEW_COLUMNS if row[v] == 1]
    if len(views) == 1:
        return views[0]
    return "multiple" if views else "none"


def load_metadata(root: str | Path, check_files: bool = True) -> pd.DataFrame:
    """Return a DataFrame with the CSV columns plus ``label``, ``path``, ``region``, ``view``.

    Args:
        root: the dataset root (the folder containing ``dataset.csv``).
        check_files: raise if any resolved image path is missing.
    """
    root = Path(root)
    df = pd.read_csv(root / CSV_NAME)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    flag_cols = [*REGION_COLUMNS, *VIEW_COLUMNS, LABEL_COLUMN, "hardware", "multiscan"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df["label"] = df[LABEL_COLUMN]
    df["path"] = [
        str(root / IMAGES_DIR / LABEL_FOLDERS[label] / image_id)
        for image_id, label in zip(df[ID_COLUMN], df["label"], strict=True)
    ]
    df["region"] = df.apply(region_key, axis=1)
    df["view"] = df.apply(view_key, axis=1)

    if check_files:
        missing = [p for p in df["path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images missing from their label folder, e.g. {missing[:3]}. "
                "Run scripts/verify_data.py for details."
            )
    return df.reset_index(drop=True)
