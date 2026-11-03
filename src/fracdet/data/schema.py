"""Names from the FracAtlas archive, in one place.

Every other module imports these instead of repeating string literals, so a
renamed column or folder is a one-line change. Confirmed against the real
Figshare v7 archive on 2026-11-03.
"""

from __future__ import annotations

from pathlib import Path

CSV_NAME = "dataset.csv"
ID_COLUMN = "image_id"
LABEL_COLUMN = "fractured"

BODY_PARTS = ["hand", "leg", "hip", "shoulder"]
# "mixed" is a flag of its own in the CSV; EDA shows how it combines with the parts.
REGION_COLUMNS = [*BODY_PARTS, "mixed"]
VIEW_COLUMNS = ["frontal", "lateral", "oblique"]

IMAGES_DIR = Path("images")
# The CSV label is the source of truth; this maps it to the folder holding the
# canonical copy (Non_fractured/ also contains 2 stale copies, see Phase 0 docs).
LABEL_FOLDERS = {1: "Fractured", 0: "Non_fractured"}
YOLO_DIR = Path("Annotations") / "YOLO"
