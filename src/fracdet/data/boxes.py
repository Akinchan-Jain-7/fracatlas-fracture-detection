"""Fracture bounding boxes from the FracAtlas YOLO files.

Phase 1 checked the files: every fractured image has one box per fracture
(719 / 719 match ``fracture_count``), every normal image has an empty file.
Phase 6 uses the boxes as ground truth for "is the heatmap on the fracture?".
They never enter training: the classifier only ever sees the 0/1 label.

YOLO format, one line per box: ``class cx cy w h``, all normalised to 0..1 by
the image width and height. Normalised coordinates survive any resize, so the
same numbers place a box on the original X-ray or in the model's letterbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fracdet.data.schema import YOLO_DIR


@dataclass(frozen=True)
class Box:
    """A box in normalised image coordinates (0..1), corner form."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


def parse_yolo(text: str) -> list[Box]:
    """Boxes from the text of one YOLO file; blank lines are ignored.

    Coordinates are clipped to the image: a box drawn slightly past the edge
    would otherwise fall into the letterbox padding.
    """
    boxes = []
    for n, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"line {n}: expected 'class cx cy w h', got {line!r}")
        cx, cy, w, h = map(float, parts[1:])
        boxes.append(
            Box(
                max(0.0, cx - w / 2),
                max(0.0, cy - h / 2),
                min(1.0, cx + w / 2),
                min(1.0, cy + h / 2),
            )
        )
    return boxes


def load_boxes(root: str | Path, image_id: str) -> list[Box]:
    """The boxes of one image (empty list for a normal image or a missing file)."""
    f = Path(root) / YOLO_DIR / f"{Path(image_id).stem}.txt"
    return parse_yolo(f.read_text()) if f.is_file() else []
