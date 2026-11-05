"""Is the heatmap on the fracture? Scored against the FracAtlas boxes.

All scores are computed in the model's letterboxed frame (384 x 384), inside
the image area only: the padding is never a valid place to look, and counting
it would make every baseline look better than it is.

- **Pointing game** (Zhang et al., 2016): a *hit* if the single hottest pixel of
  the heatmap is inside a fracture box. Simple, and independent of how the map
  is scaled or thresholded.
- **Energy in box**: the share of the heatmap's total mass inside the boxes.
  Rewards maps that concentrate on the fracture rather than just peaking there.

Each number has a built-in baseline:

- ``area_share``: the share of the image the boxes cover. A uniformly random
  point lands in a box with exactly this probability, and a flat heatmap has
  exactly this energy share. So it is both the chance hit rate and the chance
  energy share.
- ``centre_hit``: whether the centre of the image is inside a box ("always point
  at the middle", no model needed). X-rays are usually framed on the injury,
  so this prior is stronger than chance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fracdet.data.boxes import Box
from fracdet.data.transforms import LetterboxGeometry


def box_mask(boxes: Sequence[Box], geom: LetterboxGeometry) -> np.ndarray:
    """Boolean (size, size) mask of letterbox pixels whose centre lies in any box."""
    centres = np.arange(geom.size) + 0.5
    mask = np.zeros((geom.size, geom.size), dtype=bool)
    for b in boxes:
        x0, y0 = geom.to_letterbox(b.x0, b.y0)
        x1, y1 = geom.to_letterbox(b.x1, b.y1)
        rows = (centres >= y0) & (centres <= y1)
        cols = (centres >= x0) & (centres <= x1)
        mask |= rows[:, None] & cols[None, :]
    return mask


def content_mask(geom: LetterboxGeometry) -> np.ndarray:
    mask = np.zeros((geom.size, geom.size), dtype=bool)
    mask[geom.content] = True
    return mask


def localization_scores(
    cam_up: np.ndarray, boxes: Sequence[Box], geom: LetterboxGeometry
) -> dict[str, float | bool | int]:
    """Pointing-game hit, energy share and the two baselines for one image.

    ``cam_up`` is the Grad-CAM map upsampled to the letterbox size (>= 0).
    A map with no positive value anywhere in the image ("empty": nothing
    argued for fracture) counts as a miss with zero energy in the box.
    """
    if cam_up.shape != (geom.size, geom.size):
        raise ValueError(f"heatmap {cam_up.shape} does not match letterbox {geom.size}")
    content = content_mask(geom)
    inbox = box_mask(boxes, geom) & content
    if not inbox.any():
        raise ValueError("no box pixel inside the image area")

    values = np.where(content, cam_up, -np.inf)
    total = float(cam_up[content].sum())
    empty = not np.isfinite(values).any() or float(values.max()) <= 0.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    rows, cols = geom.content
    cy = (rows.start + rows.stop) // 2
    cx = (cols.start + cols.stop) // 2
    return {
        "hit": bool(not empty and inbox[peak_y, peak_x]),
        "energy_share": 0.0 if empty else float(cam_up[inbox].sum()) / total,
        "area_share": float(inbox.sum()) / float(content.sum()),
        "centre_hit": bool(inbox[cy, cx]),
        "empty": bool(empty),
        "peak_y": int(peak_y),
        "peak_x": int(peak_x),
        "n_boxes": len(boxes),
    }
