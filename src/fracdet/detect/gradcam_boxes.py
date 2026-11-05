"""Baseline: boxes from Grad-CAM heatmaps, with no box training at all.

For one image: take the classifier's Grad-CAM map (Phase 6, last stage), keep
the pixels above half of its peak, take the connected region that contains the
peak, and draw its bounding box. The box's score is the classifier's fracture
probability, so the classifier's ranking of images is the box ranking.

This is what "localisation for free" looks like. The detector, trained on the
radiologist boxes, has to beat it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from fracdet.data.transforms import LetterboxGeometry


def cam_box(
    cam_up: np.ndarray, geom: LetterboxGeometry, relative_threshold: float = 0.5
) -> np.ndarray | None:
    """Normalised (x0, y0, x1, y1) box around the heatmap's peak region, or None if empty.

    ``cam_up``: the map upsampled to the letterbox size. Padding is ignored.
    """
    rows, cols = geom.content
    cam = cam_up[rows, cols]
    peak = float(cam.max()) if cam.size else 0.0
    if peak <= 0:
        return None
    labels, _ = ndimage.label(cam >= relative_threshold * peak)
    py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
    ys, xs = np.nonzero(labels == labels[py, px])
    h, w = cam.shape
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])
