"""Phase 6 figures: heatmap galleries and the localisation summary chart.

Gallery panels come in pairs: the X-ray as the model sees it (grayscale,
letterboxed, padding cropped off) and the same X-ray with the Grad-CAM map on
top. Fracture boxes are outlined in both, and a cross marks the map's peak
(the pointing-game point). Which images appear is decided by
``configs/explain.yaml``, not by looking at them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from fracdet.data.boxes import Box  # noqa: E402
from fracdet.data.transforms import (  # noqa: E402
    DEFAULT_PREPROCESS,
    DraftGrayscale,
    Letterbox,
    LetterboxGeometry,
    letterbox_geometry,
)
from fracdet.eval.plots import GRID, INK, INK_2, SERIES, SURFACE, _style  # noqa: E402

BOX_COLOR = "#38e0c8"  # bright cyan: visible on both bone and the red end of the map
PEAK_COLOR = "#ffffff"


def model_view(path: str | Path, image_size: int) -> tuple[np.ndarray, LetterboxGeometry]:
    """The grayscale letterboxed image exactly as the model received it (before normalising)."""
    with Image.open(path) as img:
        gray = DraftGrayscale(image_size)(img)
        geom = letterbox_geometry(*gray.size, image_size)
        boxed = Letterbox(image_size, DEFAULT_PREPROCESS["pad_value"])(gray)
    return np.asarray(boxed), geom


def _draw_boxes(ax, boxes: list[Box], geom: LetterboxGeometry) -> None:
    rows, cols = geom.content
    for b in boxes:
        x0, y0 = geom.to_letterbox(b.x0, b.y0)
        x1, y1 = geom.to_letterbox(b.x1, b.y1)
        ax.add_patch(
            Rectangle(
                (x0 - cols.start, y0 - rows.start), x1 - x0, y1 - y0,
                fill=False, edgecolor=BOX_COLOR, linewidth=1.3,
            )
        )  # fmt: skip


def plot_gallery(
    items: pd.DataFrame,
    cams: dict[str, np.ndarray],
    boxes: dict[str, list[Box]],
    out: Path,
    title: str,
    image_size: int,
    cols: int = 3,
) -> None:
    """Grid of (X-ray, X-ray + heatmap) pairs.

    ``items`` needs image_id, path, prob, outcome, region; ``cams`` maps
    image_id -> upsampled (size x size) map.
    """
    _style()
    n = len(items)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(5.0 * cols, 2.75 * rows + 0.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(items.itertuples(index=False)):
        img, geom = model_view(r.path, image_size)
        rs, cs = geom.content
        crop = img[rs, cs]
        cam = cams[r.image_id][rs, cs]
        peak = cam.max()
        a_plain, a_cam = axes[k // cols, 2 * (k % cols)], axes[k // cols, 2 * (k % cols) + 1]
        a_plain.imshow(crop, cmap="gray", vmin=0, vmax=255)
        a_cam.imshow(crop, cmap="gray", vmin=0, vmax=255)
        if peak > 0:
            norm = cam / peak
            a_cam.imshow(norm, cmap="jet", alpha=0.45 * norm**0.5, vmin=0, vmax=1)
            py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
            a_cam.plot(px, py, marker="x", color=PEAK_COLOR, markersize=7, mew=2)
        for ax in (a_plain, a_cam):
            _draw_boxes(ax, boxes.get(r.image_id, []), geom)
        empty = "" if peak > 0 else "  (empty map)"
        a_plain.set_title(
            f"{Path(r.image_id).stem}  {r.region}\np = {r.prob:.3f}  {r.outcome}{empty}",
            fontsize=8.5, loc="left", fontweight="normal", color=INK,
        )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(
        0.01, 0.003,
        "Cyan boxes: radiologist fracture boxes. Heatmap: Grad-CAM evidence for fracture "
        "(red = strongest). White x: the map's peak.",
        fontsize=8, color=INK_2,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_pointing_summary(rows: list[dict], out: Path, title: str) -> None:
    """Horizontal bars: pointing-game hit rate (with CI when given) for each method/baseline.

    ``rows``: dicts with ``label``, ``value`` and optional ``low``/``high``/``kind``
    (``kind`` = "method" or "baseline", which only changes the colour).
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows, strict=True):
        color = SERIES[0] if r.get("kind", "method") == "method" else GRID
        ax.barh(yi, r["value"], color=color, height=0.6)
        if r.get("low") is not None:
            ax.plot([r["low"], r["high"]], [yi, yi], color=INK, linewidth=1.2)
        ax.text(
            r["value"] + 0.012 if r.get("low") is None else r["high"] + 0.012, yi,
            f"{r['value']:.2f}", va="center", fontsize=9, color=INK_2,
        )  # fmt: skip
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pointing-game hit rate (fractured images)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_facecolor(SURFACE)
    fig.savefig(out)
    plt.close(fig)
