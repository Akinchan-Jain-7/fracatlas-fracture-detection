"""Phase 7 gallery: X-rays with the radiologist boxes and the detector's boxes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from fracdet.data.transforms import DraftGrayscale  # noqa: E402
from fracdet.eval.plots import INK, INK_2, _style  # noqa: E402

GT_COLOR = "#38e0c8"  # cyan, as in Phase 6
TOP_COLOR = "#ff8a1f"  # orange: the detector's highest-scoring box
OTHER_COLOR = "#ffd166"  # pale yellow: further boxes above the display score
SHOW_SCORE = 0.3


def _rect(ax, b, w, h, color, lw, ls="-"):
    ax.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h,
                           fill=False, edgecolor=color, linewidth=lw, linestyle=ls))  # fmt: skip


def plot_detection_gallery(
    items: pd.DataFrame, paths: pd.Series, results: dict, out: Path, title: str, cols: int = 4
) -> None:
    """``items``: rows of the per-image table; ``results``: image_id -> ImageResult."""
    _style()
    n = len(items)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 4.0 * rows + 0.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(items.itertuples(index=False)):
        ax = axes[k // cols, k % cols]
        with Image.open(paths[r.image_id]) as img:
            gray = np.asarray(DraftGrayscale(512)(img))
        h, w = gray.shape
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        res = results[r.image_id]
        for g in res.gt:
            _rect(ax, g, w, h, GT_COLOR, 1.4)
        order = np.argsort(-res.scores, kind="stable")
        for rank, j in enumerate(order):
            if rank > 0 and res.scores[j] < SHOW_SCORE:
                break
            b = res.boxes[j]
            top = rank == 0
            _rect(ax, b, w, h, TOP_COLOR if top else OTHER_COLOR, 1.8 if top else 1.0,
                  "-" if top else "--")  # fmt: skip
            # Inside the box's top-left corner, on a dark tag: never collides with the title.
            ax.text(b[0] * w + 2, b[1] * h + 2, f"{res.scores[j]:.2f}", fontsize=7,
                    color=TOP_COLOR if top else OTHER_COLOR, va="top", ha="left",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1,
                          "edgecolor": "none"})  # fmt: skip
        verdict = "" if r.label == 0 else ("  hit" if r.det_top1_pointing else "  miss")
        ax.set_title(f"{Path(r.image_id).stem}  {r.region}\nclassifier p = "
                     f"{r.classifier_prob:.3f}{verdict}", fontsize=8.5, loc="left",
                     fontweight="normal", color=INK)  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(
        0.01,
        0.003,
        "Cyan: radiologist fracture boxes. Orange: detector's top box (score shown). "
        f"Dashed yellow: other boxes scoring >= {SHOW_SCORE}.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
