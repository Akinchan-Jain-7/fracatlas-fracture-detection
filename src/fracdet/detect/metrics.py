"""Detection metrics, written out so every step can be explained.

Each image contributes ground-truth boxes (0 for a normal X-ray) and scored
predicted boxes. All boxes are in normalised image coordinates (0..1): IoU and
"centre inside a box" do not change under a per-axis resize, so the numbers
are the same whatever size the detector worked at.

- **Matching** (per image, COCO style): predictions in score order; each takes
  the unmatched ground-truth box it overlaps most. IoU >= threshold -> true
  positive (that box is now used); otherwise false positive. A second box on an
  already-found fracture is therefore a false positive.
- **AP** (average precision): rank every prediction of the split by score and
  walk down the list, tracking precision and recall. AP is the area under the
  precision-recall curve after making precision non-increasing (all-point
  interpolation). Boxes on normal X-rays are false positives, so a detector that
  boxes healthy bones loses AP.
- **FROC**: the share of fracture boxes found when the score cut-off lets
  through on average 0.5, 1 or 2 false boxes per image. This is how detection
  results are usually read in radiology: "finds 70% of fractures at 1 false
  mark per image".
- **Top-1**: only the single highest-scoring box per fractured image. Its
  centre inside a fracture box is the Phase 6 pointing game, so the detector
  and Grad-CAM are compared on the same test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N,4) with every box in ``b`` (M,4), corners x0,y0,x1,y1."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thr: float
) -> np.ndarray:
    """True-positive flag for each prediction (in the order given), greedy by score."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable")
    ious = iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(ious.shape[1], dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    for i in order:
        if ious.shape[1] == 0:
            break
        cand = np.where(used, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i], used[j] = True, True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP; NaN when there is no ground truth at all."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp = np.asarray(tp, dtype=bool)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    # Make precision non-increasing from the right (the interpolated envelope).
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - r_prev) * precision))


def froc_points(
    scores: np.ndarray, tp: np.ndarray, n_gt: int, n_images: int, fp_rates: Sequence[float]
) -> dict[str, float]:
    """Sensitivity (share of GT boxes found) at each average false-positives-per-image level."""
    out: dict[str, float] = {}
    if n_gt == 0 or n_images == 0:
        return {f"sens_at_{f:g}_fp": float("nan") for f in fp_rates}
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp = np.asarray(tp, dtype=bool)[order]
    sens = np.concatenate([[0.0], np.cumsum(tp) / n_gt])
    fppi = np.concatenate([[0.0], np.cumsum(~tp) / n_images])
    for f in fp_rates:
        ok = fppi <= f + 1e-12
        out[f"sens_at_{f:g}_fp"] = float(sens[ok].max())
    return out


@dataclass
class ImageResult:
    """Everything the metrics need from one image."""

    image_id: str
    label: int
    gt: np.ndarray  # (M, 4) normalised
    boxes: np.ndarray  # (N, 4) normalised, any order
    scores: np.ndarray  # (N,)
    extra: dict[str, Any] = field(default_factory=dict)

    def top1(self) -> dict[str, Any]:
        """Top box's centre-in-GT and IoU >= 0.5 (fractured images only)."""
        if len(self.scores) == 0:
            return {"top1_pointing": False, "top1_iou50": False, "top1_iou": 0.0, "top1_score": 0.0}
        k = int(np.argmax(self.scores))
        b = self.boxes[k]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        inside = bool(any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in self.gt))
        iou = float(iou_matrix(b[None], self.gt).max()) if len(self.gt) else 0.0
        return {"top1_pointing": inside, "top1_iou50": iou >= 0.5, "top1_iou": iou,
                "top1_score": float(self.scores[k])}  # fmt: skip


def prepare(results: Sequence[ImageResult], iou_thr: float = 0.5) -> list[dict[str, Any]]:
    """Match every image once (at ``iou_thr`` and at 0.25); metrics then only pool."""
    out = []
    for r in results:
        rec = {
            "label": r.label,
            "scores": np.asarray(r.scores, dtype=float),
            "tp": match(r.boxes, r.scores, r.gt, iou_thr),
            "tp25": match(r.boxes, r.scores, r.gt, 0.25),
            "n_gt": len(r.gt),
        }
        if r.label == 1:
            rec.update(r.top1())
        out.append(rec)
    return out


def pooled_metrics(prepared: Sequence[dict[str, Any]], fp_rates: Sequence[float] = (0.5, 1, 2)):
    """AP (all images, fractured only, IoU 0.25), FROC points and top-1 rates."""

    def cat(rs, key):
        return np.concatenate([r[key] for r in rs]) if rs else np.zeros(0)

    frac = [r for r in prepared if r["label"] == 1]
    n_gt = sum(r["n_gt"] for r in prepared)
    s, t = cat(prepared, "scores"), cat(prepared, "tp").astype(bool)
    out = {
        "n_images": len(prepared),
        "n_fractured": len(frac),
        "n_gt_boxes": n_gt,
        "ap50_all": average_precision(s, t, n_gt),
        "ap50_fractured": average_precision(
            cat(frac, "scores"), cat(frac, "tp").astype(bool), sum(r["n_gt"] for r in frac)
        ),
        "ap25_all": average_precision(s, cat(prepared, "tp25").astype(bool), n_gt),
        **froc_points(s, t, n_gt, len(prepared), fp_rates),
    }
    if frac:
        out["top1_pointing"] = float(np.mean([r["top1_pointing"] for r in frac]))
        out["top1_iou50"] = float(np.mean([r["top1_iou50"] for r in frac]))
    normals = [r for r in prepared if r["label"] == 0]
    if normals:
        out["boxes_per_normal_at_0.5"] = float(
            np.mean([(r["scores"] >= 0.5).sum() for r in normals])
        )
    return out


def split_metrics(
    results: Sequence[ImageResult], iou_thr: float = 0.5, fp_rates: Sequence[float] = (0.5, 1, 2)
) -> dict[str, float]:
    return pooled_metrics(prepare(results, iou_thr), fp_rates)


def grouped_bootstrap(
    results: Sequence[ImageResult],
    groups: Sequence[Any],
    keys: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    iou_thr: float = 0.5,
    fp_rates: Sequence[float] = (0.5, 1, 2),
) -> dict[str, dict[str, float]]:
    """Percentile CIs for ``keys`` of the split metrics, resampling whole patient groups.

    Matching is per image, so it is done once; each resample only re-pools the
    images it drew (a group drawn twice counts twice, as in any bootstrap).
    """
    prepared = prepare(results, iou_thr)
    point = pooled_metrics(prepared, fp_rates)
    g = np.asarray(groups)
    _, gidx = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(gidx == k) for k in range(gidx.max() + 1)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[k] for k in picked])
        m = pooled_metrics([prepared[i] for i in rows], fp_rates)
        for k in keys:
            samples[k].append(m.get(k, float("nan")))
    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(arr)
                  else (float("nan"), float("nan")))  # fmt: skip
        out[k] = {"value": point.get(k), "low": float(lo), "high": float(hi)}
    return out
