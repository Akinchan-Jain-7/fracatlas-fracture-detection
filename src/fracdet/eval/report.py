"""Metrics with intervals for the whole test set and for subgroups.

Subgroups exist to catch a model that looks good overall for the wrong reason:

- ``no_hardware``: Phase 1 found 97 of 99 images with surgical hardware are
  fractured. A model can score well by spotting metal. Removing those images
  shows how well it finds fractures without that clue. If the numbers drop a
  lot, the overall score was partly "metal detection".
- ``hardware``: the other side of the same check (only a handful of images).
- ``region=<part>``: hand, leg, hip... A screening tool that only works on hands
  should say so.

A subgroup with fewer than ``min_positives`` fractures gets point estimates
only: an interval from 1-5 fractures would be uselessly wide and invites
over-reading.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fracdet.eval.bootstrap import cluster_bootstrap, metric_fn_at
from fracdet.eval.metrics import eval_metrics

# Shown in this order in printed tables.
HEADLINE = ["recall", "precision", "specificity", "npv", "f1", "roc_auc", "pr_auc"]


def subgroup_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {"all": np.ones(len(df), dtype=bool)}
    if "hardware" in df:
        masks["no_hardware"] = (df["hardware"] == 0).to_numpy()
        masks["hardware"] = (df["hardware"] == 1).to_numpy()
    if "region" in df:
        for region in sorted(df["region"].unique()):
            masks[f"region={region}"] = (df["region"] == region).to_numpy()
    return {name: m for name, m in masks.items() if m.any()}  # drop empty subgroups


def _clean(x: Any) -> Any:
    """NaN -> None so the report is valid JSON (``allow_nan=False``)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def evaluate_frame(
    df: pd.DataFrame,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
    min_positives: int = 10,
    group_col: str = "split_group",
) -> dict[str, dict[str, Any]]:
    """Per-subgroup counts, metrics at ``threshold``, and cluster-bootstrap CIs.

    ``df`` needs ``label`` and ``prob``; ``hardware``, ``region`` and
    ``split_group`` are used when present (no group column -> per-image
    bootstrap, recorded as such).
    """
    clustered = group_col in df
    out = {}
    for name, mask in subgroup_masks(df).items():
        sub = df[mask]
        y, p = sub["label"].to_numpy(), sub["prob"].to_numpy()
        n_pos = int(y.sum())
        entry: dict[str, Any] = {
            "n": int(len(sub)),
            "n_fractured": n_pos,
            "n_groups": int(sub[group_col].nunique()) if clustered else int(len(sub)),
            "n_fractured_groups": int(sub.loc[sub["label"] == 1, group_col].nunique())
            if clustered
            else n_pos,
            "metrics": eval_metrics(y, p, threshold),
        }
        if n_pos >= min_positives and n_pos < len(y):
            groups = sub[group_col].to_numpy() if clustered else np.arange(len(sub))
            entry["ci95"] = {
                k: {"low": v["low"], "high": v["high"]}
                for k, v in cluster_bootstrap(
                    y, p, groups, metric_fn_at(threshold), n_boot=n_boot, seed=seed
                ).items()
            }
        else:
            entry["ci95"] = None
            entry["note"] = f"fewer than {min_positives} fractures (or no normals): no interval"
        out[name] = entry
    return _clean(out)


def _fmt(v: float | None, ci: dict | None, key: str) -> str:
    if v is None:
        return "n/a"
    s = f"{v:.3f}"
    if ci and ci.get(key) and ci[key]["low"] is not None:
        s += f" [{ci[key]['low']:.3f}, {ci[key]['high']:.3f}]"
    return s


def markdown_table(
    report: dict[str, dict[str, Any]], keys=("recall", "precision", "specificity", "pr_auc")
) -> str:
    """Subgroups as rows: counts, then each metric with its 95% interval."""
    head = "| Subgroup | Images | Fractures (groups) | TP / FN | FP | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (5 + len(keys))
    lines = [head, sep]
    for name, e in report.items():
        m, ci = e["metrics"], e["ci95"]
        cells = [
            name,
            str(e["n"]),
            f"{e['n_fractured']} ({e['n_fractured_groups']})",
            f"{m['tp']} / {m['fn']}",
            str(m["fp"]),
            *(_fmt(m[k], ci, k) for k in keys),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
