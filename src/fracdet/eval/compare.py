"""Compare training variants across seeds on validation, and apply the selection rule.

Per run, from its own files (no images loaded, no test data):
- ``pr_auc``, ``roc_auc``: threshold-free ranking quality of the best epoch;
- ``spec_at_recall``: specificity at the Phase 3 threshold rule (highest
  threshold with recall >= target). This is the operating question: how many
  normal X-rays get flagged when we insist on catching 90% of fractures;
- ``alarms_per_100`` and ``precision`` at that same threshold.

Per variant: mean and standard deviation over seeds, and the mean *paired*
difference from the baseline (same seed vs same seed). Pairing removes the part
of the seed-to-seed variation the two runs share.

The decision rule (configured in ``configs/sweep_*.yaml``, committed before the
sweep runs): a variant wins if its mean paired gain on the primary metric is
larger than the seed noise, and it does not lose more than the noise on the
guard metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fracdet.config import load_config
from fracdet.eval.threshold import threshold_for_recall

METRICS = ["pr_auc", "roc_auc", "spec_at_recall", "precision_at_recall", "alarms_per_100"]


def run_record(run_dir: str | Path, recall_target: float) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    summary = json.loads((run_dir / "summary.json").read_text())
    preds = pd.read_csv(run_dir / "val_predictions.csv")
    op = threshold_for_recall(preds["label"], preds["prob"], recall_target)
    m = op["metrics"]
    return {
        "experiment": cfg.get("experiment", cfg.get("run_name")),
        "seed": cfg.get("seed"),
        "run_dir": str(run_dir),
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "minutes": summary["minutes"],
        "pr_auc": summary["val"]["pr_auc"],
        "roc_auc": summary["val"]["roc_auc"],
        "threshold": op["threshold"],
        "spec_at_recall": m["specificity"],
        "precision_at_recall": m["precision"],
        "recall_at_threshold": m["recall"],
        "alarms_per_100": 100 * (m["tp"] + m["fp"]) / len(preds),
    }


def aggregate(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Mean/sd per experiment, plus mean and sd of the per-seed difference from baseline."""
    base = runs[runs.experiment == baseline].set_index("seed")
    rows = []
    for exp, g in runs.groupby("experiment", sort=False):
        row: dict[str, Any] = {"experiment": exp, "n_seeds": len(g)}
        paired = g.set_index("seed").join(base[METRICS], rsuffix="_base", how="inner")
        for k in METRICS:
            row[f"{k}_mean"] = g[k].mean()
            row[f"{k}_sd"] = g[k].std(ddof=1) if len(g) > 1 else np.nan
            d = paired[k] - paired[f"{k}_base"]
            row[f"{k}_delta"] = d.mean() if len(d) else np.nan
            row[f"{k}_delta_sd"] = d.std(ddof=1) if len(d) > 1 else np.nan
        row["minutes_mean"] = g["minutes"].mean()
        row["best_epoch_mean"] = g["best_epoch"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def decide(agg: pd.DataFrame, runs: pd.DataFrame, baseline: str, primary: str, guard: str) -> dict:
    """Apply the pre-registered rule. Returns the decision and the reasoning per variant."""
    b = agg.set_index("experiment").loc[baseline]
    verdicts, winners = {}, []
    for _, r in agg.iterrows():
        if r.experiment == baseline:
            continue
        noise_p = np.nanmax([b[f"{primary}_sd"], r[f"{primary}_sd"]])
        noise_g = np.nanmax([b[f"{guard}_sd"], r[f"{guard}_sd"]])
        gain, guard_delta = r[f"{primary}_delta"], r[f"{guard}_delta"]
        ok = bool(gain > noise_p and guard_delta > -noise_g)
        verdicts[r.experiment] = {
            f"{primary}_delta": round(float(gain), 4),
            f"{primary}_noise": round(float(noise_p), 4),
            f"{guard}_delta": round(float(guard_delta), 4),
            f"{guard}_noise": round(float(noise_g), 4),
            "wins": ok,
        }
        if ok:
            winners.append((gain, r.experiment))
    if not winners:
        return {"winner": baseline, "changed": False, "verdicts": verdicts}
    winner = max(winners)[1]
    g = runs[runs.experiment == winner].sort_values(primary).reset_index(drop=True)
    median_run = g.iloc[(len(g) - 1) // 2]  # lower median for an even count: the cautious pick
    return {
        "winner": winner,
        "changed": True,
        "representative_run": median_run["run_dir"],
        "representative_seed": int(median_run["seed"]),
        "verdicts": verdicts,
    }


def markdown(agg: pd.DataFrame, baseline: str) -> str:
    def cell(r, k, digits=3):
        s = f"{r[f'{k}_mean']:.{digits}f} ± {r[f'{k}_sd']:.{digits}f}"
        if r.experiment != baseline:
            s += f" (Δ {r[f'{k}_delta']:+.{digits}f})"
        return s

    lines = [
        "| Variant | Seeds | Val PR-AUC | Specificity @ recall target | Alarms / 100 "
        "| Best epoch | Min/run |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.experiment} | {r.n_seeds} | {cell(r, 'pr_auc')} | {cell(r, 'spec_at_recall')} | "
            f"{cell(r, 'alarms_per_100', 1)} | {r.best_epoch_mean:.1f} | {r.minutes_mean:.1f} |"
        )
    return "\n".join(lines)
