# Phase 4: imbalance sweep (validation only)

| File | What |
|---|---|
| `runs.csv` | One row per run (12): best epoch, minutes, val PR-AUC / ROC-AUC, threshold, specificity, precision and alarms per 100 at recall ≥ 0.90 |
| `compare.json` | Per-variant mean / sd / paired delta vs baseline, and the decision of the pre-registered rule |
| `compare.md` | The comparison table |

`compare.json` and `compare.md` are byte-identical to the Kaggle run's files (checked by SHA-256).
`runs.csv` was rebuilt from the output viewer's table view. Recomputing `compare.json` from it with
`fracdet.eval.compare` reproduces the Kaggle file exactly. The figure is `docs/figures/phase-4/compare.png`.

**Kaggle:** notebook `notebook34061d5c16`, Version 1 ("phase4 imbalance sweep (4 variants x 3 seeds)").
- Run settings: Save & Run All, T4 x2, 73 min, code at commit `6e57f73` (notebook `63a27f7`).
- Each run's `best.pt`, `val_predictions.csv`, `metrics.csv` and log are in that version's output, under `outputs/runs/` and `outputs/sweeps/phase4_imbalance/`.

**Decision:** baseline stays. No variant's gain in specificity at 90% recall exceeded seed noise.
All three were worse on average. The test set was not used.
