# Run: baseline_resnet50 (Phase 2)

`metrics.csv` and `summary.json` are exact copies of the files written by the
run, read from the Kaggle notebook output viewer (not re-typed).

- Kaggle: notebook `notebook2ac59126bb`, **Version 1** ("phase2 baseline resnet50
  (best epoch 16)", Quick Save with output). Its output holds the full run folder
  `outputs/runs/20261103-153336_baseline_resnet50/` including `best.pt` (94 MB),
  `last.pt`, `val_predictions.csv`, `train.log`, `config.yaml`.
- Trained with commit `a579c9f`; post-training step completed with `b0f779c`
  (`scripts/finalize_run.py`) after a device bug in the original final step.
- Split: `split_v1` (MD5 of the split CSV recorded in `summary.json`).

Phase 3 attaches that notebook version as an input and evaluates exactly this
`best.pt` on the test set.
