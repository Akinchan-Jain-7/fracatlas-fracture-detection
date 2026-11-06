# Results

Every number in the README and the phase write-ups comes from a file here. The files
are copies of what the runs wrote on Kaggle (not re-typed); each folder's README says
which notebook version and run produced it, with checksums where they matter.

| Folder | Phase | What it holds |
|---|---|---|
| [`phase2_baseline_resnet50/`](phase2_baseline_resnet50/) | 2 | ResNet-50 baseline: per-epoch validation metrics and run summary |
| [`phase3_baseline_resnet50/`](phase3_baseline_resnet50/) | 3 | Baseline's pre-registered threshold and its one-shot test evaluation |
| [`phase4_imbalance/`](phase4_imbalance/) | 4 | Imbalance sweep (4 variants × 3 seeds, validation only) and the comparison against the fixed rule |
| [`phase5_models/`](phase5_models/) | 5 | Resolution and backbone sweep (5 candidates × 3 seeds, validation only) |
| [`phase3_p5_convnext_tiny_s44/`](phase3_p5_convnext_tiny_s44/) | 5 | The winner's (ConvNeXt-Tiny, seed 44) threshold and one-shot test evaluation: the headline numbers |
| [`phase6_explain/`](phase6_explain/) | 6 | Grad-CAM pointing game and box overlap, per image and summarised, validation and test |
| [`phase7_detect/`](phase7_detect/) | 7 | Detector training logs (2 seeds), validation reports, and the one-shot test of seed 42 |

`report.md` files are readable summaries; `summary.json` / `*_metrics.json` hold the
same numbers for scripts; `*_per_image.csv` files allow any number to be recomputed.
