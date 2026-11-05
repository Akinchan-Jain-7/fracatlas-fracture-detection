# Phase 7: fracture detector results (Faster R-CNN, torchvision)

Plan: `configs/detect.yaml`, committed in `8b67243` before training. Method and discussion:
`docs/phase-7-localization.md`.

| Folder | What |
|---|---|
| `test/` | **Seed 42, scored once on test**: `summary.json` (detector + Grad-CAM baseline, 95% patient-group CIs, end-to-end, protocol), `report.md`, `test_per_image.csv` (top box hit / IoU / score per image for both methods, classifier probability and flag), `test_detections.csv` (every box, normalised, with score) |
| `val_s42/` | The same on validation for seed 42 |
| `val_s43/` | Seed 43 on validation only (re-run spread; never tested) |
| `train_s42/`, `train_s43/` | Run config, per-epoch metrics, summary, training log |

Figures (galleries chosen by the rule in the plan): `docs/figures/phase-7/`.

**Kaggle:** notebook `notebook2ac59126bb`, **Version 8** ("phase7 detector train + eval
(s42, s43)"), Save & Run All, T4 x2, 2,614 s, code `8b67243`, driver
`scripts/kaggle_phase7.sh`. Files here are copies of that version's `phase7_detect.zip`.

**Model weights** (`detector_s42_best.pt`, 166 MB, SHA-256
`4a87100d08b1c998c5599016755271dfbda661d8b38bbee3494aef2863ab24c3`) are in the same
Kaggle output and kept outside git. Checked: loads with `load_detector`, epoch 5, seed 42,
trained at `8b67243`, split MD5 `99cbcb06a8e71d67f5215fda47fad124`.

**Checks:** plan committed and unchanged; test run only for the pre-registered seed;
classifier TP / FP / TN / FN equal to the committed Phase 5 counts on both splits; every
top-1 and end-to-end number recomputed from the per-image files; the validation boxes
saved during training are byte-identical to the evaluation's (same model, same images).

SHA-256:

```
27fa1bdef08a7ba593323624772c86169ce88c33367e63324e66d7997c14bb28  test/summary.json
0902345687ef9a94d7ce0a1d10da0a11c9745104fdf5ce96197ccc3696dc4c23  test/report.md
504c26071d3f95c15d8267f32c4512b13b0ec03d737022452639a79e37d09da0  test/test_per_image.csv
6480f972253725f2c9b243d88ca301a566ea2acd3bdee82ee4da868d00fa7ed1  test/test_detections.csv
8651f200e9209835ab1745d7fa347d25db1e38ee1ec801537d9d32a272d9bf37  val_s42/summary.json
5c1911a4ed7fedc19dcee9764d7a303189dd92711380eff11a97c40475c94994  val_s43/summary.json
```

Headline (test, seed 42): AP50 (all images) 0.421 [0.296, 0.528]; top box on the
fracture 0.745 [0.650, 0.837] (Grad-CAM boxes 0.304); 0.635 of fracture boxes found at
one false box per image; 72 of the 94 classifier-flagged fractures get a correct top box.
