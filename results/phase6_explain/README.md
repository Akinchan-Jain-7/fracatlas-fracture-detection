# Phase 6: Grad-CAM explainability results (ConvNeXt-Tiny, seed 44)

Analysis plan: `configs/explain.yaml`, committed in `0e52ac7` before any heatmap
was computed. Method and discussion: `docs/phase-6-explainability.md`.

| File (per split, `val/` and `test/`) | What |
|---|---|
| `summary.json` | Pointing-game hit rates with 95% patient-group CIs, energy share, baselines, sanity check (val), protocol record |
| `report.md` | The same as tables |
| `localization_per_image.csv` | Every fractured image: outcome, region, hit / energy share / peak position per layer, box area share, centre hit |

Figures (galleries chosen by the rule in the plan): `docs/figures/phase-6/`.

**Kaggle:** notebook `notebook2ac59126bb`, **Version 7** ("phase6 grad-cam rerun
(overlay fix)"), Save & Run All, Tesla T4, code at `fd69619`. The files here are
copies of that version's output (`phase6_explain.zip`). The raw heatmaps
(`cams.npz`, about 1 MB per split) and the remaining galleries stay in that output.

Version 6 (code `0e52ac7`) produced identical numbers. Version 7 only fixed how
the heatmap overlay is drawn: Kaggle's matplotlib painted the maps fully opaque.

**Protocol checks, both splits:** the plan file was committed and unchanged; the
checkpoint matched the Phase 5 threshold record; the TP / FP / TN / FN counts at
threshold 0.0968 equalled the committed Phase 5 counts (val 92 / 131 / 353 / 10,
test 94 / 142 / 345 / 8). Every headline number was recomputed from
`localization_per_image.csv` and matches `summary.json`.

SHA-256 of the copied files:

```
c139577938d1c1f9cd55fde2ee7dddb66bff336938736054491c9c738050d4ba  val/summary.json
de4d18894f6438cd1a2c9d11899acbae00d67b870c3560f10493a134deebfa74  val/report.md
98f31cd2b371e4d481f1066810e06e8cb541b356c5cf00a666505e75fe0255e8  val/localization_per_image.csv
6290b16e679cdd01dcc42314df8956daea64b8cd4c6961d4ee0e8c765599e264  test/summary.json
2da3a6cbf65d468fa6221fbdf742286fe7e4c83510298e5c4f51f4057b2deea2  test/report.md
109760251fa80964df4207a4e3ed1f4da4b892f15eaef8215e7bdfc210e6c6af  test/localization_per_image.csv
```

Headline (fractured images; pointing game = heatmap peak inside a fracture box):

| | Validation (102) | Test (102) |
|---|---|---|
| Grad-CAM `stages.3` (primary) | 0.216 [0.126, 0.325] | 0.225 [0.135, 0.318] |
| Grad-CAM `stages.2` | 0.353 | 0.353 |
| centre of image | 0.049 | 0.078 |
| chance (box area share) | 0.008 | 0.010 |
| missed fractures with a hit | 0 / 10 | 0 / 8 |
| random weights (whole network), 64 val images | 0.031 | – |
