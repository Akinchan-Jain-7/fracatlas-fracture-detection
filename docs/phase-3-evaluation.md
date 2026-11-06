# Phase 3: Proper evaluation

**Goal:** turn the Phase 2 model into one honest, defensible test number:
a decision threshold chosen on validation only and fixed in git *before* the test
set is opened, then a single test pass reported with confidence intervals, with
and without surgical-hardware images.

No new training happens in this phase. The model is exactly Phase 2's `best.pt`
(epoch 16, commit `a579c9f`).

## Results

**Operating threshold 0.00477**, chosen on validation (recall ≥ 0.90) and
committed in `c805cfa` at 07:07 UTC, four minutes before the test set was
scored at 07:10 UTC. Re-scoring validation before the test pass reproduced it
exactly (PR-AUC identical, 0 images flipped). Tesla T4, Kaggle notebook
Version 3.

### Test set (589 images, 102 fractures in 47 groups), 95% cluster-bootstrap CI

| Metric | At the operating threshold | At 0.5 (Phase 2 default) |
|---|---|---|
| **Recall** | **0.892** [0.825, 0.955] — 91 of 102 found | 0.588 — 60 of 102 |
| Precision | 0.293 [0.208, 0.379] | 0.769 |
| Specificity | 0.548 [0.496, 0.600] | 0.963 |
| NPV | 0.960 [0.933, 0.984] | 0.918 |
| Flagged per 100 X-rays | 52.8 | 13.2 |
| ROC-AUC | 0.870 [0.818, 0.919] | (same) |
| PR-AUC | 0.739 [0.625, 0.832] | (same) |

### Subgroups (test, operating threshold)

| Subgroup | Images | Fractures (groups) | Found / missed | False alarms | Recall | Specificity | PR-AUC |
|---|---|---|---|---|---|---|---|
| all | 589 | 102 (47) | 91 / 11 | 220 | 0.892 [0.825, 0.955] | 0.548 [0.496, 0.600] | 0.739 [0.625, 0.832] |
| **no hardware** | 578 | 91 (43) | 80 / 11 | 220 | 0.879 [0.800, 0.947] | 0.548 [0.498, 0.601] | **0.695** [0.565, 0.802] |
| hardware | 11 | 11 (7) | 11 / 0 | 0 | 1.000 | n/a (no normals) | n/a |
| hand | 178 | 54 (27) | 50 / 4 | 81 | 0.926 [0.840, 0.985] | 0.347 [0.262, 0.438] | 0.756 |
| leg | 309 | 30 (9) | 27 / 3 | 112 | 0.900 [0.757, 1.000] | 0.599 [0.528, 0.664] | 0.762 |
| mixed | 61 | 17 (10) | 13 / 4 | 18 | 0.765 [0.545, 0.941] | 0.591 [0.467, 0.727] | 0.715 |
| hip | 27 | 1 (1) | 1 / 0 | 6 | too few | 0.769 | too few |

Full numbers (F1, NPV, Brier, every interval, validation side by side):
`results/phase3_baseline_resnet50/test_metrics.json` and `report.md`.

### What the numbers say

1. **The recall target held.** Test recall 0.892 vs 0.902 on validation; the
   interval [0.825, 0.955] contains 0.90. The slight drop is the expected
   regression from tuning on validation, and it is within noise.
2. **But the price is high.** To catch 9 in 10 fractures, this model flags
   **53 of every 100 X-rays**, and 7 in 10 alarms are false. Its scores for
   the hardest fractures sit near zero: the threshold had to go down to
   0.0048. As a triage tool this baseline would halve a radiologist's
   workload at best. That gap is what Phases 4–5 must close, compared at
   this same 90% recall.
3. **No sign of overfitting to validation in ranking.** Test PR-AUC (0.739)
   and ROC-AUC (0.870) match validation (0.714 / 0.869) within their
   intervals.
4. **Hardware inflates the headline a little.** Removing the 11 hardware
   images drops PR-AUC from 0.739 to 0.695 and recall from 0.892 to 0.879. All
   11 hardware images were fractures and all were caught. The model clearly
   uses metal as evidence; Grad-CAM in Phase 6 will show how much.
5. **Regions differ.** Hand: highest recall but specificity only 0.35 (hand
   X-rays get high scores across the board). Mixed-region images: lowest
   recall (13 of 17). Leg: 27 of 30 with the best specificity. The leg
   result rests on only 9 fracture groups, so its interval is wide.
6. **The intervals are wide** because 102 fractures are really 47 independent
   cases. A future model needs to beat this by more than ~0.1 in PR-AUC
   before the difference is clearly real on this test set.

Predictions per image, figures (PR/ROC val vs test, confusion matrices) and
logs are in the Kaggle notebook output, Version 3
("phase3 step2 test evaluation (one-shot)"), under
`outputs/eval/baseline_resnet50/`.

---

## 1. The protocol

```
Step 1  tune_threshold.py     validation scores only ─► threshold.json
        ── commit + push threshold.json  (the pre-registration) ──
Step 2  evaluate_test.py      checks ─► re-score val ─► score TEST once ─► metrics + CIs
        ── commit the test results ──
```

Step 2 refuses to start unless:

| Check | Catches |
|---|---|
| `threshold.json` is committed and unmodified | choosing or nudging the threshold after seeing test results |
| checkpoint epoch, training commit and split MD5 match the threshold record | evaluating a different model from the one the threshold was tuned for |
| validation re-scored now reproduces PR-AUC (±0.002) and the val counts at the threshold (≤ 2 images flip) | a preprocessing or weights mismatch that would make the threshold meaningless |
| no committed `test_metrics.json` for this model, unless `--rerun-reason` is given (and recorded) | quietly re-running until the number looks good |

Git history then *shows* the order: the threshold commit is older than the
results commit. That is stronger than "trust me, I only looked once".

## 2. Design decisions

**Threshold on validation, never on test.** Any choice made by looking at the
test set (threshold, epoch, model) is tuned to it, and the test score stops
predicting performance on new patients. Validation is already slightly
optimistic (we picked the best epoch on it), so it also absorbs the threshold
choice, and the test set stays untouched.

**Recall first: the highest threshold with validation recall ≥ 0.90.** A
missed fracture can mean an unset bone; a false alarm costs a radiologist a
second look. So the rule fixes the miss rate first (catch at least 90% of
validation fractures) and then, among all thresholds that do, takes the
highest, because that raises the fewest false alarms. 0.90 is in
`configs/eval.yaml`. `tune_threshold.py` prints what 0.80 to 0.98 would each
cost on validation, so the target can be judged before committing. That is
fine: looking at validation is allowed, looking at test is not.

**Midpoint placement.** On validation, every threshold between the 92nd-ranked
fracture's score and the next lower score gives identical results. We take the
middle of that gap so that tiny numeric differences (mixed precision on another
GPU) can't flip an image that sits exactly on the line.

**Cluster bootstrap, not plain bootstrap.** The 102 test fractures come from
only **47** `split_group`s (similar images / adjacent IDs, likely the same
patient). Images in a group are correlated: if the model misses one view, it
often misses the others. Resampling single images would treat 102 fractures as
102 independent cases and give intervals that are too narrow. Resampling whole
groups (2,000 times, percentile 95% intervals) gives the honest width. A unit
test proves it: duplicating each case 5× leaves the clustered interval wide,
while the per-image interval wrongly shrinks by about √5.

**Subgroups.** Every metric is reported for `all`, `no_hardware`, `hardware`
and each body region. Phase 1 found 97 of 99 hardware images are fractured: a
model can score well by spotting metal. `no_hardware` shows how well it finds
fractures without that clue. Subgroups with fewer than 10 fractures (hip,
hardware) get point estimates only. An interval from 1–5 cases is too wide to
mean anything.

**Metrics.** Recall (sensitivity) and precision at the threshold, specificity,
NPV (of the images the model clears, how many are really normal: the number
that matters if it is used to *clear* X-rays), F1, threshold-free ROC-AUC and
PR-AUC, and the Brier score (probability calibration). Raw TP/FN/FP counts are
always shown next to rates. Metrics at 0.5 are kept for comparison with Phase 2.

**What the intervals do not cover.** The threshold is held fixed, so the
intervals reflect test-set sampling noise only, not the uncertainty from
choosing the threshold on 102 validation fractures. Expect test recall to land
*below* 0.90: the threshold was set to just barely reach 0.90 on validation,
and a different sample will usually fall a little short. That is the honest
result, not a bug.

## 3. Files added

| Path | Purpose |
|---|---|
| `src/fracdet/eval/metrics.py` | Confusion counts; full metric set incl. NPV and Brier. |
| `src/fracdet/eval/threshold.py` | All operating points (cumulative sums), the recall rule, reference points (0.5, max-F1, Youden), recall-target table. |
| `src/fracdet/eval/bootstrap.py` | Cluster bootstrap percentile intervals. |
| `src/fracdet/eval/report.py` | Subgroup metrics + intervals; markdown tables. |
| `src/fracdet/eval/protocol.py` | Pre-registration, same-model, val-reproduction and run-once guards. |
| `src/fracdet/eval/predict.py` | Score a split from a checkpoint, using the checkpoint's own preprocessing. |
| `src/fracdet/eval/pipeline.py` | The two steps (`tune`, `evaluate_test`). |
| `src/fracdet/eval/plots.py` | Threshold sweep, PR/ROC val vs test, confusion matrices. |
| `scripts/tune_threshold.py`, `scripts/evaluate_test.py` | Command-line wrappers. |
| `configs/eval.yaml` | The protocol settings (target recall, bootstrap, tolerances). |
| `tests/test_eval.py` | Rule correctness incl. ties, bootstrap behaviour, guards, end-to-end on fake data. |

Also changed: `load_split_frames` now carries `split_group`, and
`val_predictions.csv` of future runs includes it.

## 4. How to run (Kaggle)

1. Open the runner notebook; **Add Input → Your Work →** this notebook,
   **Version 1** (the Phase 2 run). Run the setup cells. **Skip** the Phase 2
   training cell.
2. Step 1 cell: `python scripts/tune_threshold.py --find-run baseline_resnet50`.
   Read the recall-target table.
3. Download `outputs/eval/baseline_resnet50/tune/*` into
   `results/phase3_baseline_resnet50/` locally, commit, push.
4. Step 2 cell (GPU): `git pull`, then
   `python scripts/evaluate_test.py --find-run baseline_resnet50`.
5. Download `outputs/eval/baseline_resnet50/test/*` into the same results
   folder, commit, push. Quick Save the notebook version.

---

## 5. Interview questions

**Q: Why not just use 0.5?**
0.5 assumes both errors cost the same and the classes are balanced. Neither is
true here: 17% fractures, and a miss is worse than a false alarm. At 0.5 the
baseline missed 44% of validation fractures.

**Q: Why tune the threshold on validation and not on test?**
Anything tuned on the test set is fitted to it, so the test score would no
longer predict performance on new X-rays. Test is used once, at the end, with
everything already fixed.

**Q: How do you prove you didn't peek at the test set?**
The threshold file is committed before the test run, and the evaluation script
refuses to run if it isn't. It also refuses a second run without a written
reason. The commit order in git history is the evidence.

**Q: Why 90% recall?**
For screening, the miss rate is the first constraint. 90% is a common
sensitivity target for triage tools, and the validation table shows its cost
(precision and alarms per 100 X-rays) next to 80–98%, so the choice is
explicit, made on validation, and easy to change for another use case.

**Q: What is a confidence interval here, and why bootstrap?**
The range of test scores we would plausibly get from a different sample of
patients. Bootstrap resamples the test set many times and takes the middle 95%
of the recomputed metric. It works for any metric without a formula.

**Q: Why resample groups instead of images?**
Several images often come from the same patient. They aren't independent, so
treating them as such overstates how much evidence we have. 102 fractures from
47 groups carry roughly the information of 47 cases, and the clustered interval
reflects that.

**Q: Test recall came out below the 90% you targeted. Is that a failure?**
No, it's expected. The threshold was placed to *just* reach 90% on validation,
and validation is one random sample. On a new sample, about half the time you
land below. The interval shows whether the gap is within noise.

**Q: Why report results without hardware images?**
Almost every image with surgical hardware is fractured, so "metal → fracture"
is a shortcut. If performance holds up without those images, the model is
finding fractures, not metal.

---

<!-- nav -->
← [Phase 2: Baseline model](phase-2-baseline.md) · [Project README](../README.md) · [Phase 4: Class imbalance](phase-4-imbalance.md) →
