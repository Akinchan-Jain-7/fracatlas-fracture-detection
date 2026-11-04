# Phase 3: Proper evaluation

**Goal:** turn the Phase 2 model into one honest, defensible test number:
a decision threshold chosen on validation only and fixed in git *before* the test
set is opened, then a single test pass reported with confidence intervals, with
and without surgical-hardware images.

No new training happens in this phase. The model is exactly Phase 2's `best.pt`
(epoch 16, commit `a579c9f`).

## Results

*Pending: filled in from `results/phase3_baseline_resnet50/` after the Kaggle run.*

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
