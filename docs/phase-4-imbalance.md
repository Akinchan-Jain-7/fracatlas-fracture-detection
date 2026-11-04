# Phase 4: Class imbalance

**Goal:** find out whether the standard imbalance remedies make the model
better at the job that matters. The yardstick: **how many normal X-rays get
flagged when we insist on catching 90% of fractures** (specificity at recall
0.90 on validation). Phase 3 measured the baseline there: about 53 alarms per
100 X-rays.

**The test set is not used to compare variants.** Picking the best of several
models by their test score would make that score optimistic (we'd be selecting
on noise). Variants are compared on validation; only a winner, if there is one,
gets a single test evaluation with the Phase 3 protocol.

## Results

*Pending: filled in from `results/phase4_imbalance/` after the Kaggle sweep.*

---

## 1. The variants

All four share everything (model, preprocessing, augmentation, optimiser,
schedule, early stopping on validation PR-AUC). Only the handling of imbalance
differs.

| Variant | Config | What changes |
|---|---|---|
| baseline | `baseline_resnet50.yaml` | Plain BCE; every image counts once (17.6% fractured). |
| weighted loss | `imb_pos_weight.yaml` | Each fractured image's loss × n_normal / n_fractured (≈ 4.7), so both classes weigh the same in total. |
| balanced sampler | `imb_balanced_sampler.yaml` | Batches drawn ~50/50 with replacement; same steps per epoch. Fractures are seen ~2.8× per epoch. |
| focal loss | `imb_focal.yaml` | `(1 − p_t)^γ · BCE`, γ = 2, no class weights: easy images (either class) are down-weighted, hard ones dominate. |

Implementation: `src/fracdet/train/imbalance.py`. The validation loss is
always plain BCE, so `val_loss` stays comparable between variants.

## 2. Experimental design

**3 seeds per variant, baseline re-run too.** One training run per variant
can't separate a real effect from luck: with 102 validation fractures,
seed-to-seed swings of ±0.02 in PR-AUC are normal. All 12 runs use the same
code and the same seeds (42, 43, 44). The Phase 2 run is *not* reused, so the
comparison is like-for-like.

**Paired by seed.** Each variant is compared with the baseline run that has
the same seed (same initial head weights, same augmentation stream), and the
differences are averaged. Pairing removes the part of the variation both runs
share.

**Selection rule, fixed before the sweep** (`configs/sweep_phase4.yaml`):

- Primary metric: validation specificity at the threshold that reaches 90%
  recall (the Phase 3 rule).
- A variant **wins** only if its mean paired gain is larger than the
  seed-to-seed noise (the larger standard deviation of baseline and variant),
  **and** it loses no more than the noise in PR-AUC.
- If several variants win, the one with the largest gain is chosen.
- If none wins, **the baseline stays**. "No improvement" is a valid result.
- The winner's representative checkpoint is its **median** seed, not its best
  seed. The best of 3 would be an optimistic pick.

**What to expect.** Weighting and sampling mostly push fracture scores up,
which a tuned threshold already compensates for. The usual finding is little
change in *ranking*. Focal loss is the one most likely to help our specific
failure: Phase 3 found the hardest fractures scored near 0.

## 3. Running it

```bash
# all 12 runs, one queue per GPU (Kaggle T4 x2), ~2.5 h
python scripts/sweep.py configs/sweep_phase4.yaml --gpus 0,1 --set train.num_workers=2
# validation comparison + decision, from the runs' own files
python scripts/compare_runs.py configs/sweep_phase4.yaml
```

`sweep.py` runs jobs seed-major (all variants for seed 42 first), so a sweep
that stops early still has complete seeds. A failed job is recorded in
`manifest.json` and the sweep continues.

## 4. Interview questions

**Q: Your data is 17% positive. Why not just oversample?**
I tried it, alongside loss weighting and focal loss, under the same budget and
seeds, and judged them on specificity at a fixed recall, because a threshold
already corrects the class prior. Rebalancing that only shifts scores changes
nothing once the threshold is tuned. Only changes in ranking matter.

**Q: What does pos_weight actually do?**
It multiplies the loss of every positive example. With n_neg/n_pos ≈ 4.7, the
total loss from fractures equals the total from normals, so the gradient no
longer mostly says "predict normal".

**Q: How is focal loss different from class weighting?**
Class weighting scales a whole class. Focal loss scales each example by how
wrong the model is: confident correct predictions get almost no weight, so
training concentrates on hard cases from either class. With γ = 0 it is plain
BCE.

**Q: Why three seeds?**
Run-to-run noise from initialisation and data order is the same size as the
effects being measured. With one run per variant, I could "win" by luck.
Three seeds give a mean and a spread, and pairing by seed sharpens the
comparison.

**Q: Why not compare on the test set?**
Choosing the best of four models by test score turns the test set into a
validation set, and the reported number becomes optimistic. Selection happens
on validation; the test set is used once, for the chosen model.

**Q: Why the median seed and not the best one?**
The best of three is partly luck, so its score would overstate what the method
delivers. The median is a fair representative of the method.
