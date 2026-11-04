# Phase 5: Model comparison (resolution and backbone)

**Goal:** Phase 4 showed the remaining errors are not about class balance; the
hardest fractures look normal to the model. Phase 5 asks whether a model that
**sees more** does better: more input pixels, or a different architecture or
pretraining. The yardstick is the same as before: validation specificity at
90% recall, 3 seeds per variant, selection rule committed before the runs.

## Results

*Pending: filled in from `results/phase5_models/` after the Kaggle sweep.*

---

## 1. The variants

| Variant | Config | Params | What it tests |
|---|---|---|---|
| baseline | `baseline_resnet50.yaml` | 23.5M | ResNet-50 at 384 px (re-run with the same code and seeds) |
| ResNet-50 at 448 | `p5_resnet50_448.yaml` | 23.5M | **Resolution**: keeps nearly all original pixels |
| DenseNet-121 | `p5_densenet121.yaml` | 7.0M | The CheXNet / MURA backbone for X-rays; dense feature reuse |
| EfficientNet-B2 | `p5_efficientnet_b2.yaml` | 7.7M | A smaller, parameter-efficient model; less room to overfit 2.7k images |
| ConvNeXt-Tiny | `p5_convnext_tiny.yaml` | 27.8M | A modern CNN with ImageNet-22k pretraining (14M images) |

**Why 448 and not 512 or 640.** 91% of FracAtlas images are 373×454 px
(Phase 1). At 384 px the long side shrinks 15%; at 448 almost nothing is
lost. Anything larger only upscales, adding compute but no detail, so a
"bigger is better" sweep would mostly measure interpolation. The ~9% of large
originals (2304×2880) gain detail at 448 too.

## 2. Design

- **Same recipe for every variant.** The optimiser, LR (2e-4), schedule,
  augmentation, batch size, early stopping and seeds are all the same; only
  `model.name` *or* `image_size` changes. A test enforces this.
  **Limitation:** nothing is tuned per backbone. A model that prefers a
  different LR may be under-sold. This is an equal-budget comparison, not a
  best-possible one.
- **Same selection rule as Phase 4** (`configs/sweep_phase5.yaml`): a variant
  wins only if its mean paired gain in specificity at 90% recall exceeds the
  seed noise, without losing PR-AUC beyond its noise. If none wins, the
  baseline stays. The winner is represented by its median seed.
- **Multiple comparisons.** With four challengers, one may clear the bar by
  luck. That is why a winner still gets exactly one test evaluation (Phase 3
  protocol) before it replaces the baseline.
- **Shortcut check (reported, not decisive).** PR-AUC on validation images
  without surgical hardware, per variant. A backbone that wins mostly by
  spotting metal would show a bigger drop there.

## 3. Running it

`notebooks/kaggle_sweep.ipynb` runs any sweep file (set `SWEEP` at the top):

```bash
python scripts/sweep.py configs/sweep_phase5.yaml --gpus 0,1 --set train.num_workers=2
python scripts/compare_runs.py configs/sweep_phase5.yaml
```

15 runs. On Kaggle T4 x2 about 2–3 h (448 px and ConvNeXt are slower than the baseline).

## 4. Interview questions

**Q: Why not just use the biggest model?**
With 2.7k training images, capacity isn't the bottleneck; generalisation is.
A larger model can memorise faster. The comparison measures that directly,
over seeds, instead of assuming it.

**Q: Why did you test 448 px rather than 512 or 640?**
Most images are 454 px on the long side. Resizing beyond the native
resolution invents pixels by interpolation. 448 removes the 15% downscale of
384 without pretending to add detail.

**Q: Why is DenseNet-121 a natural candidate for X-rays?**
It was the backbone of CheXNet (chest X-rays) and of the MURA baseline
(musculoskeletal X-rays, the closest public benchmark to FracAtlas).

**Q: Isn't comparing five models on one validation set risky?**
Yes. The more candidates, the more likely one looks better by chance. I use
3 seeds, a gain threshold set by the seed noise, a rule fixed in advance,
and a single confirming test evaluation for any winner.

**Q: Why the same learning rate for every model?**
To keep the comparison equal-budget and avoid tuning on validation for each
model, which would favour whoever got the most tuning. The downside is
stated: a model that needs a different LR could look worse than it is.
