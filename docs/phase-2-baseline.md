# Phase 2: Baseline model

**Goal:** a correct, reproducible training pipeline and a first honest number:
an ImageNet-pretrained ResNet-50 fine-tuned on `split_v1`, with no
imbalance tricks. Every later improvement (Phase 4 imbalance handling, Phase 5
backbones) is measured against this run.

**The test set is not used in this phase.** All numbers below are on
**validation** (586 images, 102 fractured).

<!-- RESULTS -->

---

## 1. Files added

| Path | Purpose |
|---|---|
| `src/fracdet/data/transforms.py` | Grayscale + fast JPEG draft decode, letterbox to a square, augmentation, normalisation; one config dict builds train and eval pipelines. |
| `src/fracdet/data/dataset.py` | `load_split_frames` (reads the committed split; only the splits you ask for) and `FracAtlasDataset` returning `(image, label, index)`. |
| `src/fracdet/models/classifier.py` | `timm` backbone with a single-logit head. |
| `src/fracdet/train/metrics.py` | PR-AUC, ROC-AUC, recall / precision / specificity at 0.5. |
| `src/fracdet/train/schedule.py` | Per-batch linear warmup + cosine decay. |
| `src/fracdet/train/engine.py` | One training epoch (AMP, grad clipping) and one prediction pass. |
| `src/fracdet/train/checkpoint.py` | Checkpoint = weights + model config + preprocessing + provenance; atomic save. |
| `src/fracdet/train/fit.py` | The driver: loaders, optimiser, epochs, early stopping, best checkpoint, run record. |
| `scripts/train.py`, `scripts/show_run.py` | Train from a config; print a run's table. |
| `configs/baseline_resnet50.yaml` | This run's settings. |
| `tests/test_dataset_model.py`, `tests/test_training.py` | Transform geometry, RGB==grayscale, determinism, metric values, schedule shape, checkpoint round-trip, tiny-batch overfit, end-to-end `fit`, and "test split never requested". |

---

## 2. Design decisions

**Grayscale for every image.** FracAtlas mixes RGB and grayscale files, and
grayscale files are more often fractured (24% vs 16%, Phase 1). Converting all
of them to one channel removes that shortcut. The channel is then repeated ×3
because the network was pretrained on colour ImageNet images. A test checks
that an RGB file and its grayscale version give identical tensors.

**Letterbox, not stretch.** Resize the long side to 384 px and pad to a square
with black. Stretching a 373×454 X-ray into a square would distort bone
geometry. 384 stays close to the native resolution of most images (454 px),
because fractures can be hairline-thin.

**Augmentation without random crops.** Flip, ±10° rotation, ±5% shift, 0.9–1.1
zoom, brightness/contrast ±20%: plausible variations of a real radiograph. A
random crop can remove the fracture while the label still says "fractured",
which teaches the model wrong labels.

**Transfer learning.** ResNet-50 pretrained on ImageNet already detects edges,
textures and shapes. We replace its 1,000-class head with one output (the
fracture logit) and fine-tune the whole network. 3.3k training images are far
too few to learn those filters from scratch.

**One logit + `BCEWithLogitsLoss`.** Binary task → one probability,
`sigmoid(z)`. The loss fuses sigmoid and log for numerical stability (no
`log(0)`).

**AdamW, no weight decay on biases/normalisation.** Weight decay regularises
the conv and linear weights; on BatchNorm scales and biases it only distorts
the normalisation, so they are exempt (standard practice).

**Warmup + cosine.** The new head starts random. One epoch of LR warmup keeps its
large early gradients from wrecking the pretrained backbone. Cosine decay
then lowers the LR smoothly to 1% of its peak.

**Mixed precision (AMP).** Float16 matrix maths on the T4's tensor cores,
float32 weights; `GradScaler` prevents tiny float16 gradients from underflowing
to zero. Gradients are clipped *after* un-scaling, so the clip threshold applies
to the real gradient.

**Best epoch by validation PR-AUC.** Not accuracy (82% for a model that never
predicts a fracture) and not loss (it can improve while ranking gets worse).
PR-AUC is threshold-free and focused on the positive class. Early stopping
after 6 epochs without improvement.

**Checkpoints carry their own preprocessing.** `best.pt` stores the weights,
model name, image size and normalisation, git commit, split name and split-file
MD5. The web app (Phase 8) loads one file and rebuilds everything, so no
constants are copied between training and serving. After training, `fit`
reloads `best.pt` from disk and re-scores validation, which proves the file
reproduces the reported number.

**The test split is never loaded.** `fit` asks only for `train` and `val`, and a
unit test spies on the loader to prove it.

---

## 3. Bugs and surprises during the build

- **Slow image loading (110 ms/image).** Full decode of 2880-px X-rays only to
  shrink them to 384 px. Fixed with JPEG draft decoding (decode directly at
  1/2, 1/4 or 1/8 scale): about 11 ms/image including augmentation.
- **`NaN` in `summary.json`.** Precision is undefined when the model predicts no
  positives (common in early epochs). Python's `json` writes `NaN`, which is
  not valid JSON. Now written as `null`, with `allow_nan=False` as a guard.
- **The accuracy trap, seen live.** A smoke run of an untrained model scored
  **83% accuracy with 0% recall** by predicting "no fracture" for everything.
