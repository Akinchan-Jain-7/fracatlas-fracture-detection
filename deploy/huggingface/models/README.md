---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - medical-imaging
  - radiology
  - x-ray
  - fracture-detection
  - object-detection
  - grad-cam
datasets:
  - FracAtlas
---

# Fracture Check models (FracAtlas)

Two PyTorch models for finding bone fractures in musculoskeletal X-rays, used by the
[Fracture Check](https://huggingface.co/spaces/akinchanJain7/fracture-check) demo.

| File | Model | Job |
|---|---|---|
| `classifier_convnext_tiny_s44.pt` | ConvNeXt-Tiny (timm `convnext_tiny.fb_in22k_ft_in1k`), 384 px grayscale | Does the X-ray show a fracture? |
| `detector_s42_best.pt` | Faster R-CNN v2, ResNet-50 + FPN (torchvision), COCO-pretrained | Where is the fracture? (boxes) |

**A student research project, not a medical device.** Not for diagnosis or for
deciding whether someone should seek care.

## How they are used together

The classifier decides *whether*: an X-ray scoring at least **0.0968** is flagged. That
threshold was fixed on validation data for 90% recall. The detector decides *where*, and
its boxes are shown only on flagged X-rays. A Grad-CAM heatmap from the classifier's last
stage shows where the classifier found evidence.

## Test results

On the held-out FracAtlas test split (589 X-rays, 102 fractured, 230 image groups), each
scored once with the analysis plan committed beforehand. Intervals are 95% bootstrap
intervals over image groups.

| Classifier (threshold 0.0968) | |
|---|---|
| Fractures caught (recall) | 94 / 102 = 0.922 [0.859, 0.976] |
| Fracture-free X-rays cleared (specificity) | 0.708 [0.643, 0.764] |
| Flagged X-rays that had a fracture (precision) | 0.398 |
| Cleared X-rays without a fracture (NPV) | 0.977 |
| ROC-AUC / PR-AUC | 0.937 / 0.853 |

| Detector (seed 42) | |
|---|---|
| Top box centre inside a fracture box | 0.745 [0.650, 0.837] |
| AP50, normal X-rays included | 0.421 [0.296, 0.528] |
| Fracture boxes found at 1 false box per image | 0.635 |
| Of the 94 flagged fractures, top box on the fracture | 72 |

## Limitations

- **Surgical hardware.** Plates and screws push the classifier's score up, and the
  detector tends to box the implant (4 of 11 hardware fractures boxed correctly on test).
- **Children.** Growth plates near the ends of long bones cause false alarms.
- **Scope.** Plain X-rays of hands, arms, legs, hips and shoulders only, from one
  dataset (three hospitals in Bangladesh). Performance elsewhere is unknown.
- **Precision is low by design.** The threshold favours catching fractures, so most
  flagged X-rays in practice are not fractured.

## Loading

The files are checkpoints written by the project's own code (weights plus training
config and provenance). Load them with the `fracdet` package from the GitHub repo:

```python
from fracdet.serve.pipeline import FracturePipeline

pipeline = FracturePipeline.from_dir("path/to/folder/with/both/files")
result = pipeline.predict(open("xray.jpg", "rb").read())
print(result["probability"], result["flagged"], result["boxes"])
```

`FracturePipeline` checks each file's SHA-256 and its training provenance (epoch, git
commit, data-split checksum) before use.

| File | SHA-256 |
|---|---|
| `classifier_convnext_tiny_s44.pt` | `f54efa2468c56c29208295db1df5c79d7eef878965fefea045c2dbcc09ea2784` |
| `detector_s42_best.pt` | `4a87100d08b1c998c5599016755271dfbda661d8b38bbee3494aef2863ab24c3` |

Full methods, evaluation protocol and per-image results:
[github.com/Akinchan-Jain-7/fracatlas-fracture-detection](https://github.com/Akinchan-Jain-7/fracatlas-fracture-detection).

## Training data

FracAtlas: Abedeen, I., Rahman, M. A., Prottyasha, F. Z., et al. *FracAtlas: A Dataset
for Fracture Classification, Localization and Segmentation of Musculoskeletal
Radiographs.* Scientific Data 10, 521 (2023). Licensed CC BY 4.0. The images are not
included here.

Author: Akinchan Jain, B.Tech CSE (AI), Bennett University.
