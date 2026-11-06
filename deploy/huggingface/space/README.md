---
title: Fracture Check
emoji: 🦴
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Screen hand, arm, leg and hip X-rays for possible fractures
models:
  - akinchanJain7/fracture-detection-model
---

# Fracture Check

Upload an X-ray of a hand, arm, leg, hip or shoulder. A ConvNeXt-Tiny classifier
estimates whether it shows a fracture, a Grad-CAM heatmap shows where it looked,
and a Faster R-CNN detector marks where the break most likely is. The page also
explains how to read the result and includes a general fracture care guide.

**A student research project, not a medical device.** It must not be used to
decide whether to seek care; a doctor must read every X-ray.

- Code, methods and every result: [github.com/Akinchan-Jain-7/fracatlas-fracture-detection](https://github.com/Akinchan-Jain-7/fracatlas-fracture-detection)
- Models and model card: [akinchanJain7/fracture-detection-model](https://huggingface.co/akinchanJain7/fracture-detection-model)

Test-set results (589 X-rays, 102 fractured, scored once): the classifier caught
94 of 102 fractures and cleared 71% of fracture-free X-rays; the detector's top
box landed on the fracture in 75% of fractured X-rays.

This Space runs the project's FastAPI app (not a Gradio interface): `app.py`
downloads the code from GitHub and the models from the model repo, then serves it.

Trained on FracAtlas: Abedeen, I., Rahman, M. A., Prottyasha, F. Z., et al.
*FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of
Musculoskeletal Radiographs.* Scientific Data 10, 521 (2023). CC BY 4.0.

Built by Akinchan Jain, B.Tech CSE (AI), Bennett University.
