## test: detector vs Grad-CAM boxes

Detector `20261105-081309_detect_frcnn_r50_s42` (seed 42, epoch 5). 589 images, 102 fractured, 148 fracture boxes. 95% CIs resample patient groups.

| Metric | Detector | Grad-CAM boxes |
|---|---|---|
| AP50, all images (primary) | 0.421 [0.296, 0.528] | 0.001 [0.000, 0.003] |
| AP50, fractured images only | 0.508 [0.388, 0.615] | 0.001 [0.000, 0.003] |
| AP at IoU 0.25, all images | 0.553 [0.418, 0.663] | 0.024 [0.005, 0.073] |
| Fracture boxes found at 0.5 false boxes / image | 0.554 | 0.014 |
| Fracture boxes found at 1 false box / image | 0.635 [0.535, 0.746] | 0.014 [0.000, 0.032] |
| Fracture boxes found at 2 false boxes / image | 0.703 | 0.014 |
| Top box centre in a fracture box (pointing) | 0.745 [0.650, 0.837] | 0.304 [0.203, 0.410] |
| Top box IoU >= 0.5 | 0.559 [0.440, 0.670] | 0.020 [0.000, 0.047] |

Top-1 pointing by region (fractured images):

| Region | n | Detector | Grad-CAM |
|---|---|---|---|
| hand | 54 | 0.741 | 0.222 |
| hip | 1 | 1.000 | 1.000 |
| leg | 30 | 0.767 | 0.400 |
| mixed | 17 | 0.706 | 0.353 |

End to end: {'fractures': 102, 'flagged_by_classifier': 94, 'flagged_and_detector_top1_pointing': 72, 'flagged_and_detector_top1_iou50': 54}

Protocol: {"plan": "configs/detect.yaml", "plan_committed": true, "plan_commit": "8b672437c9d17e470a3fe2e6ed6081767cd1c802 2026-11-05T08:01:41+00:00", "rerun_reason": null, "classifier_counts": {"tp": 94, "fp": 142, "tn": 345, "fn": 8}, "classifier_committed_counts": {"tp": 94, "fp": 142, "tn": 345, "fn": 8}, "code": {"commit": "8b672437c9d17e470a3fe2e6ed6081767cd1c802", "dirty": false}, "evaluated_at": "2026-11-05T08:56:01+0000", "device": "Tesla T4", "seconds": 95.2}
