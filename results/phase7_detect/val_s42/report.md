## val: detector vs Grad-CAM boxes

Detector `20261105-081309_detect_frcnn_r50_s42` (seed 42, epoch 5). 586 images, 102 fractured, 140 fracture boxes. 95% CIs resample patient groups.

| Metric | Detector | Grad-CAM boxes |
|---|---|---|
| AP50, all images (primary) | 0.546 [0.420, 0.654] | 0.001 [0.000, 0.013] |
| AP50, fractured images only | 0.608 [0.506, 0.706] | 0.001 [0.000, 0.014] |
| AP at IoU 0.25, all images | 0.641 [0.525, 0.731] | 0.019 [0.005, 0.057] |
| Fracture boxes found at 0.5 false boxes / image | 0.657 | 0.021 |
| Fracture boxes found at 1 false box / image | 0.721 [0.637, 0.805] | 0.021 [0.000, 0.063] |
| Fracture boxes found at 2 false boxes / image | 0.764 | 0.021 |
| Top box centre in a fracture box (pointing) | 0.794 [0.712, 0.876] | 0.343 [0.238, 0.459] |
| Top box IoU >= 0.5 | 0.686 [0.585, 0.782] | 0.029 [0.000, 0.084] |

Top-1 pointing by region (fractured images):

| Region | n | Detector | Grad-CAM |
|---|---|---|---|
| hand | 54 | 0.852 | 0.389 |
| hip | 2 | 0.500 | 0.000 |
| leg | 31 | 0.806 | 0.258 |
| mixed | 15 | 0.600 | 0.400 |

End to end: {'fractures': 102, 'flagged_by_classifier': 92, 'flagged_and_detector_top1_pointing': 74, 'flagged_and_detector_top1_iou50': 64}

Protocol: {"plan": "configs/detect.yaml", "plan_committed": true, "plan_commit": "8b672437c9d17e470a3fe2e6ed6081767cd1c802 2026-11-05T08:01:41+00:00", "rerun_reason": null, "classifier_counts": {"tp": 92, "fp": 131, "tn": 353, "fn": 10}, "classifier_committed_counts": {"tp": 92, "fp": 131, "tn": 353, "fn": 10}, "code": {"commit": "8b672437c9d17e470a3fe2e6ed6081767cd1c802", "dirty": false}, "evaluated_at": "2026-11-05T08:52:49+0000", "device": "Tesla T4", "seconds": 86.5}
