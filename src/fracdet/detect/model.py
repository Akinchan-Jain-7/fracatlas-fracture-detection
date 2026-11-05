"""The detector: torchvision Faster R-CNN v2 with a 2-class head.

Faster R-CNN in one paragraph: a ResNet-50 + FPN backbone turns the image into
feature maps at several scales. A Region Proposal Network slides over them and
proposes boxes likely to contain *something*. Each proposal's features are
pooled to a fixed size (RoIAlign) and a small head classifies it (fracture vs
background) and refines its coordinates. Training minimises four losses: RPN
objectness and box regression, head classification and box regression.

COCO weights give a working proposal + refinement machine; only the final
classifier layer is replaced (91 COCO classes -> background + fracture).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

SUPPORTED = {
    "fasterrcnn_resnet50_fpn_v2": (
        detection.fasterrcnn_resnet50_fpn_v2,
        detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    ),
    "fasterrcnn_resnet50_fpn": (
        detection.fasterrcnn_resnet50_fpn,
        detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    ),
}


def build_detector(cfg: dict[str, Any]) -> nn.Module:
    """``cfg`` = the ``detector`` section of configs/detect.yaml."""
    if cfg["name"] not in SUPPORTED:
        raise ValueError(f"Unknown detector {cfg['name']!r}; supported: {sorted(SUPPORTED)}")
    builder, weights = SUPPORTED[cfg["name"]]
    model = builder(
        weights=weights if cfg.get("pretrained", True) else None,
        weights_backbone=None,
        min_size=cfg.get("min_size", 800),
        max_size=cfg.get("max_size", 1333),
        box_detections_per_img=cfg.get("detections_per_img", 20),
        box_score_thresh=cfg.get("score_thresh", 0.01),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.get("num_classes", 2))
    return model


@torch.no_grad()
def detect(model: nn.Module, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Boxes (pixels of each input image) and scores, highest score first."""
    model.eval()
    out = model(images)
    return [{"boxes": o["boxes"].float().cpu(), "scores": o["scores"].float().cpu()} for o in out]
