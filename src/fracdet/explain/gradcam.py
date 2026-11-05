"""Grad-CAM: which parts of the X-ray pushed the fracture score up.

How it works, for one image and one layer with activations ``A`` (C maps of
h x w) and the model's fracture logit ``z``:

1. Forward pass, keeping ``A``.
2. Gradient ``dz/dA``: for every channel and position, how much ``z`` would rise
   if that activation rose.
3. Channel weight ``w_c`` = the average of that gradient over the h x w grid:
   how much channel c, as a whole, argues for "fracture".
4. Map = ReLU( sum_c w_c * A_c ). Positions where the channels that argue for
   fracture are active. ReLU keeps only evidence *for* fracture.
5. Upsample the h x w map to the input size to overlay it.

Resolution is the main limitation: the last ConvNeXt stage is 12 x 12 at
384 px, so one cell covers 32 x 32 pixels. Grad-CAM shows the region the model
used, not the exact fracture line.

Batches are explained in one backward pass by differentiating the *sum* of the
logits. That equals per-image Grad-CAM because, in eval mode, no layer mixes
images within a batch (LayerNorm and BatchNorm both act per image then).
A test checks this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from fracdet.data.transforms import LetterboxGeometry, eval_transform_with_geometry


def feature_layers(model: nn.Module) -> list[str]:
    """Module names of the backbone's stages, shallow to deep (from timm's feature_info)."""
    info = getattr(model, "feature_info", None)
    if not info:
        raise ValueError("model has no timm feature_info; pass layer names explicitly")
    return [f["module"] for f in info]


def resolve_layer(model: nn.Module, layer: int | str) -> str:
    """``-1`` -> last stage name, ``-2`` -> one before, a string -> itself."""
    return feature_layers(model)[layer] if isinstance(layer, int) else layer


@dataclass
class GradCAMResult:
    logits: torch.Tensor  # (N,) fracture logits, fp32, on CPU
    cams: dict[str, torch.Tensor]  # layer -> (N, h, w), >= 0, on CPU


def cam_from(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Grad-CAM map (N, h, w) from activations and gradients of shape (N, C, h, w)."""
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
    return F.relu((weights * activations).sum(dim=1))


class GradCAM:
    """Grad-CAM for one or more layers from a single forward + backward pass.

    Usage::

        cam = GradCAM(model, ["stages.3", "stages.2"])
        result = cam(x)          # x: (N, 3, H, W) on the model's device
        result.cams["stages.3"]  # (N, 12, 12)
    """

    def __init__(self, model: nn.Module, layers: Sequence[str]) -> None:
        modules = dict(model.named_modules())
        missing = [n for n in layers if n not in modules]
        if missing:
            raise KeyError(f"layers {missing} not in model; stages are {feature_layers(model)}")
        self.model = model
        self.layers = list(layers)
        self._modules = [modules[n] for n in self.layers]

    def __call__(self, x: torch.Tensor) -> GradCAMResult:
        self.model.eval()
        acts: dict[str, torch.Tensor] = {}

        def keep(name: str):
            def hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"{name} returns {type(output).__name__}, not a tensor")
                acts[name] = output

            return hook

        handles = [
            m.register_forward_hook(keep(n))
            for n, m in zip(self.layers, self._modules, strict=True)
        ]
        try:
            with torch.enable_grad():
                # The input needs a gradient so the graph is built even if the
                # weights are frozen. autograd.grad then differentiates only with
                # respect to the activations: no .grad is written on any weight.
                x = x.detach().float().requires_grad_(True)
                logits = self.model(x).squeeze(1).float()
                grads = torch.autograd.grad(logits.sum(), [acts[n] for n in self.layers])
        finally:
            for h in handles:
                h.remove()
        cams = {
            n: cam_from(acts[n].detach().float(), g.float()).cpu()
            for n, g in zip(self.layers, grads, strict=True)
        }
        return GradCAMResult(logits.detach().cpu(), cams)


def upsample(cam: torch.Tensor | np.ndarray, size: int) -> np.ndarray:
    """(h, w) or (N, h, w) map -> same at ``size x size``, bilinear, as float32 numpy."""
    t = torch.as_tensor(np.asarray(cam, dtype=np.float32))
    squeeze = t.dim() == 2
    t = t[None] if squeeze else t
    up = F.interpolate(t[:, None], size=(size, size), mode="bilinear", align_corners=False)
    out = up[:, 0].numpy()
    return out[0] if squeeze else out


def to_original(cam_up: np.ndarray, geom: LetterboxGeometry, size: tuple[int, int]) -> np.ndarray:
    """A letterbox-frame map -> the original image's (width, height), padding cut off."""
    rows, cols = geom.content
    crop = Image.fromarray(np.ascontiguousarray(cam_up[rows, cols], dtype=np.float32), mode="F")
    return np.asarray(crop.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def explain_image(
    model: nn.Module,
    img: Image.Image,
    preprocess: dict[str, Any] | None = None,
    layer: int | str = -1,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Probability and Grad-CAM heatmap for one PIL image, on its own pixel grid.

    This is the call the web app (Phase 8) will make. Preprocessing comes from
    the checkpoint (``ckpt["preprocess"]``), exactly as in training and evaluation.

    Returns ``{"prob", "heatmap", "layer"}``; ``heatmap`` has the original
    image's height x width, scaled to 0..1 (all zeros if there was no evidence
    for fracture anywhere).
    """
    name = resolve_layer(model, layer)
    x, geom = eval_transform_with_geometry(preprocess)(img)
    result = GradCAM(model, [name])(x[None].to(device))
    cam = to_original(upsample(result.cams[name][0], geom.size), geom, img.size)
    peak = float(cam.max())
    return {
        "prob": float(torch.sigmoid(result.logits[0])),
        "heatmap": cam / peak if peak > 0 else cam,
        "layer": name,
    }
