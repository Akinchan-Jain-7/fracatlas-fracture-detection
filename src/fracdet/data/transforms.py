"""Image preprocessing and augmentation.

Two pipelines are built from ONE config dict (``cfg["preprocess"]`` +
``cfg["augment"]``). That dict is stored inside every checkpoint, so inference
(Phase 8) rebuilds exactly the preprocessing the model was trained with.

Pipeline (eval):  grayscale (fast JPEG draft decode) -> letterbox to a square
                  -> tensor -> 3 channels -> normalise
Pipeline (train): same, plus geometric + intensity augmentation before the tensor step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_PREPROCESS: dict[str, Any] = {
    "image_size": 384,
    "pad_value": 0,  # X-ray background is black; pad with black
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
}
DEFAULT_AUGMENT: dict[str, Any] = {
    "hflip_p": 0.5,
    "rotate_deg": 10.0,
    "translate": 0.05,
    "scale": [0.9, 1.1],
    "brightness": 0.2,
    "contrast": 0.2,
}


class Letterbox:
    """Resize so the LONG side equals ``size``, then pad to ``size x size``.

    Keeps the aspect ratio: squashing a 373x454 X-ray into a square would
    distort bone geometry the model has to judge. Padding goes on both sides
    so the image stays centred.
    """

    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new(img.mode, (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"Letterbox(size={self.size}, fill={self.fill})"


class DraftGrayscale:
    """Decode as single-channel 'L', at reduced scale when the file allows it.

    - Grayscale for every image: FracAtlas mixes RGB and grayscale files, and
      grayscale files are more often fractured (24% vs 16%). Converting all of
      them removes that shortcut.
    - JPEG "draft" mode decodes directly at 1/2, 1/4 or 1/8 scale, choosing the
      largest reduction that keeps both sides >= ``min_side``. A 2304x2880
      X-ray we are about to shrink to 384 px decodes ~10x faster, with no
      visible loss because we downscale afterwards anyway.

    Must be the first step (draft only works before pixels are loaded). Train
    and inference both use it, so they see identical pixels.
    """

    def __init__(self, min_side: int) -> None:
        self.min_side = min_side

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.format == "JPEG":
            img.draft("L", (self.min_side, self.min_side))
        return img.convert("L")

    def __repr__(self) -> str:
        return f"DraftGrayscale(min_side={self.min_side})"


def _repeat_to_3ch(x: torch.Tensor) -> torch.Tensor:
    # ImageNet-pretrained networks expect 3 channels; repeat the gray one.
    return x.expand(3, -1, -1) if x.shape[0] == 1 else x


def build_transforms(
    preprocess: dict[str, Any] | None = None,
    augment: dict[str, Any] | None = None,
    train: bool = False,
) -> Callable[[Image.Image], torch.Tensor]:
    p = {**DEFAULT_PREPROCESS, **(preprocess or {})}
    a = {**DEFAULT_AUGMENT, **(augment or {})}

    steps: list[Callable] = [
        DraftGrayscale(p["image_size"]),
        Letterbox(p["image_size"], p["pad_value"]),
    ]
    if train:
        steps += [
            v2.RandomHorizontalFlip(p=a["hflip_p"]),
            # One affine op = one interpolation (stacking rotate+scale+shift blurs more).
            # No random crops: a crop can cut the fracture out while the label
            # still says "fractured".
            v2.RandomAffine(
                degrees=a["rotate_deg"],
                translate=(a["translate"], a["translate"]),
                scale=tuple(a["scale"]),
                fill=p["pad_value"],
            ),
            v2.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        ]
    steps += [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
        _repeat_to_3ch,
        v2.Normalize(mean=p["mean"], std=p["std"]),
    ]
    return v2.Compose(steps)
