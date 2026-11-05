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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where the image sits inside the ``size x size`` letterboxed square.

    Grad-CAM (Phase 6) and the web app (Phase 8) need this to map between the
    model's frame and the original X-ray: fracture boxes go in, heatmaps come out.
    """

    size: int
    new_w: int
    new_h: int
    left: int
    top: int

    @property
    def content(self) -> tuple[slice, slice]:
        """Row and column slices of the image area (padding excluded)."""
        return slice(self.top, self.top + self.new_h), slice(self.left, self.left + self.new_w)

    def to_letterbox(self, x: float, y: float) -> tuple[float, float]:
        """Normalised image coordinates (0..1, as in YOLO files) -> letterbox pixels."""
        return self.left + x * self.new_w, self.top + y * self.new_h


def letterbox_geometry(w: int, h: int, size: int) -> LetterboxGeometry:
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return LetterboxGeometry(size, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2)


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
        g = letterbox_geometry(*img.size, self.size)
        resized = img.resize((g.new_w, g.new_h), Image.Resampling.BILINEAR)
        canvas = Image.new(img.mode, (self.size, self.size), self.fill)
        canvas.paste(resized, (g.left, g.top))
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


def eval_transform_with_geometry(
    preprocess: dict[str, Any] | None = None,
) -> Callable[[Image.Image], tuple[torch.Tensor, LetterboxGeometry]]:
    """The eval pipeline, also returning where the image landed in the square.

    Same steps (and so the same pixels) as ``build_transforms(train=False)``.
    The geometry is taken from the image *after* the draft decode, because the
    reduced-scale JPEG decode can change the size by a rounding pixel.
    """
    p = {**DEFAULT_PREPROCESS, **(preprocess or {})}
    steps = build_transforms(p, train=False).transforms
    decode, rest = steps[0], v2.Compose(steps[1:])
    assert isinstance(decode, DraftGrayscale)

    def apply(img: Image.Image) -> tuple[torch.Tensor, LetterboxGeometry]:
        gray = decode(img)
        return rest(gray), letterbox_geometry(*gray.size, p["image_size"])

    return apply
