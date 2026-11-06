"""The app's brain: one X-ray in, one structured answer out. No web code here.

Keeping this separate from the web server means it can be tested without a
server, used from a notebook, and wrapped by any web framework.

For each image:

1. **Classifier** (Phase 5, ConvNeXt-Tiny): fracture probability. Above the
   pre-registered threshold (0.0968) the X-ray is *flagged*.
2. **Grad-CAM** (Phase 6) on the same forward pass: where the classifier found
   evidence for fracture.
3. **Detector** (Phase 7, Faster R-CNN): boxes around likely fractures. Shown
   only on flagged X-rays: on its own it puts a confident box on about 1 normal
   X-ray in 5 (Phase 7), so the classifier decides *whether*, the detector *where*.

Each model gets its own fresh decode of the uploaded bytes, through exactly the
same steps as in training and evaluation (``DraftGrayscale`` reduces JPEGs while
decoding, so a decoded image cannot be reused for the other model).
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from fracdet.data.transforms import DraftGrayscale
from fracdet.detect.model import detect
from fracdet.detect.train import load_detector
from fracdet.explain.gradcam import explain_image
from fracdet.train.checkpoint import load_checkpoint

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "serve.yaml"


class ModelFileError(RuntimeError):
    """A model file is missing, corrupted, or not the evaluated model."""


class BadImageError(ValueError):
    """The upload is not an image the app can read."""


def load_serve_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _check_file(path: Path, expected_sha: str | None) -> None:
    if not path.is_file():
        raise ModelFileError(f"model file not found: {path}")
    if expected_sha and sha256_of(path) != expected_sha:
        raise ModelFileError(f"{path.name}: SHA-256 differs from configs/serve.yaml")


def _check_provenance(name: str, actual: dict[str, Any], expect: dict[str, Any]) -> None:
    wrong = {k: (actual.get(k), v) for k, v in expect.items() if actual.get(k) != v}
    if wrong:
        detail = "; ".join(f"{k}: file has {a!r}, expected {e!r}" for k, (a, e) in wrong.items())
        raise ModelFileError(f"{name} is not the evaluated model ({detail})")


def heatmap_png(heat: np.ndarray, max_side: int, max_alpha: float = 0.55) -> str:
    """0..1 heatmap -> base64 RGBA PNG; transparent where there is no evidence.

    Opacity grows with the square root of the value (as in the Phase 6 figures),
    so weak evidence stays faint and the X-ray shows through.
    """
    h, w = heat.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        small = Image.fromarray(heat.astype(np.float32), mode="F").resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR
        )
        heat = np.asarray(small, dtype=np.float32)
    v = np.clip(heat, 0.0, 1.0)
    rgba = matplotlib.colormaps["turbo"](v)
    rgba[..., 3] = max_alpha * np.sqrt(v)
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).round().astype(np.uint8), mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def open_image(data: bytes) -> Image.Image:
    """Decode an upload (JPEG or PNG), applying any EXIF rotation as a browser would."""
    try:
        with Image.open(io.BytesIO(data)) as probe:  # full decode once, to reject broken files
            fmt = probe.format
            probe.load()
        img = Image.open(io.BytesIO(data))  # fresh and lazy: JPEG draft decoding still works
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise BadImageError("The file is not a readable JPEG or PNG image.") from e
    if fmt not in ("JPEG", "PNG"):
        raise BadImageError(f"{fmt} images are not supported; use JPEG or PNG.")
    if img.getexif().get(0x0112, 1) != 1:  # rotated phone photo; X-ray files have no rotation tag
        img = ImageOps.exif_transpose(img)
    return img


@dataclass
class LoadedModels:
    classifier: torch.nn.Module
    classifier_ckpt: dict[str, Any]
    detector: torch.nn.Module
    detector_ckpt: dict[str, Any]


class FracturePipeline:
    """Load both models once, then call :meth:`predict` for every upload."""

    def __init__(self, models: LoadedModels, cfg: dict[str, Any], device: str = "cpu") -> None:
        self.models = models
        self.cfg = cfg
        self.device = torch.device(device)
        self.threshold = float(cfg["classifier"]["threshold"])
        # One prediction at a time: bounds memory, and torch already uses all cores.
        self._lock = threading.Lock()

    @classmethod
    def from_dir(
        cls, models_dir: str | Path, cfg: dict[str, Any] | None = None, device: str = "cpu"
    ) -> FracturePipeline:
        cfg = cfg or load_serve_config()
        models_dir = Path(models_dir)
        c, d = cfg["classifier"], cfg["detector"]
        c_path, d_path = models_dir / c["file"], models_dir / d["file"]
        _check_file(c_path, c.get("sha256"))
        _check_file(d_path, d.get("sha256"))

        classifier, c_ckpt = load_checkpoint(c_path, device=device)
        _check_provenance(
            "classifier",
            {"epoch": c_ckpt["epoch"],
             "trained_commit": c_ckpt["provenance"].get("git", {}).get("commit"),
             "split_csv_md5": c_ckpt["provenance"].get("split_csv_md5")},
            c["expect"],
        )  # fmt: skip
        detector, d_ckpt = load_detector(d_path, device=device)
        _check_provenance(
            "detector",
            {"epoch": d_ckpt["epoch"], "seed": d_ckpt["provenance"].get("seed"),
             "trained_commit": d_ckpt["provenance"].get("git", {}).get("commit"),
             "split_csv_md5": d_ckpt["provenance"].get("split_csv_md5")},
            d["expect"],
        )  # fmt: skip
        return cls(LoadedModels(classifier, c_ckpt, detector, d_ckpt), cfg, device)

    # ------------------------------------------------------------------ predict
    def predict(self, data: bytes) -> dict[str, Any]:
        """Everything the page needs for one uploaded X-ray (all coordinates 0..1)."""
        img_cls = open_image(data)
        width, height = img_cls.size
        t0 = time.perf_counter()
        with self._lock:
            c = self.cfg["classifier"]
            exp = explain_image(
                self.models.classifier, img_cls, self.models.classifier_ckpt.get("preprocess", {}),
                layer=c["gradcam_layer"], device=self.device,
            )  # fmt: skip
            t1 = time.perf_counter()
            boxes = self._boxes(open_image(data))
            t2 = time.perf_counter()

        prob = float(exp["prob"])
        flagged = prob >= self.threshold
        d = self.cfg["detector"]
        cutoff = float(d["extra_box_min_score"])
        shown = []
        if flagged and boxes:
            shown = [boxes[0]] + [b for b in boxes[1:] if b["score"] >= cutoff]
            shown = shown[: d["max_boxes"]]
            for b in shown:
                b["confident"] = b["score"] >= cutoff
        return {
            "probability": prob,
            "threshold": self.threshold,
            "flagged": flagged,
            "image": {"width": width, "height": height},
            "heatmap_png": heatmap_png(exp["heatmap"], self.cfg["limits"]["heatmap_max_side"]),
            "heatmap_layer": exp["layer"],
            "boxes": shown,
            "boxes_hidden_reason": None if flagged else "not_flagged",
            "box_cutoff": cutoff,
            "timing_ms": {
                "classifier_and_heatmap": round(1000 * (t1 - t0)),
                "detector": round(1000 * (t2 - t1)),
            },  # fmt: skip
        }

    @torch.inference_mode()
    def _boxes(self, img: Image.Image) -> list[dict[str, Any]]:
        """Detector boxes, highest score first, normalised to the image (as in Phase 7)."""
        min_side = self.models.detector_ckpt["detector_cfg"].get("min_size", 800)
        gray = DraftGrayscale(min_side)(img)
        w, h = gray.size
        x = torch.from_numpy(np.asarray(gray, dtype=np.uint8).copy()).float().div(255)
        x = x[None].expand(3, -1, -1).contiguous().to(self.device)
        out = detect(self.models.detector, [x])[0]
        scale = np.array([w, h, w, h], dtype=float)
        result = []
        for (x0, y0, x1, y1), s in zip(out["boxes"].numpy() / scale, out["scores"].numpy(),
                                       strict=True):  # fmt: skip
            result.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                           "score": float(s)})  # fmt: skip
        return sorted(result, key=lambda b: -b["score"])

    def info(self) -> dict[str, Any]:
        """Static facts for the page: threshold, cut-offs, model card, versions."""
        c, d = self.models.classifier_ckpt, self.models.detector_ckpt
        return {
            "threshold": self.threshold,
            "box_cutoff": float(self.cfg["detector"]["extra_box_min_score"]),
            "classifier": {"name": c["model_cfg"]["name"], "epoch": c["epoch"],
                           "image_size": c.get("preprocess", {}).get("image_size")},
            "detector": {"name": d["detector_cfg"]["name"], "epoch": d["epoch"],
                         "seed": d["provenance"].get("seed")},
            "model_card": self.cfg["model_card"],
            "limits": {"max_upload_mb": self.cfg["limits"]["max_upload_mb"],
                       "accepted_types": self.cfg["limits"]["accepted_types"]},
        }  # fmt: skip
