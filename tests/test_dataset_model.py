import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fracdet.data.dataset import FracAtlasDataset, load_split_frames
from fracdet.data.transforms import IMAGENET_MEAN, IMAGENET_STD, Letterbox, build_transforms
from fracdet.models.classifier import build_model, count_parameters, predict_proba


def test_letterbox_keeps_aspect_and_centres():
    img = Image.new("L", (100, 200), 255)  # tall white image
    out = Letterbox(64, fill=0)(img)
    assert out.size == (64, 64)
    a = np.asarray(out)
    # 200 -> 64 tall, 100 -> 32 wide: white band 32 px wide in the middle
    assert a[:, 16:48].min() == 255
    assert a[:, :15].max() == 0 and a[:, 49:].max() == 0


def test_eval_transform_shape_and_normalisation():
    t = build_transforms({"image_size": 64}, train=False)
    black = t(Image.new("RGB", (80, 50), (0, 0, 0)))
    assert black.shape == (3, 64, 64) and black.dtype == torch.float32
    expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)])
    assert torch.allclose(black[:, 0, 0], expected, atol=1e-5)


def test_rgb_and_grayscale_files_give_identical_tensors():
    """The colour-mode shortcut is removed: same pixels, different file mode."""
    rng = np.random.default_rng(0)
    gray = Image.fromarray(rng.integers(0, 256, (50, 40), dtype=np.uint8), mode="L")
    t = build_transforms({"image_size": 32}, train=False)
    assert torch.equal(t(gray), t(gray.convert("RGB")))


def test_eval_is_deterministic_train_is_random():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (60, 60), dtype=np.uint8))
    ev = build_transforms({"image_size": 48}, train=False)
    tr = build_transforms({"image_size": 48}, train=True)
    assert torch.equal(ev(img), ev(img))
    torch.manual_seed(0)
    outs = [tr(img) for _ in range(5)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def _fake_split(fake_root, tmp_path):
    meta = pd.read_csv(fake_root / "dataset.csv")
    splits = np.array(["train", "val", "test", "excluded"])[np.arange(len(meta)) % 4]
    path = tmp_path / "split.csv"
    pd.DataFrame({"image_id": meta.image_id, "split": splits}).to_csv(path, index=False)
    return path


def test_load_split_frames_only_returns_requested(fake_root, tmp_path):
    split_csv = _fake_split(fake_root, tmp_path)
    frames = load_split_frames(fake_root, split_csv, ["train", "val"])
    assert set(frames) == {"train", "val"}
    assert len(frames["train"]) == 10 and set(frames["train"].split) == {"train"}
    with pytest.raises(ValueError):
        load_split_frames(fake_root, split_csv, ["excluded"])


def test_dataset_item(fake_root, tmp_path):
    frames = load_split_frames(fake_root, _fake_split(fake_root, tmp_path), ["train"])
    ds = FracAtlasDataset(frames["train"], build_transforms({"image_size": 32}))
    x, y, i = ds[3]
    assert x.shape == (3, 32, 32) and y.dtype == torch.float32 and i == 3
    assert float(y) == frames["train"].label.iloc[3]


def test_model_outputs_one_logit():
    model = build_model({"name": "resnet18", "pretrained": False})
    probs = predict_proba(model, torch.randn(2, 3, 64, 64))
    assert probs.shape == (2,) and ((probs >= 0) & (probs <= 1)).all()
    assert count_parameters(model)["trainable"] > 1_000_000
