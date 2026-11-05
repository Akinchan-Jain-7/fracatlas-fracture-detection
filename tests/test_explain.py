"""Phase 6: letterbox geometry, boxes, Grad-CAM, localisation scores, sanity check, pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from PIL import Image
from torch import nn

from fracdet.data.boxes import Box, load_boxes, parse_yolo
from fracdet.data.transforms import (
    Letterbox,
    LetterboxGeometry,
    build_transforms,
    eval_transform_with_geometry,
    letterbox_geometry,
)
from fracdet.eval.protocol import ProtocolError
from fracdet.explain.gradcam import (
    GradCAM,
    explain_image,
    feature_layers,
    resolve_layer,
    to_original,
    upsample,
)
from fracdet.explain.localize import box_mask, content_mask, localization_scores
from fracdet.explain.pipeline import select_gallery
from fracdet.explain.sanity import randomized_copy, rank_correlation


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("w,h", [(373, 454), (454, 373), (2304, 2880), (100, 100), (1, 50)])
def test_letterbox_geometry_matches_the_letterbox(w, h):
    g = letterbox_geometry(w, h, 384)
    img = Image.new("L", (w, h), 255)  # white image on black padding
    arr = np.asarray(Letterbox(384, 0)(img))
    rows, cols = g.content
    assert (arr[rows, cols] == 255).all()
    assert arr.sum() == 255 * g.new_w * g.new_h  # nothing white outside the content
    assert max(g.new_w, g.new_h) == 384


def test_to_letterbox_maps_image_corners_to_content_corners():
    g = letterbox_geometry(373, 454, 384)
    assert g.to_letterbox(0, 0) == (g.left, g.top)
    assert g.to_letterbox(1, 1) == (g.left + g.new_w, g.top + g.new_h)


def test_eval_transform_with_geometry_gives_identical_pixels(tmp_path):
    # Big JPEG, so the reduced-scale draft decode actually kicks in.
    rng = np.random.default_rng(0)
    arr = (rng.random((2880, 2304)) * 255).astype(np.uint8)
    path = tmp_path / "big.jpg"
    Image.fromarray(arr).save(path, quality=90)
    with Image.open(path) as img:
        expected = build_transforms({"image_size": 384}, train=False)(img)
    with Image.open(path) as img:
        x, g = eval_transform_with_geometry({"image_size": 384})(img)
    assert torch.equal(x, expected)
    assert (g.new_w, g.new_h) == (307, 384)


# --------------------------------------------------------------------- boxes
def test_parse_yolo_converts_and_clips():
    boxes = parse_yolo("0 0.5 0.5 0.2 0.4\n\n0 0.95 0.1 0.2 0.4\n")
    assert boxes[0] == Box(0.4, 0.3, 0.6, 0.7)
    assert boxes[1].x1 == 1.0 and boxes[1].y0 == 0.0  # clipped to the image
    assert boxes[0].area == pytest.approx(0.08)
    with pytest.raises(ValueError, match="line 1"):
        parse_yolo("0 0.5 0.5")


def test_load_boxes_from_fake_dataset(fake_root):
    meta = pd.read_csv(fake_root / "dataset.csv")
    frac = meta[meta.fractured == 1].image_id.iloc[0]
    normal = meta[meta.fractured == 0].image_id.iloc[0]
    assert load_boxes(fake_root, frac) == [Box(0.4, 0.4, 0.6, 0.6)]
    assert load_boxes(fake_root, normal) == []
    assert load_boxes(fake_root, "IMG9999999.jpg") == []


# ------------------------------------------------------------------ Grad-CAM
class _Toy(nn.Module):
    """logit = sign * mean brightness of an 8x-pooled map: Grad-CAM must equal that map."""

    def __init__(self, sign: float = 1.0) -> None:
        super().__init__()
        self.stage = nn.AvgPool2d(8)
        self.sign = sign
        self.feature_info = [{"module": "stage"}]

    def forward(self, x):
        return self.sign * self.stage(x[:, :1]).mean(dim=(1, 2, 3))[:, None]


def _bright_square(size=64, y=8, x=40, s=8):
    x_ = torch.zeros(1, 3, size, size)
    x_[..., y : y + s, x : x + s] = 1.0
    return x_


def test_gradcam_points_at_the_evidence():
    res = GradCAM(_Toy(), ["stage"])(_bright_square())
    cam = res.cams["stage"][0]
    assert cam.shape == (8, 8)
    assert torch.argmax(cam).item() == 1 * 8 + 5  # cell (row 1, col 5) = the square
    assert (cam >= 0).all()


def test_gradcam_is_empty_when_nothing_argues_for_fracture():
    res = GradCAM(_Toy(sign=-1.0), ["stage"])(_bright_square())
    assert res.cams["stage"].sum() == 0


def test_gradcam_batch_equals_one_image_at_a_time():
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=1).eval()
    layers = [resolve_layer(model, -1), resolve_layer(model, -2)]
    x = torch.randn(3, 3, 64, 64)
    batch = GradCAM(model, layers)(x)
    for i in range(3):
        single = GradCAM(model, layers)(x[i : i + 1])
        for n in layers:
            torch.testing.assert_close(batch.cams[n][i], single.cams[n][0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(batch.logits[i], single.logits[0], rtol=1e-4, atol=1e-5)
    assert all(p.grad is None for p in model.parameters())  # no weight gradients written


def test_gradcam_leaves_no_hooks_behind():
    model = _Toy()
    GradCAM(model, ["stage"])(_bright_square())
    assert not model.stage._forward_hooks


def test_layer_names_for_the_project_model():
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    assert feature_layers(model)[-2:] == ["stages.2", "stages.3"]
    assert resolve_layer(model, -1) == "stages.3"
    with pytest.raises(KeyError, match="not in model"):
        GradCAM(model, ["nope"])


def test_upsample_and_back_to_original_size():
    g = letterbox_geometry(100, 50, 64)
    cam = upsample(np.ones((4, 4), dtype=np.float32), 64)
    assert cam.shape == (64, 64)
    back = to_original(cam, g, (100, 50))
    assert back.shape == (50, 100)  # numpy: rows = height
    assert np.allclose(back, 1.0)


def test_explain_image_returns_a_map_on_the_original_grid():
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=1).eval()
    img = Image.fromarray((np.random.default_rng(0).random((90, 70)) * 255).astype(np.uint8))
    out = explain_image(model, img, {"image_size": 64})
    assert 0 < out["prob"] < 1
    assert out["heatmap"].shape == (90, 70)
    assert out["heatmap"].min() >= 0 and out["heatmap"].max() <= 1
    assert out["layer"] == "layer4"


# -------------------------------------------------------------- localisation
def _geom():
    return LetterboxGeometry(size=100, new_w=60, new_h=100, left=20, top=0)


def test_box_mask_area_and_padding():
    g = _geom()
    m = box_mask([Box(0.0, 0.0, 0.5, 0.5)], g)
    assert m.sum() == 30 * 50
    assert m[:, :20].sum() == 0  # the box starts at the content's left edge, not the canvas's
    assert content_mask(g).sum() == 60 * 100


def test_localization_hit_miss_and_baselines():
    g = _geom()
    box = [Box(0.0, 0.0, 0.5, 0.5)]  # top-left quarter of the image (letterbox x 20..50)
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[10, 30] = 1.0
    s = localization_scores(cam, box, g)
    assert s["hit"] and s["energy_share"] == 1.0
    assert s["area_share"] == pytest.approx(0.25)
    assert not s["centre_hit"]  # centre (50, 50) is outside the quarter box
    cam[90, 70] = 2.0  # stronger peak, outside the box
    s = localization_scores(cam, box, g)
    assert not s["hit"] and s["energy_share"] == pytest.approx(1 / 3)


def test_peaks_in_the_padding_are_ignored():
    g = _geom()
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[5, 5] = 100.0  # in the left padding
    cam[10, 30] = 1.0  # inside the box
    assert localization_scores(cam, [Box(0.0, 0.0, 0.5, 0.5)], g)["hit"]


def test_empty_map_is_a_miss():
    s = localization_scores(np.zeros((100, 100)), [Box(0.4, 0.4, 0.6, 0.6)], _geom())
    assert s["empty"] and not s["hit"] and s["energy_share"] == 0.0
    assert s["centre_hit"]


# ------------------------------------------------------------------- sanity
def test_randomized_copy_levels():
    torch.manual_seed(1)
    cfg = {"name": "convnext_tiny"}
    trained = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    with torch.no_grad():  # make "trained" weights differ from any fresh init
        for p in trained.parameters():
            p.add_(0.1)
    partial = randomized_copy(trained, cfg, "head_and_last_stage", seed=0)
    full = randomized_copy(trained, cfg, "full", seed=0)
    t, p, f = trained.state_dict(), partial.state_dict(), full.state_dict()
    for k in t:
        if not t[k].is_floating_point():
            continue
        reset = k.startswith(("stages.3.", "head."))
        assert torch.equal(t[k], p[k]) != reset, k
        assert not torch.equal(t[k], f[k]), k
    with pytest.raises(ValueError):
        randomized_copy(trained, cfg, "half", seed=0)


def test_rank_correlation():
    a = np.arange(20.0)
    assert rank_correlation(a, a**3) == pytest.approx(1.0)
    assert rank_correlation(a, -a) == pytest.approx(-1.0)
    assert np.isnan(rank_correlation(a, np.zeros(20)))


# ------------------------------------------------------------------ galleries
def test_select_gallery_follows_the_rule():
    preds = pd.DataFrame(
        {
            "image_id": [f"i{k}" for k in range(8)],
            "label": [1, 1, 1, 0, 0, 0, 0, 0],
            "outcome": ["TP", "FN", "FN", "FP", "FP", "FP", "TN", "TN"],
            "region": ["leg", "leg", "hand", "hand", "hand", "leg", "hand", "leg"],
            "prob": [0.9, 0.05, 0.02, 0.8, 0.3, 0.6, 0.01, 0.02],
        }
    )
    fn = select_gallery(preds, {"select": {"outcome": "FN"}, "take": "all"}, 1, 0)
    assert fn.image_id.tolist() == ["i1", "i2"]
    hand_fp = select_gallery(
        preds, {"select": {"outcome": "FP", "region": "hand"}, "order": "prob_desc"}, 1, 0
    )
    assert hand_fp.image_id.tolist() == ["i3"]  # the most confident one
    other_fp = select_gallery(preds, {"select": {"outcome": "FP", "region_not": "hand"}}, 5, 0)
    assert other_fp.image_id.tolist() == ["i5"]
    a = select_gallery(preds, {"select": {"label": 0}, "order": "random"}, 3, seed=7)
    b = select_gallery(preds, {"select": {"label": 0}, "order": "random"}, 3, seed=7)
    assert a.image_id.tolist() == b.image_id.tolist()  # seeded


# ---------------------------------------------------------------- end to end
def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
        capture_output=True,
    )  # fmt: skip


def test_explain_split_end_to_end(tmp_path, fake_root):
    from fracdet.config import load_config, save_config
    from fracdet.eval.pipeline import tune
    from fracdet.explain.pipeline import explain_split
    from fracdet.logging_utils import get_logger
    from fracdet.train.fit import fit
    from tests.test_training import _fit_cfg

    log = get_logger("t")
    cfg = _fit_cfg(tmp_path, fake_root)
    cfg["preprocess"]["image_size"] = 64
    run_dir = tmp_path / "outputs" / "runs" / "20260101-000000_test"
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    fit(cfg, fake_root, run_dir, log)
    eval_cfg = {"eval": {"target_recall": 0.9, "bootstrap": {"n": 20, "seed": 0, "alpha": 0.05},
                         "group_column": "split_group", "min_positives_for_ci": 1}}  # fmt: skip
    tune(run_dir, eval_cfg, tmp_path / "tune", log)

    repo = tmp_path / "repo"
    (repo / "results" / "phase3_test").mkdir(parents=True)
    (repo / "configs").mkdir()
    _git(repo, "init", "-q")
    shutil.copy(tmp_path / "tune" / "threshold.json", repo / "results" / "phase3_test")
    plan = load_config("configs/explain.yaml")
    plan["run_name"] = "test"
    plan["threshold_file"] = str(repo / "results" / "phase3_test" / "threshold.json")
    plan["gradcam"]["batch_size"] = 4
    plan["sanity"]["n_images"] = 3
    plan_path = repo / "configs" / "explain.yaml"
    save_config(plan, plan_path)

    kwargs = dict(run_dir=run_dir, data_root=fake_root, cfg=plan, log=log,
                  config_path=plan_path, num_workers=0, device="cpu")  # fmt: skip
    # Test split refused while the analysis plan is uncommitted.
    with pytest.raises(ProtocolError, match="not committed"):
        explain_split(split="test", out_dir=tmp_path / "never", **kwargs)

    # Validation runs anyway (dry runs allowed), with the sanity check.
    val = explain_split(split="val", out_dir=tmp_path / "val", **kwargs)
    out = tmp_path / "val"
    for f in ["summary.json", "report.md", "localization_per_image.csv", "cams.npz",
              "figures/val_pointing_game.png"]:  # fmt: skip
        assert (out / f).is_file(), f
    assert val["layers"] == {"primary": "layer4", "extra": ["layer3"]}
    assert val["protocol"]["outcome_counts"] == val["protocol"]["committed_counts"]
    assert set(val["sanity"]["levels"]) == {"head_and_last_stage", "full"}
    per = pd.read_csv(out / "localization_per_image.csv")
    assert len(per) == val["localization"]["all"]["n_images"] > 0
    assert per["area_share"].between(0, 1).all()
    cams = np.load(out / "cams.npz", allow_pickle=True)
    assert cams["layer4"].shape == (10, 2, 2) and cams["layer3"].shape == (10, 4, 4)
    assert json.loads((out / "summary.json").read_text())["phase"] == 6

    # Committed plan -> the test split runs.
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "plan")
    test = explain_split(split="test", out_dir=tmp_path / "test", **kwargs)
    assert test["protocol"]["plan_committed"] and test["sanity"] is None

    # A different model state (wrong committed counts) is caught.
    (repo / "results" / "phase3_test" / "test_metrics.json").write_text(
        json.dumps({"test": {"all": {"metrics": {"tp": 99, "fp": 0, "tn": 0, "fn": 0}}}})
    )
    with pytest.raises(ProtocolError, match="differ from committed"):
        explain_split(split="test", out_dir=tmp_path / "test2", **kwargs)
