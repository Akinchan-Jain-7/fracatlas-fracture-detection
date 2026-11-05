"""Phase 7: detection data, metrics, Grad-CAM boxes, training and the evaluation protocol."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from fracdet.data.boxes import Box
from fracdet.data.transforms import LetterboxGeometry
from fracdet.detect.data import (
    DetectionDataset,
    PositivesPlusNegatives,
    boxes_to_pixels,
    collate,
    hflip,
)
from fracdet.detect.gradcam_boxes import cam_box
from fracdet.detect.metrics import (
    ImageResult,
    average_precision,
    froc_points,
    grouped_bootstrap,
    iou_matrix,
    match,
    split_metrics,
)
from fracdet.detect.model import build_detector, detect
from fracdet.eval.protocol import ProtocolError


# ------------------------------------------------------------------- metrics
def test_iou_matrix_known_values():
    a = np.array([[0, 0, 2, 2], [0, 0, 1, 1]])
    b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]])
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1 / 7)
    assert iou[0, 1] == 1.0 and iou[0, 2] == 0.0
    assert iou[1, 1] == pytest.approx(0.25)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_iou_is_unchanged_by_a_per_axis_resize():
    a, b = np.array([[0.1, 0.2, 0.4, 0.5]]), np.array([[0.2, 0.3, 0.5, 0.7]])
    s = np.array([373, 454, 373, 454])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(iou_matrix(a * s, b * s)[0, 0])


def test_match_counts_a_duplicate_box_as_false_positive():
    gt = np.array([[0, 0, 1, 1]])
    boxes = np.array([[0, 0, 1, 1], [0.05, 0, 1, 1], [2, 2, 3, 3]])
    tp = match(boxes, np.array([0.9, 0.8, 0.95]), gt, 0.5)
    assert tp.tolist() == [True, False, False]


def test_average_precision_examples():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == 1.0
    # TP, FP, TP over 2 boxes: (P, R) = (1, .5), (.5, .5), (.67, 1);
    # the envelope gives 0.5 * 1 + 0.5 * 0.667
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert average_precision(np.array([0.9]), np.array([True]), 4) == pytest.approx(0.25)
    assert np.isnan(average_precision(np.zeros(0), np.zeros(0, bool), 0))
    assert average_precision(np.zeros(0), np.zeros(0, bool), 3) == 0.0


def test_froc_points():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    tp = np.array([True, False, True, False])
    f = froc_points(scores, tp, n_gt=4, n_images=2, fp_rates=[0.5, 1.0])
    assert f == {"sens_at_0.5_fp": 0.5, "sens_at_1_fp": 0.5}
    f = froc_points(scores, tp, n_gt=4, n_images=1, fp_rates=[1.0, 2.0])
    assert f == {"sens_at_1_fp": 0.5, "sens_at_2_fp": 0.5}


def _res(i, label, gt, boxes, scores):
    return ImageResult(f"i{i}", label, np.array(gt, float).reshape(-1, 4),
                       np.array(boxes, float).reshape(-1, 4), np.array(scores, float))  # fmt: skip


def test_boxes_on_normal_images_cost_ap_but_not_the_fractured_only_ap():
    frac = _res(0, 1, [[0.4, 0.4, 0.6, 0.6]], [[0.4, 0.4, 0.6, 0.6]], [0.5])
    clean = _res(1, 0, [], [], [])
    noisy = _res(1, 0, [], [[0.1, 0.1, 0.2, 0.2]], [0.9])
    assert split_metrics([frac, clean])["ap50_all"] == 1.0
    m = split_metrics([frac, noisy])
    assert m["ap50_all"] == pytest.approx(0.5) and m["ap50_fractured"] == 1.0
    assert m["boxes_per_normal_at_0.5"] == 1.0


def test_top1_pointing_and_iou():
    r = _res(0, 1, [[0.4, 0.4, 0.6, 0.6]], [[0.45, 0.45, 0.9, 0.9], [0.4, 0.4, 0.6, 0.6]],
             [0.3, 0.2])  # fmt: skip
    t = r.top1()
    assert not t["top1_pointing"]  # top box centre (0.675, 0.675) is outside
    assert not t["top1_iou50"] and t["top1_score"] == 0.3
    r.scores = np.array([0.1, 0.2])
    assert r.top1()["top1_pointing"] and r.top1()["top1_iou50"]
    assert not _res(0, 1, [[0, 0, 1, 1]], [], []).top1()["top1_pointing"]


def test_grouped_bootstrap_is_seeded_and_brackets_the_value():
    rng = np.random.default_rng(0)
    rs = []
    for i in range(60):
        lab = int(i % 3 == 0)
        gt = [[0.4, 0.4, 0.5, 0.5]] if lab else []
        good = lab and rng.random() < 0.6
        boxes = [[0.4, 0.4, 0.5, 0.5]] if good else [[0.1, 0.1, 0.2, 0.2]]
        rs.append(_res(i, lab, gt, boxes, [rng.random()]))
    groups = np.arange(60) // 2
    a = grouped_bootstrap(rs, groups, ["ap50_all", "top1_pointing"], n_boot=200, seed=1)
    b = grouped_bootstrap(rs, groups, ["ap50_all", "top1_pointing"], n_boot=200, seed=1)
    assert a == b
    for v in a.values():
        assert v["low"] <= v["value"] <= v["high"]


# ---------------------------------------------------------------------- data
def test_boxes_to_pixels_and_hflip():
    px = boxes_to_pixels([Box(0.1, 0.2, 0.3, 0.6)], width=100, height=50)
    torch.testing.assert_close(px, torch.tensor([[10.0, 10.0, 30.0, 30.0]]))
    img = torch.zeros(3, 50, 100)
    img[:, 10:30, 10:30] = 1
    flipped, fb = hflip(img, px)
    torch.testing.assert_close(fb, torch.tensor([[70.0, 10.0, 90.0, 30.0]]))
    assert flipped[0, 20, 80] == 1 and flipped[0, 20, 20] == 0
    assert boxes_to_pixels([], 10, 10).shape == (0, 4)


def test_detection_dataset_targets(fake_root):
    from fracdet.data.metadata import load_metadata

    meta = load_metadata(fake_root)
    ds = DetectionDataset(meta, str(fake_root), min_side=64)
    i_frac = int(np.flatnonzero(meta.label == 1)[0])
    i_norm = int(np.flatnonzero(meta.label == 0)[0])
    x, t, _ = ds[i_frac]
    assert x.shape == (3, 128, 96) and x.min() >= 0 and x.max() <= 1
    torch.testing.assert_close(t["boxes"], torch.tensor([[38.4, 51.2, 57.6, 76.8]]))
    assert t["labels"].tolist() == [1]
    _, t0, _ = ds[i_norm]
    assert t0["boxes"].shape == (0, 4) and t0["labels"].shape == (0,)
    images, targets, idx = collate([ds[i_frac], ds[i_norm]])
    assert len(images) == len(targets) == 2 and idx == [i_frac, i_norm]
    aug = DetectionDataset(meta, str(fake_root), 64, {"hflip_p": 1.0}, train=True)
    torch.testing.assert_close(aug[i_frac][1]["boxes"], torch.tensor([[38.4, 51.2, 57.6, 76.8]]))


def test_sampler_keeps_all_positives_and_redraws_negatives():
    labels = [1] * 5 + [0] * 50
    s = PositivesPlusNegatives(labels, ratio=2.0, seed=0)
    s.set_epoch(1)
    e1 = list(s)
    s.set_epoch(2)
    e2 = list(s)
    assert len(e1) == len(s) == 15
    assert set(range(5)) <= set(e1) and set(range(5)) <= set(e2)
    assert set(e1) != set(e2)  # different normals each epoch
    s.set_epoch(1)
    assert list(s) == e1  # reproducible


# ----------------------------------------------------------- Grad-CAM boxes
def test_cam_box_wraps_the_peak_region_and_ignores_padding():
    g = LetterboxGeometry(size=100, new_w=50, new_h=100, left=25, top=0)
    cam = np.zeros((100, 100))
    cam[40:60, 30:40] = 1.0  # region around the peak (letterbox x 30..39)
    cam[45, 35] = 2.0
    cam[10:20, 60:70] = 1.5  # a separate, weaker blob: not part of the box
    cam[0:5, 0:5] = 9.0  # in the padding: ignored
    box = cam_box(cam, g, relative_threshold=0.5)
    np.testing.assert_allclose(box, [5 / 50, 40 / 100, 15 / 50, 60 / 100])
    assert cam_box(np.zeros((100, 100)), g) is None


# --------------------------------------------------------------------- model
def test_detector_has_two_classes_and_outputs_boxes():
    torch.manual_seed(0)
    m = build_detector({"name": "fasterrcnn_resnet50_fpn_v2", "pretrained": False,
                        "min_size": 64, "max_size": 96, "score_thresh": 0.0})  # fmt: skip
    assert m.roi_heads.box_predictor.cls_score.out_features == 2
    out = detect(m, [torch.rand(3, 80, 60)])
    assert set(out[0]) == {"boxes", "scores"} and out[0]["boxes"].shape[1] == 4
    if len(out[0]["boxes"]):
        assert out[0]["boxes"][:, 2].max() <= 60 and out[0]["boxes"][:, 3].max() <= 80
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector({"name": "yolo"})


# ---------------------------------------------------------------- end to end
def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True)  # fmt: skip


def test_train_and_evaluate_detector_end_to_end(tmp_path, fake_root):
    from fracdet.config import load_config, save_config
    from fracdet.detect.evaluate import evaluate_detector
    from fracdet.detect.train import fit_detector
    from fracdet.eval.pipeline import tune
    from fracdet.logging_utils import get_logger
    from fracdet.train.fit import fit
    from tests.test_training import _fit_cfg

    log = get_logger("t")
    # A tiny classifier with a tuned threshold (the end-to-end and Grad-CAM baseline need it).
    ccfg = _fit_cfg(tmp_path, fake_root)
    ccfg["preprocess"]["image_size"] = 64
    cdir = tmp_path / "outputs" / "runs" / "20260101-000000_clf"
    cdir.mkdir(parents=True)
    save_config(ccfg, cdir / "config.yaml")
    fit(ccfg, fake_root, cdir, log)
    eval_cfg = {"eval": {"target_recall": 0.9, "bootstrap": {"n": 10, "seed": 0, "alpha": 0.05},
                         "group_column": "split_group", "min_positives_for_ci": 1}}  # fmt: skip
    tune(cdir, eval_cfg, tmp_path / "tune", log)

    repo = tmp_path / "repo"
    (repo / "thr").mkdir(parents=True)
    _git(repo, "init", "-q")
    shutil.copy(tmp_path / "tune" / "threshold.json", repo / "thr")

    cfg = load_config("configs/detect.yaml", [
        f"paths.splits_dir={tmp_path / 'splits'}", "split.name=s", "detector.pretrained=false",
        "detector.min_size=96", "detector.max_size=128", "train.epochs=1",
        "train.num_workers=0", "train.batch_size=2", "seed=42",
    ])  # fmt: skip
    cfg["evaluate"]["bootstrap"]["n"] = 20
    cfg["evaluate"]["classifier_threshold_file"] = str(repo / "thr" / "threshold.json")
    plan = repo / "detect.yaml"
    save_config(cfg, plan)
    runs = {}
    for seed in (42, 43):
        cfg["seed"] = seed
        d = tmp_path / "outputs" / "runs" / f"20260101-000000_det_s{seed}"
        d.mkdir(parents=True)
        save_config(cfg, d / "config.yaml")
        fit_detector(cfg, fake_root, d, log)
        runs[seed] = d
    cfg["seed"] = 42
    pd.read_csv(runs[42] / "val_detections.csv")  # written

    kw = dict(data_root=fake_root, cfg=cfg, log=log, config_path=plan,
              results_root=repo / "results", classifier_dir=cdir, num_workers=0,
              device="cpu")  # fmt: skip
    with pytest.raises(ProtocolError, match="not committed"):
        evaluate_detector(runs[42], split="test", out_dir=tmp_path / "never", **kw)
    val = evaluate_detector(runs[43], split="val", out_dir=tmp_path / "val", **kw)  # dry run ok
    assert val["detector"]["seed"] == 43
    assert set(val["detector_metrics"]["ci95"]) >= {"ap50_all", "top1_pointing"}
    assert val["protocol"]["classifier_counts"] == val["protocol"]["classifier_committed_counts"]
    per = pd.read_csv(tmp_path / "val" / "val_per_image.csv")
    assert len(per) == val["detector_metrics"]["point"]["n_images"] == 10

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "plan")
    with pytest.raises(ProtocolError, match="pre-registered"):
        evaluate_detector(runs[43], split="test", out_dir=tmp_path / "never", **kw)
    test = evaluate_detector(runs[42], split="test", out_dir=tmp_path / "test", **kw)
    assert test["protocol"]["plan_committed"]
    assert (tmp_path / "test" / "report.md").is_file()
    assert test["figures"]  # galleries are drawn for test
    # Once results are committed, the test split is refused without a reason.
    done = repo / "results" / "phase7_detect" / "test"
    done.mkdir(parents=True)
    shutil.copy(tmp_path / "test" / "summary.json", done)
    with pytest.raises(ProtocolError, match="scored once"):
        evaluate_detector(runs[42], split="test", out_dir=tmp_path / "again", **kw)
