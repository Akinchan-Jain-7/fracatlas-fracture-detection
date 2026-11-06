"""Phase 8: serve config vs committed results, the inference pipeline, and the web API."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from PIL import Image

from fracdet.detect.model import build_detector
from fracdet.detect.train import save_detector
from fracdet.serve.pipeline import (
    BadImageError,
    FracturePipeline,
    ModelFileError,
    heatmap_png,
    load_serve_config,
    open_image,
    sha256_of,
)
from fracdet.train.checkpoint import save_checkpoint

CFG = load_serve_config()


def _json(path: str):
    return json.loads(Path(path).read_text())


# ------------------------------------------------- config matches the results
def test_classifier_settings_match_the_phase5_records():
    rec = json.loads(Path("results/phase3_p5_convnext_tiny_s44/threshold.json").read_text())
    c = CFG["classifier"]
    assert c["threshold"] == rec["threshold"]
    assert c["expect"] == {
        "epoch": rec["model"]["best_epoch"],
        "trained_commit": rec["model"]["trained_commit"],
        "split_csv_md5": rec["model"]["split_csv_md5"],
    }


def test_detector_settings_match_the_phase7_records():
    s = json.loads(Path("results/phase7_detect/test/summary.json").read_text())["detector"]
    d = CFG["detector"]
    assert d["expect"]["epoch"] == s["epoch"] and d["expect"]["seed"] == s["seed"]
    assert d["expect"]["trained_commit"] == s["trained_commit"]
    readme = Path("results/phase7_detect/README.md").read_text()
    assert d["sha256"] in readme


def test_box_cutoff_is_the_validation_rule():
    from scripts.choose_box_cutoff import box_cutoff

    val = pd.read_csv("results/phase7_detect/val_s42/val_detections.csv")
    d = CFG["detector"]
    assert d["extra_box_min_score"] == pytest.approx(
        box_cutoff(val, d["extra_box_budget"]), abs=5e-5
    )


def test_model_card_matches_the_test_results():
    mc = CFG["model_card"]
    m = json.loads(Path("results/phase3_p5_convnext_tiny_s44/test_metrics.json").read_text())[
        "test"
    ]["all"]
    for k, v in mc["classifier"].items():
        assert v == pytest.approx(m["metrics"][k], abs=5e-5), k
    assert (mc["test_images"], mc["test_fractured"]) == (m["n"], m["n_fractured"])
    det = json.loads(Path("results/phase7_detect/test/summary.json").read_text())[
        "detector_metrics"
    ]["point"]
    for k in ("top1_pointing", "ap50_all", "sens_at_1_fp"):
        assert mc["detector"][k] == pytest.approx(det[k], abs=5e-5), k
    per = pd.read_csv("results/phase7_detect/test/test_per_image.csv")
    hw = per[(per.label == 1) & (per.hardware == 1)]
    assert mc["detector"]["hardware_top1_pointing"] == [int(hw.det_top1_pointing.sum()), len(hw)]


def test_box_cutoff_rule_on_a_toy_example():
    from scripts.choose_box_cutoff import box_cutoff

    d = pd.DataFrame({
        "image_id": ["a", "a", "b", "c", "d"], "label": [0, 0, 0, 0, 1],
        "rank": [0, 1, 0, -1, 0], "score": [0.9, 0.4, 0.7, np.nan, 0.99],
    })  # fmt: skip
    c = box_cutoff(d, budget=0.25)  # 3 normal images -> at most 0 boxes (floor(0.75))
    assert 0.9 < c < 0.9001
    # 2 false boxes allowed: the cut-off sits just above the third-highest normal score (0.4),
    # so the 0.9 and 0.7 boxes are kept.
    assert 0.4 < box_cutoff(d, budget=0.67) < 0.4001


# ------------------------------------------------------- tiny model files
@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    """A tiny classifier + detector saved in the real formats, and a matching config."""
    d = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    ccfg = {"model": {"name": "resnet18", "pretrained": False, "num_classes": 1},
            "preprocess": {"image_size": 64}}  # fmt: skip
    clf = timm.create_model("resnet18", pretrained=False, num_classes=1)
    save_checkpoint(d / "clf.pt", clf, ccfg, epoch=3, metrics={},
                    provenance={"git": {"commit": "c0ffee"}, "split_csv_md5": "md5"})  # fmt: skip
    dcfg = {"name": "fasterrcnn_resnet50_fpn_v2", "pretrained": False, "min_size": 64,
            "max_size": 96, "score_thresh": 0.0, "detections_per_img": 10}  # fmt: skip
    det = build_detector(dcfg)
    save_detector(d / "det.pt", det, {"detector": dcfg}, epoch=2, metrics={},
                  provenance={"git": {"commit": "beef"}, "split_csv_md5": "md5",
                              "seed": 7})  # fmt: skip
    cfg = copy.deepcopy(CFG)
    cfg["classifier"].update(
        file="clf.pt",
        sha256=None,
        expect={"epoch": 3, "trained_commit": "c0ffee", "split_csv_md5": "md5"},
    )
    cfg["detector"].update(file="det.pt", sha256=sha256_of(d / "det.pt"),
                           expect={"epoch": 2, "seed": 7, "trained_commit": "beef",
                                   "split_csv_md5": "md5"})  # fmt: skip
    return d, cfg


def _jpeg(w=90, h=120, seed=0) -> bytes:
    arr = (np.random.default_rng(seed).random((h, w)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------- pipeline
def test_pipeline_checks_files_and_provenance(tiny_models):
    d, cfg = tiny_models
    FracturePipeline.from_dir(d, cfg)  # loads
    bad = copy.deepcopy(cfg)
    bad["classifier"]["expect"]["epoch"] = 99
    with pytest.raises(ModelFileError, match="classifier is not the evaluated model.*epoch"):
        FracturePipeline.from_dir(d, bad)
    bad = copy.deepcopy(cfg)
    bad["detector"]["sha256"] = "0" * 64
    with pytest.raises(ModelFileError, match="SHA-256"):
        FracturePipeline.from_dir(d, bad)
    with pytest.raises(ModelFileError, match="not found"):
        FracturePipeline.from_dir(d / "nowhere", cfg)


def test_predict_flagged_and_not_flagged(tiny_models):
    d, cfg = tiny_models
    p = FracturePipeline.from_dir(d, cfg)
    p.threshold = 0.0  # everything flagged
    r = p.predict(_jpeg())
    assert r["flagged"] and r["image"] == {"width": 90, "height": 120}
    assert 0 <= r["probability"] <= 1
    assert 1 <= len(r["boxes"]) <= cfg["detector"]["max_boxes"]
    scores = [b["score"] for b in r["boxes"]]
    assert scores == sorted(scores, reverse=True)
    assert all(b["score"] >= r["box_cutoff"] for b in r["boxes"][1:])
    for b in r["boxes"]:
        assert 0 <= b["x0"] <= b["x1"] <= 1 and 0 <= b["y0"] <= b["y1"] <= 1
    heat = Image.open(io.BytesIO(base64.b64decode(r["heatmap_png"])))
    assert heat.mode == "RGBA" and heat.size == (90, 120)
    p.threshold = 1.01  # nothing flagged
    r = p.predict(_jpeg())
    assert not r["flagged"] and r["boxes"] == [] and r["boxes_hidden_reason"] == "not_flagged"


def test_predict_matches_the_evaluation_code(tiny_models, tmp_path):
    """Same probability as eval.predict.score_split would give (same decode + transform)."""
    from fracdet.data.transforms import build_transforms

    d, cfg = tiny_models
    p = FracturePipeline.from_dir(d, cfg)
    data = _jpeg(300, 400, seed=3)
    with Image.open(io.BytesIO(data)) as img:
        x = build_transforms({"image_size": 64}, train=False)(img)
    with torch.no_grad():
        expected = torch.sigmoid(p.models.classifier(x[None])).item()
    assert p.predict(data)["probability"] == pytest.approx(expected, abs=1e-5)


def test_open_image_rejects_and_rotates():
    with pytest.raises(BadImageError, match="not a readable"):
        open_image(b"not an image")
    buf = io.BytesIO()
    Image.new("L", (10, 10)).save(buf, format="GIF")
    with pytest.raises(BadImageError, match="GIF"):
        open_image(buf.getvalue())
    # A phone photo saved sideways with an EXIF "rotate 90" tag comes out upright.
    img = Image.new("L", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    assert open_image(buf.getvalue()).size == (20, 40)
    assert open_image(_jpeg(30, 50)).size == (30, 50)


def test_heatmap_png_is_transparent_where_there_is_no_evidence():
    heat = np.zeros((40, 20), dtype=np.float32)
    heat[10, 5] = 1.0
    img = Image.open(io.BytesIO(base64.b64decode(heatmap_png(heat, max_side=100))))
    a = np.asarray(img)
    assert a.shape == (40, 20, 4) and a[0, 0, 3] == 0 and a[10, 5, 3] > 100
    small = Image.open(io.BytesIO(base64.b64decode(heatmap_png(np.ones((400, 200)), 100))))
    assert small.size == (50, 100)


# -------------------------------------------------------------------- API
@pytest.fixture
def client(tiny_models, tmp_path, monkeypatch):
    import yaml
    from fastapi.testclient import TestClient

    d, cfg = tiny_models
    cfg_path = tmp_path / "serve.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("FRACDET_MODELS_DIR", str(d))
    monkeypatch.setenv("FRACDET_SERVE_CONFIG", str(cfg_path))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_api_happy_path(client):
    assert client.get("/api/health").json()["status"] == "ok"
    info = client.get("/api/info").json()
    assert info["threshold"] == CFG["classifier"]["threshold"]
    assert info["model_card"]["test_images"] == 589
    r = client.post("/api/predict", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"probability", "flagged", "heatmap_png", "boxes", "threshold"}
    page = client.get("/")
    assert page.status_code == 200 and "Fracture Check" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/fonts/barlow-condensed-latin-700-normal.woff2").status_code == 200


def test_api_rejects_bad_uploads(client):
    r = client.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/predict", files={"file": ("x.jpg", b"not a jpeg", "image/jpeg")})
    assert r.status_code == 400 and "readable" in r.json()["detail"]
    r = client.post("/api/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    big = b"\xff" * (CFG["limits"]["max_upload_mb"] * 1024 * 1024 + 1)
    r = client.post("/api/predict", files={"file": ("x.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_api_reports_missing_models(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("FRACDET_MODELS_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("FRACDET_SERVE_CONFIG", raising=False)
    from app.main import app

    with TestClient(app) as c:
        h = c.get("/api/health")
        assert h.status_code == 503 and "not found" in h.json()["error"]
        r = c.post("/api/predict", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 503
        assert c.get("/").status_code == 200  # the page still loads and explains
