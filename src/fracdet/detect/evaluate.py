"""Phase 7 evaluation of one detector run on one split, against the Grad-CAM baseline.

Order of work:

1. Checks. For the test split: the plan (configs/detect.yaml) is committed and
   unchanged, the run is the pre-registered seed, and no test result for it is
   committed yet (unless a re-run reason is given).
2. Detector boxes for every image of the split.
3. Classifier probabilities with the Phase 5 code (the TP/FP/TN/FN counts must
   match the committed ones) and Grad-CAM boxes for the baseline.
4. Metrics with patient-grouped CIs, per-image table, galleries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fracdet.config import load_config
from fracdet.data.dataset import load_split_frames
from fracdet.data.download import md5_of_file
from fracdet.detect.gradcam_boxes import cam_box
from fracdet.detect.metrics import ImageResult, grouped_bootstrap, split_metrics
from fracdet.detect.train import detections_frame, load_detector, predict_frame
from fracdet.eval.pipeline import _clean, _split_csv, _with_groups, _write_json, find_run
from fracdet.eval.predict import score_split
from fracdet.eval.protocol import ProtocolError, check_same_model, committed_and_unchanged
from fracdet.explain.gradcam import resolve_layer, upsample
from fracdet.explain.pipeline import _committed_counts, _geom, compute_cams
from fracdet.train.checkpoint import git_info, load_checkpoint

CI_KEYS = ["ap50_all", "ap50_fractured", "ap25_all", "sens_at_1_fp", "top1_pointing", "top1_iou50"]
SEARCH = ["outputs/runs", "/kaggle/input"]


def results_dir(root: str | Path = "results") -> Path:
    return Path(root) / "phase7_detect"


def _metrics_block(results, groups, ev) -> dict[str, Any]:
    point = split_metrics(results, ev["iou_threshold"], ev["froc_fp_per_image"])
    b = ev["bootstrap"]
    ci = grouped_bootstrap(results, groups, CI_KEYS, n_boot=b["n"], seed=b["seed"],
                           alpha=b["alpha"], iou_thr=ev["iou_threshold"],
                           fp_rates=ev["froc_fp_per_image"])  # fmt: skip
    return {"point": point, "ci95": ci}


def gradcam_results(
    classifier_dir: Path, data_root, split_csv, split, frame, cfg, device, num_workers, log
) -> tuple[list[ImageResult], pd.DataFrame]:
    """Classifier probabilities (Phase 5 code) + one Grad-CAM box per image."""
    preds, ckpt = score_split(classifier_dir / "best.pt", data_root, split_csv, split,
                              device=device, num_workers=num_workers)  # fmt: skip
    if preds["image_id"].tolist() != frame["image_id"].tolist():
        raise RuntimeError("classifier and detector see the split in different orders")
    model, _ = load_checkpoint(classifier_dir / "best.pt", device=device)
    layer = resolve_layer(model, cfg["baselines"]["gradcam_box"]["layer"])
    cams, _, geoms = compute_cams(model, frame["path"].tolist(), ckpt.get("preprocess", {}),
                                  [layer], device, num_workers=num_workers)  # fmt: skip
    size = int(geoms[0][0])
    rel = cfg["baselines"]["gradcam_box"]["relative_threshold"]
    out = []
    for i, r in enumerate(frame.itertuples(index=False)):
        box = cam_box(upsample(cams[layer][i], size), _geom(geoms[i]), rel)
        boxes = np.zeros((0, 4)) if box is None else box[None]
        scores = np.zeros(0) if box is None else np.array([preds["prob"].iloc[i]])
        out.append(ImageResult(r.image_id, int(r.label), np.zeros((0, 4)), boxes, scores))
    log.info("Grad-CAM boxes at %s for %d images", layer, len(out))
    return out, preds


def evaluate_detector(
    run_dir: str | Path,
    data_root: str | Path,
    cfg: dict[str, Any],
    split: str,
    out_dir: str | Path,
    log: logging.Logger,
    config_path: str | Path = "configs/detect.yaml",
    results_root: str | Path = "results",
    classifier_dir: str | Path | None = None,
    rerun_reason: str | None = None,
    require_preregistration: bool = True,
    num_workers: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ev = cfg["evaluate"]
    t0 = time.time()

    # ---- 1. checks (before any image of the split is read)
    plan_ok, plan_commit = committed_and_unchanged(config_path)
    det, dckpt = load_detector(run_dir / "best.pt", device=dev)
    seed = dckpt["provenance"]["seed"]
    if split == "test":
        if require_preregistration and not plan_ok:
            raise ProtocolError(f"Phase 7 plan not committed: {plan_commit}")
        if seed != cfg["reported_seed"]:
            raise ProtocolError(f"seed {seed} is not the pre-registered test model "
                                f"(reported_seed {cfg['reported_seed']})")  # fmt: skip
        done = results_dir(results_root) / "test" / "summary.json"
        if done.is_file() and not rerun_reason:
            raise ProtocolError(f"{done} already committed: the test set is scored once")
    elif not plan_ok:
        log.warning("plan not committed (%s): fine for a dry run only", plan_commit)
    run_cfg = load_config(run_dir / "config.yaml")
    split_csv = _split_csv(run_cfg)
    if md5_of_file(split_csv) != dckpt["provenance"]["split_csv_md5"]:
        raise ProtocolError("split file differs from the one the detector was trained with")

    frame = load_split_frames(data_root, split_csv, [split])[split]
    frame = _with_groups(frame, split_csv)
    if "split_group" not in frame:
        frame["split_group"] = frame["image_id"]

    # ---- 2. detector
    log.info("detector %s (seed %d, epoch %d) on %s: %d images", run_dir.name, seed,
             dckpt["epoch"], split, len(frame))  # fmt: skip
    det_results = predict_frame(det, frame, data_root, dckpt["detector_cfg"], dev,
                                num_workers=num_workers)  # fmt: skip
    del det
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_frame(det_results).to_csv(out_dir / f"{split}_detections.csv", index=False)

    # ---- 3. classifier + Grad-CAM baseline
    record = json.loads(Path(ev["classifier_threshold_file"]).read_text())
    threshold = float(record["threshold"])
    cdir = Path(classifier_dir) if classifier_dir else find_run(ev["classifier_run"], SEARCH)
    cam_results, cpreds = gradcam_results(cdir, data_root, split_csv, split, frame, cfg, dev,
                                          num_workers, log)  # fmt: skip
    _, cckpt = load_checkpoint(cdir / "best.pt")
    check_same_model(cckpt, record, md5_of_file(split_csv))
    for r, d in zip(cam_results, det_results, strict=True):
        r.gt = d.gt  # same ground truth for both methods
    flagged = cpreds["prob"].to_numpy() >= threshold
    labels = frame["label"].to_numpy()
    counts = {
        "tp": int((flagged & (labels == 1)).sum()), "fp": int((flagged & (labels == 0)).sum()),
        "tn": int((~flagged & (labels == 0)).sum()), "fn": int((~flagged & (labels == 1)).sum()),
    }  # fmt: skip
    committed = _committed_counts(Path(ev["classifier_threshold_file"]).parent, split)
    if committed is not None and counts != committed:
        raise ProtocolError(f"classifier outcomes {counts} differ from committed {committed}")

    # ---- 4. metrics
    groups = frame["split_group"].to_numpy()
    detector = _metrics_block(det_results, groups, ev)
    baseline = _metrics_block(cam_results, groups, ev)
    rows = []
    for i, (d, c) in enumerate(zip(det_results, cam_results, strict=True)):
        row = {"image_id": d.image_id, "label": d.label, "region": frame["region"].iloc[i],
               "hardware": int(frame["hardware"].iloc[i]), "split_group": groups[i],
               "n_gt": len(d.gt), "classifier_prob": float(cpreds["prob"].iloc[i]),
               "classifier_flag": bool(flagged[i])}  # fmt: skip
        dt = d.top1()
        ct = c.top1()
        row.update({f"det_{k}": v for k, v in dt.items()})
        row.update({f"cam_{k}": v for k, v in ct.items()})
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / f"{split}_per_image.csv", index=False)
    frac = per[per.label == 1]
    by_region = {
        reg: {"n": int(len(g)), "det_top1_pointing": float(g.det_top1_pointing.mean()),
              "cam_top1_pointing": float(g.cam_top1_pointing.mean())}
        for reg, g in frac.groupby("region")
    }  # fmt: skip
    flagged_frac = frac[frac.classifier_flag]
    end_to_end = {
        "fractures": int(len(frac)),
        "flagged_by_classifier": int(len(flagged_frac)),
        "flagged_and_detector_top1_pointing": int(flagged_frac.det_top1_pointing.sum()),
        "flagged_and_detector_top1_iou50": int(flagged_frac.det_top1_iou50.sum()),
    }

    figures_made = []
    if split == "test":
        from fracdet.detect.figures import plot_detection_gallery

        g = cfg["galleries"]
        for spec in g["figures"]:
            items = select_items(per, spec, g["per_figure"], g["seed"])
            if items.empty:
                continue
            path = out_dir / "figures" / f"{split}_{spec['name']}.png"
            lookup = {r.image_id: r for r in det_results}
            plot_detection_gallery(items, frame.set_index("image_id")["path"], lookup, path,
                                   f"{split}: {spec['name'].replace('_', ' ')} "
                                   f"({len(items)} images)")  # fmt: skip
            figures_made.append(str(path.relative_to(out_dir)))

    result = _clean({
        "phase": 7, "split": split,
        "detector": {"run": run_dir.name, "seed": seed, "epoch": dckpt["epoch"],
                     "trained_commit": dckpt["provenance"]["git"]["commit"],
                     "val_at_selection": dckpt["metrics"]},
        "protocol": {"plan": str(config_path), "plan_committed": plan_ok,
                     "plan_commit": plan_commit if plan_ok else None,
                     "rerun_reason": rerun_reason, "classifier_counts": counts,
                     "classifier_committed_counts": committed,
                     "code": git_info(), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                     "seconds": round(time.time() - t0, 1)},
        "detector_metrics": detector, "gradcam_box_metrics": baseline,
        "top1_pointing_by_region": by_region, "end_to_end": end_to_end,
        "figures": figures_made,
    })  # fmt: skip
    _write_json(result, out_dir / "summary.json")
    (out_dir / "report.md").write_text(report_markdown(result))
    log.info("done in %.0f s -> %s", time.time() - t0, out_dir)
    return result


def select_items(per: pd.DataFrame, spec: dict[str, Any], take: int, seed: int) -> pd.DataFrame:
    df = per
    for k, v in spec["select"].items():
        col = f"det_{k}" if f"det_{k}" in df else k
        df = df[df[col] == v]
    if spec.get("order") == "top_score_desc":
        df = df.sort_values("det_top1_score", ascending=False, kind="stable")
    else:
        df = df.sample(frac=1.0, random_state=seed)
    return df.head(take)


def _fmt(block: dict[str, Any], key: str) -> str:
    c = block["ci95"].get(key)
    v = block["point"].get(key)
    if v is None:
        return "–"
    if c is None:
        return f"{v:.3f}"
    return f"{v:.3f} [{c['low']:.3f}, {c['high']:.3f}]"


def report_markdown(result: dict[str, Any]) -> str:
    d, b = result["detector_metrics"], result["gradcam_box_metrics"]
    rows = [
        ("AP50, all images (primary)", "ap50_all"),
        ("AP50, fractured images only", "ap50_fractured"),
        ("AP at IoU 0.25, all images", "ap25_all"),
        ("Fracture boxes found at 0.5 false boxes / image", "sens_at_0.5_fp"),
        ("Fracture boxes found at 1 false box / image", "sens_at_1_fp"),
        ("Fracture boxes found at 2 false boxes / image", "sens_at_2_fp"),
        ("Top box centre in a fracture box (pointing)", "top1_pointing"),
        ("Top box IoU >= 0.5", "top1_iou50"),
    ]
    p = d["point"]
    lines = [
        f"## {result['split']}: detector vs Grad-CAM boxes",
        "",
        f"Detector `{result['detector']['run']}` (seed {result['detector']['seed']}, "
        f"epoch {result['detector']['epoch']}). {p['n_images']} images, {p['n_fractured']} "
        f"fractured, {p['n_gt_boxes']} fracture boxes. 95% CIs resample patient groups.",
        "",
        "| Metric | Detector | Grad-CAM boxes |",
        "|---|---|---|",
        *[f"| {name} | {_fmt(d, k)} | {_fmt(b, k)} |" for name, k in rows],
        "",
        "Top-1 pointing by region (fractured images):",
        "",
        "| Region | n | Detector | Grad-CAM |",
        "|---|---|---|---|",
        *[f"| {r} | {v['n']} | {v['det_top1_pointing']:.3f} | {v['cam_top1_pointing']:.3f} |"
          for r, v in result["top1_pointing_by_region"].items()],
        "",
        f"End to end: {result['end_to_end']}",
        "",
        f"Protocol: {json.dumps(result['protocol'])}",
        "",
    ]  # fmt: skip
    return "\n".join(lines)
