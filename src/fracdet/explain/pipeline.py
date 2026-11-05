"""Phase 6 run over one split: probabilities, heatmaps, localisation scores, figures.

Order of work, so nothing is analysed before it is pinned down:

1. Checks: the analysis plan (``configs/explain.yaml``) is committed (required
   for the test split), and the checkpoint is the one the Phase 5 threshold
   was tuned for.
2. Probabilities with ``score_split``, the exact code that produced the
   committed Phase 5 numbers, so every image keeps its TP / FN / FP / TN label.
   The TP/FP/TN/FN counts must match the committed results.
3. Grad-CAM for every image (fp32), raw maps saved before any scoring.
4. Localisation scores on fractured images, summaries with patient-grouped CIs.
5. Sanity check (validation only) and the pre-declared galleries.
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
from torch.utils.data import DataLoader, Dataset

from fracdet.config import load_config
from fracdet.data.boxes import Box, load_boxes
from fracdet.data.dataset import load_split_frames
from fracdet.data.download import md5_of_file
from fracdet.data.transforms import LetterboxGeometry, eval_transform_with_geometry
from fracdet.eval.bootstrap import cluster_bootstrap
from fracdet.eval.pipeline import _clean, _split_csv, _with_groups, _write_json
from fracdet.eval.predict import score_split
from fracdet.eval.protocol import ProtocolError, check_same_model, committed_and_unchanged
from fracdet.explain import figures
from fracdet.explain.gradcam import GradCAM, resolve_layer, upsample
from fracdet.explain.localize import localization_scores
from fracdet.explain.sanity import randomized_copy, rank_correlation
from fracdet.train.checkpoint import git_info, load_checkpoint

GEOM_FIELDS = ("size", "new_w", "new_h", "left", "top")


class _ExplainDataset(Dataset):
    """Images plus where each one sits in the letterbox square."""

    def __init__(self, paths: list[str], preprocess: dict[str, Any]) -> None:
        self.paths = paths
        self.transform = eval_transform_with_geometry(preprocess)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as img:
            x, g = self.transform(img)
        return x, torch.tensor([getattr(g, f) for f in GEOM_FIELDS]), i


def _geom(row: torch.Tensor | np.ndarray) -> LetterboxGeometry:
    return LetterboxGeometry(*(int(v) for v in row))


def compute_cams(
    model: torch.nn.Module,
    paths: list[str],
    preprocess: dict[str, Any],
    layers: list[str],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Raw Grad-CAM maps per layer (N, h, w), fp32 probabilities (N,), geometries (N, 5)."""
    loader = DataLoader(
        _ExplainDataset(paths, preprocess), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )  # fmt: skip
    gc = GradCAM(model, layers)
    cams: dict[str, list[np.ndarray]] = {n: [] for n in layers}
    probs, geoms = [], []
    for x, g, _ in loader:
        res = gc(x.to(device, non_blocking=True))
        for n in layers:
            cams[n].append(res.cams[n].numpy())
        probs.append(torch.sigmoid(res.logits).numpy())
        geoms.append(g.numpy())
    return (
        {n: np.concatenate(v) for n, v in cams.items()},
        np.concatenate(probs),
        np.concatenate(geoms),
    )


def _rate_with_ci(hits: np.ndarray, groups: np.ndarray, boot: dict[str, Any]) -> dict[str, Any]:
    """Hit rate with a patient-grouped (cluster) bootstrap 95% interval."""
    if len(hits) == 0:
        return {"n": 0, "value": None}
    ci = cluster_bootstrap(
        hits.astype(float), np.zeros(len(hits)), groups,
        lambda y, _p: {"rate": float(np.mean(y))},
        n_boot=boot["n"], seed=boot["seed"], alpha=boot["alpha"],
    )["rate"]  # fmt: skip
    return {"n": int(len(hits)), "value": ci["value"], "low": ci["low"], "high": ci["high"]}


def summarize(
    scores: pd.DataFrame,
    layers: list[str],
    boot: dict[str, Any],
    group_by: tuple[str, ...] = ("outcome", "region"),
) -> dict[str, Any]:
    """Pointing game + energy share per layer, overall and by outcome / region."""

    def block(df: pd.DataFrame) -> dict[str, Any]:
        groups = df["split_group"].to_numpy()
        out: dict[str, Any] = {
            "n_images": int(len(df)),
            "n_groups": int(df["split_group"].nunique()),
            "chance_hit_rate": float(df["area_share"].mean()) if len(df) else None,
            "centre_hit_rate": _rate_with_ci(df["centre_hit"].to_numpy(), groups, boot),
        }
        for n in layers:
            out[n] = {
                "hit_rate": _rate_with_ci(df[f"{n}:hit"].to_numpy(), groups, boot),
                "energy_share_mean": float(df[f"{n}:energy_share"].mean()) if len(df) else None,
                "energy_lift_median": (
                    float((df[f"{n}:energy_share"] / df["area_share"]).median())
                    if len(df)
                    else None
                ),
                "empty_maps": int(df[f"{n}:empty"].sum()),
            }
        return out

    result = {"all": block(scores)}
    for col in group_by:
        for value in sorted(scores[col].unique()):
            result[f"{col}={value}"] = block(scores[scores[col] == value])
    return result


def select_gallery(preds: pd.DataFrame, spec: dict[str, Any], take: int, seed: int) -> pd.DataFrame:
    """Images for one gallery, chosen by the rule in the config (no looking first)."""
    df = preds
    for key, value in spec["select"].items():
        keep = df["region"] != value if key == "region_not" else df[key] == value
        df = df[keep]
    if spec.get("order") == "prob_desc":
        df = df.sort_values("prob", ascending=False, kind="stable")
    elif spec.get("order") == "random":
        df = df.sample(frac=1.0, random_state=seed)
    if spec.get("take") != "all":
        df = df.head(take)
    return df.sort_values("prob", ascending=False, kind="stable")


def _committed_counts(results_dir: Path, split: str) -> dict[str, int] | None:
    keys = ("tp", "fp", "tn", "fn")
    if split == "test":
        f = results_dir / "test_metrics.json"
        if f.is_file():
            m = json.loads(f.read_text())["test"]["all"]["metrics"]
            return {k: int(m[k]) for k in keys}
    if split == "val":
        f = results_dir / "threshold.json"
        if f.is_file():
            m = json.loads(f.read_text())["val"]["at_threshold"]
            return {k: int(m[k]) for k in keys}
    return None


def explain_split(
    run_dir: str | Path,
    data_root: str | Path,
    cfg: dict[str, Any],
    split: str,
    out_dir: str | Path,
    log: logging.Logger,
    config_path: str | Path = "configs/explain.yaml",
    require_preregistration: bool = True,
    num_workers: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    # ---- 1. checks
    plan_ok, plan_commit = committed_and_unchanged(config_path)
    if not plan_ok:
        if split == "test" and require_preregistration:
            raise ProtocolError(f"Analysis plan not committed: {plan_commit}")
        log.warning("analysis plan not committed (%s): fine for a dry run only", plan_commit)
    record = json.loads(Path(cfg["threshold_file"]).read_text())
    threshold = float(record["threshold"])
    run_cfg = load_config(run_dir / "config.yaml")
    split_csv = _split_csv(run_cfg)

    # ---- 2. probabilities exactly as in Phase 5, and outcome labels
    log.info("scoring %s with %s", split, run_dir / "best.pt")
    preds, ckpt = score_split(
        run_dir / "best.pt", data_root, split_csv, split, device=device_t, num_workers=num_workers
    )
    check_same_model(ckpt, record, md5_of_file(split_csv))
    preds = _with_groups(preds, split_csv)
    if "split_group" not in preds:  # split without groups -> per-image bootstrap
        preds["split_group"] = preds["image_id"]
    pred_pos = preds["prob"] >= threshold
    preds["outcome"] = np.select(
        [pred_pos & (preds["label"] == 1), ~pred_pos & (preds["label"] == 1), pred_pos],
        ["TP", "FN", "FP"],
        "TN",
    )
    counts = {k.lower(): int((preds["outcome"] == k).sum()) for k in ("TP", "FP", "TN", "FN")}
    committed = _committed_counts(Path(cfg["threshold_file"]).parent, split)
    log.info("outcomes %s | committed %s", counts, committed)
    if committed is not None and counts != committed:
        raise ProtocolError(f"Outcome counts {counts} differ from committed {committed}")

    # ---- 3. Grad-CAM (fp32), raw maps saved before any scoring
    model, _ = load_checkpoint(run_dir / "best.pt", device=device_t)
    g = cfg["gradcam"]
    layers = [resolve_layer(model, i) for i in [g["primary_layer"], *g["extra_layers"]]]
    primary = layers[0]
    frame = load_split_frames(data_root, split_csv, [split])[split]
    if frame["image_id"].tolist() != preds["image_id"].tolist():
        raise RuntimeError("split order changed between scoring and Grad-CAM")
    preds["path"] = frame["path"].to_numpy()
    preprocess = ckpt.get("preprocess", {})
    log.info("Grad-CAM on %d images, layers %s, device %s", len(preds), layers, device_t)
    cams, prob32, geoms = compute_cams(
        model, preds["path"].tolist(), preprocess, layers, device_t,
        batch_size=g["batch_size"], num_workers=num_workers,
    )  # fmt: skip
    preds["prob_fp32"] = prob32
    max_prob_diff = float(np.abs(prob32 - preds["prob"].to_numpy()).max())
    log.info("max |prob(fp32) - prob(Phase 5 scoring)| = %.2e", max_prob_diff)
    np.savez_compressed(
        out_dir / "cams.npz",
        image_id=preds["image_id"].to_numpy(),
        geometry=geoms,
        **{n.replace(".", "_"): cams[n].astype(np.float32) for n in layers},
    )

    # ---- 4. localisation on fractured images
    size = int(geoms[0][0])
    boxes: dict[str, list[Box]] = {}
    rows = []
    for i in np.flatnonzero(preds["label"].to_numpy() == 1):
        r = preds.iloc[i]
        b = load_boxes(data_root, r["image_id"])
        if not b:
            raise ValueError(f"fractured image {r['image_id']} has no box")
        boxes[r["image_id"]] = b
        geom = _geom(geoms[i])
        row = {
            k: r[k] for k in ("image_id", "region", "hardware", "outcome", "prob", "split_group")
        }
        for n in layers:
            s = localization_scores(upsample(cams[n][i], size), b, geom)
            row["area_share"], row["centre_hit"] = s["area_share"], s["centre_hit"]
            row["n_boxes"] = s["n_boxes"]
            for k in ("hit", "energy_share", "empty", "peak_y", "peak_x"):
                row[f"{n}:{k}"] = s[k]
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "localization_per_image.csv", index=False)
    boot = {"n": 2000, "seed": 0, "alpha": 0.05}
    summary = summarize(scores, layers, boot, tuple(cfg["localization"]["groups"]))

    # ---- 5a. sanity check (validation only)
    sanity: dict[str, Any] | None = None
    if split == "val":
        sanity = run_sanity(model, ckpt, preds, scores, cams, geoms, boxes, cfg, primary, device_t,
                            preprocess, num_workers, log)  # fmt: skip

    # ---- 5b. galleries
    preds["outcome"] = preds["outcome"].astype(str)
    gal = cfg["galleries"]
    made = []
    for spec in gal["figures"]:
        items = select_gallery(preds, spec, gal["per_figure"], gal["seed"])
        if items.empty:
            continue
        up = {
            r.image_id: upsample(cams[primary][preds.index.get_loc(idx)], size)
            for idx, r in zip(items.index, items.itertuples(index=False), strict=True)
        }
        path = out_dir / "figures" / f"{split}_{spec['name']}.png"
        figures.plot_gallery(
            items, up, boxes, path,
            f"{split}: {spec['name'].replace('_', ' ')} "
            f"({len(items)} images, Grad-CAM at {primary})",
            size,
        )  # fmt: skip
        made.append(str(path.relative_to(out_dir)))

    s_all = summary["all"]
    bars = [
        {"label": f"Grad-CAM, {primary}", **_bar(s_all[primary]["hit_rate"])},
        *[{"label": f"Grad-CAM, {n}", **_bar(s_all[n]["hit_rate"])} for n in layers[1:]],
    ]
    if sanity:
        for level, v in sanity["levels"].items():
            bars.append({"label": f"random weights: {level}", "value": v["hit_rate"],
                         "kind": "baseline"})  # fmt: skip
    bars += [
        {"label": "centre of image", **_bar(s_all["centre_hit_rate"]), "kind": "baseline"},
        {"label": "chance (box area)", "value": s_all["chance_hit_rate"], "kind": "baseline"},
    ]
    figures.plot_pointing_summary(
        bars, out_dir / "figures" / f"{split}_pointing_game.png",
        f"{split}: is the heatmap's peak on the fracture box?",
    )  # fmt: skip

    result = _clean(
        {
            "phase": 6,
            "split": split,
            "model": record["model"],
            "threshold": threshold,
            "layers": {"primary": primary, "extra": layers[1:]},
            "protocol": {
                "analysis_plan": str(config_path),
                "plan_committed": plan_ok,
                "plan_commit": plan_commit if plan_ok else None,
                "outcome_counts": counts,
                "committed_counts": committed,
                "max_prob_diff_fp32": max_prob_diff,
                "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "code": git_info(),
                "device": torch.cuda.get_device_name(0) if device_t.type == "cuda" else "cpu",
                "seconds": round(time.time() - t0, 1),
                "bootstrap": {**boot, "unit": "split_group"},
            },
            "localization": summary,
            "sanity": sanity,
            "figures": made,
        }
    )
    _write_json(result, out_dir / "summary.json")
    (out_dir / "report.md").write_text(report_markdown(result))
    log.info("done in %.0f s -> %s", time.time() - t0, out_dir)
    return result


def _bar(rate: dict[str, Any]) -> dict[str, Any]:
    return {"value": rate["value"], "low": rate.get("low"), "high": rate.get("high")}


def run_sanity(model, ckpt, preds, scores, cams, geoms, boxes, cfg, layer, device, preprocess,
               num_workers, log) -> dict[str, Any]:  # fmt: skip
    """Model-randomisation test on the first ``n_images`` fractured validation images."""
    s = cfg["sanity"]
    idx = np.flatnonzero(preds["label"].to_numpy() == 1)[: s["n_images"]]
    paths = preds["path"].iloc[idx].tolist()
    size = int(geoms[0][0])
    trained_hits = scores.set_index("image_id").loc[preds["image_id"].iloc[idx], f"{layer}:hit"]
    out: dict[str, Any] = {
        "n_images": len(idx),
        "trained_hit_rate": float(trained_hits.mean()),
        "levels": {},
    }
    for level in s["levels"]:
        rnd = randomized_copy(model, ckpt["model_cfg"], level, seed=s["seed"])
        rcams, _, _ = compute_cams(rnd, paths, preprocess, [layer], device,
                                   batch_size=cfg["gradcam"]["batch_size"],
                                   num_workers=num_workers)  # fmt: skip
        corrs, hits = [], []
        for j, i in enumerate(idx):
            geom = _geom(geoms[i])
            rs, cs = geom.content
            a = upsample(cams[layer][i], size)
            b = upsample(rcams[layer][j], size)
            corrs.append(rank_correlation(a[rs, cs], b[rs, cs]))
            hits.append(localization_scores(b, boxes[preds["image_id"].iloc[i]], geom)["hit"])
        c = np.asarray(corrs, dtype=float)
        out["levels"][level] = {
            "spearman_median": float(np.nanmedian(c)) if np.isfinite(c).any() else None,
            "spearman_mean": float(np.nanmean(c)) if np.isfinite(c).any() else None,
            "flat_maps": int(np.isnan(c).sum()),
            "hit_rate": float(np.mean(hits)),
        }
        log.info("sanity %s: %s", level, out["levels"][level])
    return out


def _fmt_rate(r: dict[str, Any] | None) -> str:
    if not r or r.get("value") is None:
        return "–"
    if r.get("low") is None:
        return f"{r['value']:.3f}"
    return f"{r['value']:.3f} [{r['low']:.3f}, {r['high']:.3f}]"


def report_markdown(result: dict[str, Any]) -> str:
    """The summary tables, ready to paste into the docs."""
    loc = result["localization"]
    primary = result["layers"]["primary"]
    layers = [primary, *result["layers"]["extra"]]
    lines = [
        f"## {result['split']}: Grad-CAM localisation (fractured images)",
        "",
        "Pointing game = share of fractured images whose heatmap peak is inside a fracture box. "
        "95% CIs resample patient groups.",
        "",
        "| Group | n | "
        + " | ".join(f"hit rate, {n}" for n in layers)
        + " | centre | chance | energy share / area share, "
        + primary
        + " |",
        "|---|---|" + "---|" * len(layers) + "---|---|---|",
    ]
    for name, b in loc.items():
        if b["n_images"] == 0:
            continue
        cells = [_fmt_rate(b[n]["hit_rate"]) for n in layers]
        p = b[primary]
        lines.append(
            f"| {name} | {b['n_images']} | "
            + " | ".join(cells)
            + f" | {_fmt_rate(b['centre_hit_rate'])} | {b['chance_hit_rate']:.3f}"
            + f" | {p['energy_share_mean']:.3f} / {b['chance_hit_rate']:.3f} |"
        )
    s = result.get("sanity")
    if s:
        lines += [
            "",
            f"## Sanity check: randomised weights ({s['n_images']} fractured validation images)",
            "",
            f"Trained model hit rate on these images: {s['trained_hit_rate']:.3f}",
            "",
            "| Randomised | Spearman with trained maps (median) | hit rate | flat maps |",
            "|---|---|---|---|",
        ]
        for level, v in s["levels"].items():
            rho = "–" if v["spearman_median"] is None else f"{v['spearman_median']:.3f}"
            lines.append(f"| {level} | {rho} | {v['hit_rate']:.3f} | {v['flat_maps']} |")
    pr = result["protocol"]
    lines += [
        "",
        f"Outcomes at threshold {result['threshold']:.4f}: {pr['outcome_counts']} "
        f"(committed: {pr['committed_counts']}). Plan committed: {pr['plan_committed']} "
        f"({pr['plan_commit']}). Device: {pr['device']}, {pr['seconds']} s.",
        "",
    ]
    return "\n".join(lines)
