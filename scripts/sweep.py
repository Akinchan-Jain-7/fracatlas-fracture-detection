"""Run every (config, seed) of a sweep file, one queue per GPU.

Each job is an ordinary ``scripts/train.py`` run (its own run folder, log and
checkpoint), with ``seed`` and ``run_name=<name>_s<seed>`` overridden. Jobs are
ordered seed-major (all variants for seed 42, then 43, ...), so a sweep that
stops early still has complete seeds to compare. A failed job is recorded and
the sweep continues.

Writes ``outputs/sweeps/<sweep name>/manifest.json`` (updated as jobs finish)
and one log per job next to it.

Usage:
    python scripts/sweep.py configs/sweep_phase4.yaml                 # one GPU / CPU
    python scripts/sweep.py configs/sweep_phase4.yaml --gpus 0,1 --set train.num_workers=2
    python scripts/sweep.py configs/sweep_phase4.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


def jobs_for(sweep: dict, config_dir: Path) -> list[dict]:
    jobs = []
    for seed in sweep["seeds"]:
        for name in sweep["configs"]:
            cfg_path = config_dir / name
            run_name = yaml.safe_load(cfg_path.read_text()).get("run_name", cfg_path.stem)
            jobs.append(
                {
                    "experiment": run_name,
                    "seed": seed,
                    "config": str(cfg_path),
                    "run_name": f"{run_name}_s{seed}",
                    "status": "pending",
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--gpus", default="", help="comma-separated GPU ids; empty = one worker")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="for every job")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep = yaml.safe_load(args.sweep.read_text())
    out = args.out_root / sweep["name"]
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(sweep, args.sweep.parent)
    workers = [g.strip() for g in args.gpus.split(",") if g.strip()] or [None]
    manifest = {"sweep": sweep["name"], "sweep_file": str(args.sweep), "jobs": jobs}
    lock = threading.Lock()

    def save() -> None:
        with lock:
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.dry_run:
        for j in jobs:
            print(j["run_name"], j["config"])
        return

    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: str | None) -> None:
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            cmd = [
                sys.executable, "scripts/train.py", "--config", job["config"],
                "--set", f"seed={job['seed']}", f"run_name={job['run_name']}",
                f"experiment={job['experiment']}", *args.set,
            ]  # fmt: skip
            job.update(status="running", gpu=gpu, started=time.strftime("%H:%M:%S"))
            save()
            t0 = time.time()
            log_path = out / f"{job['run_name']}.log"
            with log_path.open("w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            lines = log_path.read_text().splitlines()
            summary = next((ln for ln in reversed(lines) if ln.startswith("SUMMARY_JSON ")), None)
            job.update(
                status="done" if rc == 0 and summary else "failed",
                returncode=rc,
                minutes=round((time.time() - t0) / 60, 1),
                run_dir=json.loads(summary[13:])["run_dir"] if summary else None,
            )
            save()
            print(f"[{time.strftime('%H:%M:%S')}] {job['run_name']}: {job['status']} "
                  f"({job['minutes']} min, gpu {gpu})", flush=True)  # fmt: skip

    save()
    threads = [threading.Thread(target=worker, args=(g,)) for g in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [j["run_name"] for j in jobs if j["status"] != "done"]
    print(
        f"sweep done: {len(jobs) - len(failed)}/{len(jobs)} ok"
        + (f"; failed {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
