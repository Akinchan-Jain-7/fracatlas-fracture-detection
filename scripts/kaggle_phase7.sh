#!/usr/bin/env bash
# Phase 7 on Kaggle (T4 x2), in one go:
#   1. train seeds 42 and 43 in parallel, one per GPU;
#   2. evaluate both on validation (with the Grad-CAM box baseline);
#   3. evaluate the pre-registered seed (42) on test, once;
#   4. zip everything worth keeping into /kaggle/working/phase7_detect.zip.
# Everything that decides the result is in configs/detect.yaml, committed before this runs.
# Run from the repo root with FRACDET_DATA_DIR set (the notebook does both).
set -euo pipefail

mkdir -p outputs/logs
CUDA_VISIBLE_DEVICES=0 python scripts/train_detector.py --seed 42 > outputs/logs/train_s42.log 2>&1 &
P42=$!
CUDA_VISIBLE_DEVICES=1 python scripts/train_detector.py --seed 43 > outputs/logs/train_s43.log 2>&1 &
P43=$!
wait $P42 || { echo "seed 42 failed"; tail -50 outputs/logs/train_s42.log; exit 1; }
wait $P43 || { echo "seed 43 failed (seed 42 is still evaluated)"; tail -30 outputs/logs/train_s43.log; }
grep -h "epoch\|done" outputs/logs/train_s4*.log | tail -60

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_detector.py --seed 42 --split val
if ls outputs/runs/*_detect_frcnn_r50_s43/best.pt >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_detector.py --seed 43 --split val
fi
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_detector.py --seed 42 --split test

cat outputs/detect_eval/*/val/report.md outputs/detect_eval/*/test/report.md
# Keep the evaluations, run records and the reported model's weights (for the Phase 8 app).
zip -qr /kaggle/working/phase7_detect.zip outputs/detect_eval outputs/logs \
  outputs/runs/*_detect_frcnn_r50_s4*/{config.yaml,train.log,metrics.csv,summary.json,val_detections.csv}
cp outputs/runs/*_detect_frcnn_r50_s42/best.pt /kaggle/working/detector_s42_best.pt
du -sh /kaggle/working/phase7_detect.zip /kaggle/working/detector_s42_best.pt
