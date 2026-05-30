#!/usr/bin/env bash
# run_bench300_overnight.sh — 4-model overnight benchmark on 240 balanced complexes
#
# Models: pretrained | v3c | v4c | v5c
# Dataset: data/benchmark300.csv (240 complexes: 3 SS × 4 len × 20 each)
# n-samples: 5, seed: 42
# Estimated runtime: ~10-14 hours on RTX 5070
#
# Usage:
#   nohup bash scripts/run_bench300_overnight.sh > logs/bench300_overnight.log 2>&1 &
#   tail -f logs/bench300_overnight.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
V3C="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase2/rapidock_finetuned_best.pt"
V4C="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase2/rapidock_finetuned_best.pt"
V5C="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2/rapidock_finetuned_best.pt"

BENCH_CSV="$REPO/data/benchmark300.csv"
OUT_DIR="$REPO/logs/analysis_bench300"

for f in "$PRETRAINED" "$V3C" "$V4C" "$V5C" "$BENCH_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

echo "============================================================"
echo "[bench300] Starting 4-model overnight benchmark"
echo "  Dataset:  $BENCH_CSV ($(wc -l < "$BENCH_CSV") lines)"
echo "  Out dir:  $OUT_DIR"
echo "  Models:   pretrained + v3c + v4c + v5c"
echo "  n-samples: 5  seed: 42"
echo "  Started:  $(date)"
echo "============================================================"

mkdir -p "$OUT_DIR"

conda run --no-capture-output -n score-env python3 -u \
    "$REPO/scripts/benchmark_inference_multi.py" \
    --benchmark-csv  "$BENCH_CSV" \
    --pretrained     "$PRETRAINED" \
    --finetuned      "$V5C" \
    --also-compare   "$V3C" "$V4C" \
    --n-samples      5 \
    --seed           42 \
    --out-dir        "$OUT_DIR"

echo ""
echo "============================================================"
echo "[bench300] DONE — $(date)"
echo "  Results: $OUT_DIR/benchmark_summary.csv"
echo "============================================================"
