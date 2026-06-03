#!/usr/bin/env bash
# run_post_pocket_eval.sh — post-pocket-fix benchmark: 60 complexes × 8 models × 25 poses
#
# Re-evaluates all model variants with the crop_to_pocket() fix in
# prepare_receptor_pdb() applied end-to-end.  Uses ref2015 (PyRosetta) for
# pose ranking.
#
# Models: pretrained, v1, v2, v3, v3c, v4c, v5c, v6
# Dataset: 60 stratified complexes (5 per SS×length cell) from benchmark300.csv
# n-samples: 25 per model  seed: 42
# Estimated runtime: 8-14 hours on RTX 5070
#
# Usage:
#   nohup bash scripts/run_post_pocket_eval.sh \
#       > logs/analysis_post_pocket_fix/run.log 2>&1 &
#   tail -f logs/analysis_post_pocket_fix/run.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/logs/analysis_post_pocket_fix"
mkdir -p "$OUT"

echo "============================================================"
echo "[post-pocket-eval] Starting 8-model benchmark"
echo "  Dataset:   benchmark300.csv → 60 stratified complexes"
echo "  Models:    pretrained + v1 + v2 + v3 + v3c + v4c + v5c + v6"
echo "  n-samples: 25  seed: 42"
echo "  Out dir:   $OUT"
echo "  Started:   $(date)"
echo "============================================================"

conda run --no-capture-output -n score-env python3 -u \
    "$REPO/scripts/run_post_pocket_eval.py" \
    --n-samples   25 \
    --n-per-cell  5 \
    --seed        42 \
    --out-dir     "$OUT"

echo ""
echo "============================================================"
echo "[post-pocket-eval] DONE — $(date)"
echo "  Results: $OUT/benchmark_summary.csv"
echo "           $OUT/aggregate_stats.json"
echo "           $OUT/model_comparison.png"
echo "============================================================"
