#!/usr/bin/env bash
# Sequential chain: v4c done → v3c done → v5c
# Runs analysis after each experiment completes
set -euo pipefail
REPO="/home/igem/unknown_software"
PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
BENCH_CSV="$REPO/data/benchmark30.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [seq-chain] $*"; }

wait_for_file() {
    local f="$1" label="$2"
    log "Waiting for $label ($f)..."
    until [[ -f "$f" ]]; do sleep 60; done
    log "$label COMPLETE ✅"
}

run_analysis() {
    local label="$1" ckpt="$2" outdir="$3"
    log "=== ANALYSIS $label ==="
    mkdir -p "$outdir"
    conda run --no-capture-output -n score-env python3 -u \
        "$REPO/scripts/benchmark_inference_multi.py" \
        --benchmark-csv  "$BENCH_CSV" \
        --pretrained     "$PRETRAINED" \
        --finetuned      "$ckpt" \
        --n-samples 5 --seed 42 \
        --out-dir        "$outdir" 2>&1 | tee "$outdir/analysis.log"
    log "=== ANALYSIS $label DONE ==="
}

# ── 1. Wait for v4c to fully complete ──────────────────────────────────────
V4C_P2_FINAL="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase2/rapidock_finetuned_final.pt"
wait_for_file "$V4C_P2_FINAL" "v4c P2"
run_analysis "1 (v4c)" \
    "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase2/rapidock_finetuned_best.pt" \
    "$REPO/logs/analysis_v4c"

# ── 2. Launch v3c (now that v4c is done and GPU is free), wait for it ────────
V3C_P2_FINAL="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase2/rapidock_finetuned_final.pt"
if [[ ! -f "$V3C_P2_FINAL" ]]; then
    log "Launching v3c (v4c done, GPU free)..."
    sleep 15  # let VRAM settle
    nohup bash "$REPO/scripts/chain_training_v3c.sh" >> "$REPO/logs/chain_training_v3c.log" 2>&1 &
    log "v3c launched PID $!"
fi
wait_for_file "$V3C_P2_FINAL" "v3c P2"
run_analysis "2 (v3c)" \
    "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase2/rapidock_finetuned_best.pt" \
    "$REPO/logs/analysis_v3c"

# ── 3. Launch v5c, wait for it to complete ─────────────────────────────────
log "Launching v5c..."
nohup bash "$REPO/scripts/chain_training_v5c.sh" >> "$REPO/logs/chain_training_v5c.log" 2>&1 &
V5C_PID=$!
log "v5c launched PID $V5C_PID"

V5C_P2_FINAL="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2/rapidock_finetuned_final.pt"
wait_for_file "$V5C_P2_FINAL" "v5c P2"
run_analysis "3 (v5c)" \
    "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2/rapidock_finetuned_best.pt" \
    "$REPO/logs/analysis_v5c"

# ── 4. Final comparison: pretrained vs all three ───────────────────────────
log "=== ANALYSIS 4 (full comparison) ==="
mkdir -p "$REPO/logs/analysis_final"
conda run --no-capture-output -n score-env python3 -u \
    "$REPO/scripts/benchmark_inference_multi.py" \
    --benchmark-csv  "$BENCH_CSV" \
    --pretrained     "$PRETRAINED" \
    --finetuned      "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase2/rapidock_finetuned_best.pt" \
    --also-compare \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase2/rapidock_finetuned_best.pt" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2/rapidock_finetuned_best.pt" \
    --n-samples 5 --seed 42 \
    --out-dir "$REPO/logs/analysis_final" 2>&1 | tee "$REPO/logs/analysis_final/analysis.log"
log "=== ALL ANALYSES COMPLETE ==="
