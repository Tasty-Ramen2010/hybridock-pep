#!/usr/bin/env bash
# chain_training_phases.sh — launches Phase 1, then Phase 2, then Phase 3 in sequence.
#
# Usage:
#   bash scripts/chain_training_phases.sh 2>&1 | tee logs/chain_training.log &
#
# Phase layout:
#   Phase 1: score heads + output convs (27.9% = 2.1M params), 30 epochs, lr=1e-4
#   Phase 2: +last two equivariant blocks (73.0% = 5.5M params), 50 epochs, lr=5e-5, warmup=3
#   Phase 3: full retrain (all 7.5M params), 200 epochs, lr=1e-4, warmup=10
#
# Key fix (May 27 2026): --esm-device cpu avoids WSL2 TDR crash that killed the
#   previous run at batch 790/874 during ESM embedding pre-computation.
#   ESM runs once per phase on CPU (~40 min per phase); the training loop then
#   uses GPU for the actual forward/backward passes.
#
# All checkpoints saved in finetune_peppc_phase{N}/ output dirs.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val.csv"
PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase3"

P1_BEST="$P1_OUT/rapidock_finetuned_best.pt"
P2_BEST="$P2_OUT/rapidock_finetuned_best.pt"
P3_BEST="$P3_OUT/rapidock_finetuned_best.pt"

# Common conda runner
CONDA="conda run --no-capture-output -n rapidock"

# ── Helper: assert checkpoint was produced ────────────────────────────────────
require_ckpt() {
    local path="$1"
    local phase="$2"
    if [ ! -f "$path" ]; then
        echo ""
        echo "========================================================================"
        echo "[chain] FATAL: $phase did not produce a best checkpoint."
        echo "  Expected: $path"
        echo "  Check logs above for the Python traceback."
        echo "========================================================================"
        exit 2
    fi
    echo "[chain] $phase checkpoint: $path ($(du -h "$path" | cut -f1))"
}

# ─── Phase 1 ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain] Phase 1 — score heads only (27.9%), lr=1e-4, 30 epochs"
echo "  ESM device: cpu  (avoids WSL2 TDR crash at batch 790/874)"
echo "========================================================================"
mkdir -p "$P1_OUT"
$CONDA python3 "$SCRIPT" \
    --train-csv      "$TRAIN_CSV" \
    --val-csv        "$VAL_CSV" \
    --checkpoint     "$PRETRAINED" \
    --output-dir     "$P1_OUT" \
    --unfreeze-phase 1 \
    --n-epochs       30 \
    --lr             1e-4 \
    --warmup-epochs  0 \
    --grad-accum     4 \
    --save-every     5 \
    --seed           42 \
    --esm-device     cpu \
    --bail-on-zero

require_ckpt "$P1_BEST" "Phase 1"
echo "[chain] Phase 1 complete."

# Give a moment before starting Phase 2
sleep 5

# ─── Phase 2 ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain] Phase 2 — +last 2 equivariant blocks (73.0%), lr=5e-5, 50 epochs, warmup=3"
echo "  ESM device: cpu  (one-time cost; training loop stays on GPU)"
echo "========================================================================"
mkdir -p "$P2_OUT"
$CONDA python3 "$SCRIPT" \
    --train-csv      "$TRAIN_CSV" \
    --val-csv        "$VAL_CSV" \
    --checkpoint     "$P1_BEST" \
    --output-dir     "$P2_OUT" \
    --unfreeze-phase 2 \
    --n-epochs       50 \
    --lr             5e-5 \
    --warmup-epochs  3 \
    --grad-accum     4 \
    --save-every     5 \
    --seed           42 \
    --esm-device     cpu \
    --bail-on-zero

require_ckpt "$P2_BEST" "Phase 2"
echo "[chain] Phase 2 complete."

sleep 5

# ─── Phase 3 ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain] Phase 3 — full retrain (all 7.5M params), lr=1e-4, 200 epochs, warmup=10"
echo "  ESM device: cpu  (one-time cost; training loop stays on GPU)"
echo "========================================================================"
mkdir -p "$P3_OUT"
$CONDA python3 "$SCRIPT" \
    --train-csv      "$TRAIN_CSV" \
    --val-csv        "$VAL_CSV" \
    --checkpoint     "$P2_BEST" \
    --output-dir     "$P3_OUT" \
    --unfreeze-phase 3 \
    --n-epochs       200 \
    --lr             1e-4 \
    --warmup-epochs  10 \
    --grad-accum     4 \
    --save-every     10 \
    --seed           42 \
    --esm-device     cpu \
    --bail-on-zero

if [ ! -f "$P3_BEST" ]; then
    echo "[chain] WARNING: Phase 3 did not produce a best checkpoint (val loss may never have improved)."
    # Check for final checkpoint as fallback
    P3_FINAL="$P3_OUT/rapidock_finetuned_final.pt"
    if [ -f "$P3_FINAL" ]; then
        echo "[chain] Phase 3 final checkpoint exists: $P3_FINAL"
    fi
else
    echo "[chain] Phase 3 complete. Best: $P3_BEST"
fi

echo ""
echo "========================================================================"
echo "[chain] ALL 3 PHASES COMPLETE."
echo "  Phase 1 best: $P1_BEST"
echo "  Phase 2 best: $P2_BEST"
echo "  Phase 3 best: $P3_BEST"
echo "========================================================================"
