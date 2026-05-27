#!/usr/bin/env bash
# chain_training_phases.sh — launches Phase 1, then Phase 2, then Phase 3 in sequence.
#
# Usage:
#   bash scripts/chain_training_phases.sh 2>&1 | tee logs/chain_training.log &
#
# Phase layout (hyperparameters revised after deep analysis — see docs/training_strategy_analysis.md):
#   Phase 1: score heads + output convs + output embeddings (27.9% = 2.1M), 30 ep, lr=1e-4
#   Phase 2: +cross_convs.2/3 + intra_convs.3 (NOT intra_convs.2), 50 ep, lr=2e-5, warmup=5
#            cross_convs prioritised over intra_convs; intra_convs.2 held for P3 full context
#   Phase 3: full retrain all 7.5M, 100 ep (was 200), lr=2e-5→1e-7 cosine (was 1e-4 plateau)
#            lower LR+epochs prevent catastrophic forgetting of inner-layer physics priors
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
# cross_convs.3, cross_convs.2, intra_convs.3 (NOT intra_convs.2 — see analysis)
# LR reduced 5e-5 → 2e-5; warmup 3 → 5 (more conservative since we're touching
# deeper geometric layers; intra_convs.2 frozen until Phase 3 full-model context)
echo ""
echo "========================================================================"
echo "[chain] Phase 2 — cross_convs.2/3 + intra_convs.3, lr=2e-5, 50 epochs, warmup=5"
echo "  ESM device: cpu  (cache hit expected — ~30s load vs ~4h recompute)"
echo "========================================================================"
mkdir -p "$P2_OUT"
$CONDA python3 "$SCRIPT" \
    --train-csv      "$TRAIN_CSV" \
    --val-csv        "$VAL_CSV" \
    --checkpoint     "$P1_BEST" \
    --output-dir     "$P2_OUT" \
    --unfreeze-phase 2 \
    --n-epochs       50 \
    --lr             2e-5 \
    --warmup-epochs  5 \
    --lr-schedule    plateau \
    --grad-accum     4 \
    --save-every     5 \
    --seed           42 \
    --esm-device     cpu \
    --bail-on-zero

require_ckpt "$P2_BEST" "Phase 2"
echo "[chain] Phase 2 complete."

sleep 5

# ─── Phase 3 ─────────────────────────────────────────────────────────────────
# Full retrain — all 7.5M params including intra_convs.0/1, cross_convs.0/1
# LR reduced 1e-4 → 2e-5  (5× lower: prevents catastrophic forgetting of inner
#   layers' physically grounded geometric representations)
# Epochs reduced 200 → 100  (after 80 epochs of P1+P2 the remaining optimum is
#   close; 200 full-model epochs at any LR risks eroding physics priors)
# Cosine schedule (not plateau): deterministic monotone decay avoids oscillation
#   around pre-trained inner-layer representations; reaches 1e-7 at epoch 100
echo ""
echo "========================================================================"
echo "[chain] Phase 3 — full retrain (all 7.5M), lr=2e-5→1e-7 cosine, 100 epochs, warmup=10"
echo "  ESM device: cpu  (cache hit expected)"
echo "========================================================================"
mkdir -p "$P3_OUT"
$CONDA python3 "$SCRIPT" \
    --train-csv      "$TRAIN_CSV" \
    --val-csv        "$VAL_CSV" \
    --checkpoint     "$P2_BEST" \
    --output-dir     "$P3_OUT" \
    --unfreeze-phase 3 \
    --n-epochs       100 \
    --lr             2e-5 \
    --warmup-epochs  10 \
    --lr-schedule    cosine \
    --cosine-min-lr  1e-7 \
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
