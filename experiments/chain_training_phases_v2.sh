#!/usr/bin/env bash
# chain_training_phases_v2.sh — Improved fine-tuning chain starting from
# the ORIGINAL pretrained checkpoint.
#
# Key differences from v1:
#   Phase 1: lr=2e-5 (was 1e-4), warmup=5, epochs=25, grad-clip=0.5
#             — prevents score-head destabilisation (v1 caused val=1e16 at epoch 11)
#   Phase 2: lr=5e-6 (was 2e-5), warmup=8, epochs=45
#             — deeper layers need much lower LR to avoid overwriting physics priors
#   Phase 3: lr=1e-5, warmup=10, epochs=100, cosine, layerwise LR decay,
#             early-stop=20, grad-clip=1.0
#             — layer-wise decay: heads 1.0×, late convs 0.5×, middle 0.2×,
#               early equivariant 0.05× (protects geometric/physical priors)
#
# Output dirs: finetune_peppc_v2_phase{1,2,3}/
# Log: logs/chain_training_v2.log
#
# Usage:
#   bash experiments/chain_training_phases_v2.sh 2>&1 | tee logs/chain_training_v2.log &

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

# ── Always start from the ORIGINAL pretrained checkpoint ─────────────────────
# This is deliberate: v1's Phase 1 best.pt (epoch 4) is ~99.6% pretrained anyway,
# so there is no benefit to inheriting it. Starting from pretrained gives a clean
# baseline and lets us measure the full benefit of the improved hyperparameters.
PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3"

P1_BEST="$P1_OUT/rapidock_finetuned_best.pt"
P2_BEST="$P2_OUT/rapidock_finetuned_best.pt"

CONDA="conda run --no-capture-output -n rapidock"

require_ckpt() {
    local path="$1"; local phase="$2"
    if [ ! -f "$path" ]; then
        echo ""
        echo "========================================================================"
        echo "[chain-v2] FATAL: $phase did not produce a best checkpoint."
        echo "  Expected: $path"
        echo "========================================================================"
        exit 2
    fi
    echo "[chain-v2] $phase checkpoint: $path ($(du -h "$path" | cut -f1))"
}

# ─── Phase 1 (v2) ────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain-v2] Phase 1 — score heads (27.9%), lr=2e-5, warmup=5, 25 epochs"
echo "  Key change: lr 5× lower than v1 (1e-4→2e-5) to prevent score-field drift"
echo "  grad-clip=0.5 (tighter than default 1.0) for extra stability"
echo "========================================================================"
mkdir -p "$P1_OUT"
$CONDA python3 -u "$SCRIPT" \
    --train-csv          "$TRAIN_CSV" \
    --val-csv            "$VAL_CSV" \
    --checkpoint         "$PRETRAINED" \
    --output-dir         "$P1_OUT" \
    --unfreeze-phase     1 \
    --n-epochs           25 \
    --lr                 2e-5 \
    --warmup-epochs      5 \
    --lr-schedule        plateau \
    --grad-accum         4 \
    --grad-clip-norm     0.5 \
    --save-every         5 \
    --early-stop-patience 12 \
    --seed               42 \
    --esm-device         cuda \
    --bail-on-zero

require_ckpt "$P1_BEST" "Phase 1 (v2)"
echo "[chain-v2] Phase 1 complete."
sleep 5

# ─── Phase 2 (v2) ────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain-v2] Phase 2 — +cross_convs.2/3 + intra_convs.3, lr=5e-6, warmup=8, 45 epochs"
echo "  Key change: lr 4× lower than v1 (2e-5→5e-6) for deeper layer stability"
echo "  8-epoch warmup gives cross_conv layers time to orient before full LR"
echo "========================================================================"
mkdir -p "$P2_OUT"
$CONDA python3 -u "$SCRIPT" \
    --train-csv          "$TRAIN_CSV" \
    --val-csv            "$VAL_CSV" \
    --checkpoint         "$P1_BEST" \
    --output-dir         "$P2_OUT" \
    --unfreeze-phase     2 \
    --n-epochs           45 \
    --lr                 5e-6 \
    --warmup-epochs      8 \
    --lr-schedule        plateau \
    --grad-accum         4 \
    --grad-clip-norm     1.0 \
    --save-every         5 \
    --early-stop-patience 15 \
    --seed               42 \
    --esm-device         cuda \
    --bail-on-zero

require_ckpt "$P2_BEST" "Phase 2 (v2)"
echo "[chain-v2] Phase 2 complete."
sleep 5

# ─── Phase 3 (v2) ────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[chain-v2] Phase 3 — full retrain (7.5M), lr=1e-5→1e-7 cosine, layerwise decay"
echo "  Layer-wise LR: heads 1.0×, late_convs 0.5×, middle 0.2×, early 0.05×"
echo "  Peak lr=1e-5 (half of v1) — gentler global alignment, not full relearning"
echo "========================================================================"
mkdir -p "$P3_OUT"
$CONDA python3 -u "$SCRIPT" \
    --train-csv            "$TRAIN_CSV" \
    --val-csv              "$VAL_CSV" \
    --checkpoint           "$P2_BEST" \
    --output-dir           "$P3_OUT" \
    --unfreeze-phase       3 \
    --n-epochs             100 \
    --lr                   1e-5 \
    --warmup-epochs        10 \
    --lr-schedule          cosine \
    --cosine-min-lr        1e-7 \
    --grad-accum           4 \
    --grad-clip-norm       1.0 \
    --layerwise-lr-decay \
    --save-every           10 \
    --early-stop-patience  20 \
    --seed                 42 \
    --esm-device           cuda \
    --bail-on-zero

echo ""
echo "========================================================================"
echo "[chain-v2] ALL 3 PHASES (v2) COMPLETE."
echo "  Phase 1 best: $P1_BEST"
echo "  Phase 2 best: $P2_BEST"
echo "  Phase 3 out:  $P3_OUT"
echo "========================================================================"
