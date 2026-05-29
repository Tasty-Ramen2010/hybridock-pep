#!/usr/bin/env bash
# chain_training_v5.sh — V5 ultra-conservative adaptation
#
# Design goal: test whether MINIMAL specialization with MAXIMAL preservation
# of pretrained diffusion behavior produces the best broad generalization.
#
# Architecture:
#   P1: score heads ONLY (27.9% params) — calibrate outputs, touch nothing else
#   P2: + cross_convs.3 only (41.8% params) — single outermost receptor ring
#   P3: full except ESM; AGGRESSIVE layerwise LR (0.30/0.10/0.02 vs 0.50/0.20/0.05)
#
# Key conservative choices:
#   - Lower LR throughout: P1=1e-5, P2=2e-6, P3 peak=5e-6
#   - Stronger grad_clip=0.5 (vs 1.0 in v3/v4)
#   - Longer warmup: P1=8, P2=8, P3=15
#   - EMA=0.9997→0.9998→0.9999 (slower EMA integration)
#   - Save every epoch after warmup for late-checkpoint ensemble
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash scripts/chain_training_v5.sh > logs/chain_training_v5.log 2>&1 &
#   tail -f logs/chain_training_v5.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5_phase3"

CONDA_ENV="rapidock"

echo "[v5-chain] Repo:       $REPO"
echo "[v5-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v5-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads ONLY (27.9% params)
#   Ultra-conservative: only output calibration, no receptor-interaction layer
#   LR=1e-5 (vs v3/v4's 2e-5); tighter grad_clip=0.5; EMA=0.9997
#   Longer warmup=8 to prevent head LR overshoot
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v5-chain] Phase 1: score heads ONLY  (epochs 1-20)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v5-mode \
        --n-epochs       20 \
        --lr             1e-5 \
        --warmup-epochs  8 \
        --lr-schedule    plateau \
        --grad-clip-norm 0.5 \
        --ema-decay      0.9997 \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 8 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: score heads + cross_convs.3 ONLY (41.8% params)
#   Adds ONE outermost cross-conv ring; no differential LR (single uniform rate)
#   LR=2e-6; warmup=8; grad_clip=0.5; EMA=0.9998
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v5-chain] Phase 2: heads + cross_convs.3  (epochs 1-30)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v5-mode \
        --n-epochs       30 \
        --lr             2e-6 \
        --warmup-epochs  8 \
        --lr-schedule    plateau \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-6 \
        --ema-decay      0.9998 \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 8 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P2_CKPT="$P2_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P2_CKPT" "p2"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full model except ESM; AGGRESSIVE layerwise LR decay
#   Peak=5e-6 (lower than v3/v4's 7e-6); final=1e-7; warmup=15 (longer)
#   Aggressive multipliers: late=0.30×, mid=0.10×, early=0.02×
#   EMA=0.9999; save every epoch from ep15 (after warmup)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v5-chain] Phase 3: full model (ESM frozen); aggressive layerwise  (epochs 1-60)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v5-mode \
        --n-epochs       60 \
        --lr             5e-6 \
        --warmup-epochs  15 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-7 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --grad-accum     4 \
        --save-every     10 \
        --save-every-after 15 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P3_CKPT="$P3_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P3_CKPT" "p3"

echo ""
echo "=================================================================="
echo "[v5-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late checkpoints (every epoch from warmup): $P3_OUT/rapidock_finetuned_epoch*.pt"
echo ""
echo "  Benchmark multiple late-stable checkpoints for robustness:"
echo "  python3 scripts/benchmark_inference_multi.py --model-label v5 ..."
echo "=================================================================="
