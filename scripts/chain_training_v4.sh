#!/usr/bin/env bash
# chain_training_v4.sh — V4 pure cross_conv receptor-interaction adaptation
#
# Design goal: test whether targeting cross_convs.2/3 WITHOUT touching any
# intra_convs (and without pretrained-weight L2 reg) improves receptor
# specialization while preserving peptide-physics priors.
#
# Key differences vs v3:
#   - P1 AND P2: same unfreeze pattern (score heads + cross_convs.2/3 ONLY)
#     → intra_convs entirely frozen throughout P1 and P2
#   - P2: 2-tier differential LR (cross_convs.2=0.7×, rest=1.0×)
#     → auto-2-tier: intra_convs.3 frozen → its optimizer tier is empty
#   - No pretrained-weight L2 regularization
#   - WD=1e-6 in P2 (lighter than v3's 1e-5)
#   - P3: same as v3 (cosine, 7e-6→1e-7, ESM frozen, layerwise 0.5/0.2/0.05)
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash scripts/chain_training_v4.sh > logs/chain_training_v4.log 2>&1 &
#   tail -f logs/chain_training_v4.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4_phase3"

CONDA_ENV="rapidock"

echo "[v4-chain] Repo:       $REPO"
echo "[v4-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v4-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + cross_convs.2/3 (50.5% params)
#   Same as V3-P1; intra_convs entirely frozen
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v4-chain] Phase 1: cross_convs.2/3 + score heads  (epochs 1-20)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v4-mode \
        --n-epochs       20 \
        --lr             2e-5 \
        --warmup-epochs  5 \
        --lr-schedule    plateau \
        --grad-clip-norm 1.0 \
        --ema-decay      0.9995 \
        --grad-accum     4 \
        --save-every     5 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: SAME unfreeze pattern as P1; 2-tier differential LR
#   cross_convs.3=1.0× (5e-6), cross_convs.2=0.7× (3.5e-6)
#   No pretrained-reg; lighter WD than v3
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v4-chain] Phase 2: cross_convs.2/3 + 2-tier diff LR  (epochs 1-40)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v4-mode \
        --n-epochs       40 \
        --lr             5e-6 \
        --warmup-epochs  6 \
        --lr-schedule    plateau \
        --grad-clip-norm 1.0 \
        --weight-decay   1e-6 \
        --ema-decay      0.9997 \
        --grad-accum     4 \
        --save-every     5 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P2_CKPT="$P2_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P2_CKPT" "p2"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full model except ESM; cosine; standard layerwise decay
#   Identical to v3 P3 except no pretrained-reg
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v4-chain] Phase 3: full model (ESM frozen); cosine  (epochs 1-80)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v4-mode \
        --n-epochs       80 \
        --lr             7e-6 \
        --warmup-epochs  10 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-7 \
        --layerwise-lr-decay \
        --grad-clip-norm 1.0 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --grad-accum     4 \
        --save-every     10 \
        --save-every-after 20 \
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
echo "[v4-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late checkpoints (every epoch from ep20): $P3_OUT/rapidock_finetuned_epoch*.pt"
echo "=================================================================="
