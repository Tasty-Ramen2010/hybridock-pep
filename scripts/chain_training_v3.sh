#!/usr/bin/env bash
# chain_training_v3.sh — V3 controlled-specialization fine-tuning chain
#
# Design goal: improve receptor–peptide interaction specialization WITHOUT
# collapsing diversity or destabilizing the pretrained score field.
#
# Key differences from v1/v2/v2b:
#   - P1: cross_convs.2/3 only — intra_convs entirely frozen (preserves peptide physics)
#   - P2: intra_convs.3 added at 0.15× LR; 3-tier differential LR; EMA=0.9997
#   - Weak L2 regularization toward pretrained weights throughout P2/P3
#   - P3: full model minus ESM; peak LR=7e-6 (lower than v2's 1e-5); EMA=0.9999
#   - Checkpoints saved every epoch after epoch 20 for late-stage ensemble averaging
#   - Oscillation amplitude + score-norm variance logged every epoch
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash scripts/chain_training_v3.sh > logs/chain_training_v3.log 2>&1 &
#   tail -f logs/chain_training_v3.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3_phase3"

CONDA_ENV="rapidock"

echo "[v3-chain] Repo:       $REPO"
echo "[v3-chain] Script:     $TRAIN_SCRIPT"
echo "[v3-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v3-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + cross_convs.2/3 ONLY
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v3-chain] Phase 1: cross_convs.2/3 + score heads  (epochs 1-20)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v3-mode \
        --n-epochs       20 \
        --lr             2e-5 \
        --warmup-epochs  5 \
        --lr-schedule    plateau \
        --grad-clip-norm 1.0 \
        --weight-decay   1e-6 \
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
# Phase 2: + intra_convs.3 at 0.15× LR; 3-tier differential LR
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v3-chain] Phase 2: + intra_convs.3 (0.15×); diff LR  (epochs 1-40)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v3-mode \
        --n-epochs       40 \
        --lr             5e-6 \
        --warmup-epochs  6 \
        --lr-schedule    plateau \
        --grad-clip-norm 1.0 \
        --weight-decay   1e-5 \
        --ema-decay      0.9997 \
        --pretrained-reg-lambda 2e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 rec_node_embedding pep_node_embedding \
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
# Phase 3: full model except ESM; cosine LR; layerwise decay
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v3-chain] Phase 3: full model (ESM frozen); cosine  (epochs 1-80)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v3-mode \
        --n-epochs       80 \
        --lr             7e-6 \
        --warmup-epochs  10 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-7 \
        --layerwise-lr-decay \
        --grad-clip-norm 1.0 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --pretrained-reg-lambda 1e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 rec_node_embedding pep_node_embedding \
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
echo "[v3-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late-epoch checkpoints (every epoch from ep20):"
echo "  $P3_OUT/rapidock_finetuned_epoch*.pt"
echo ""
echo "  Next steps:"
echo "    1. Run benchmark: python3 scripts/benchmark_inference_multi.py --model-label v3 ..."
echo "    2. Compare val losses vs v1/v2/v2b in training_stats.xlsx"
echo "=================================================================="
