#!/usr/bin/env bash
# chain_training_v6.sh — V6 training: gap-fill specialization (long/very_long)
#
# Three-phase progressive training from rapidock_global.pt:
#   P1 (ep 1-8):  tor_bb_bond_conv + score heads; lr=3e-6; grad_clip=0.5
#   P2 (ep 9-35): +cross_convs; lr=5e-6→5e-7 cosine; tier oversampling; L2 reg λ=3e-4
#   P3 (ep 36-45): uniform 1×; lr=5e-7; polish
#
# Usage: bash scripts/chain_training_v6.sh [phase]
#   phase = 1 (default), 2, or 3
#   Phases run sequentially within this script if called without arg.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"
CONDA_ENV="rapidock"
TRAIN_CSV="$REPO/data/v6_train_combined.csv"
VAL_CSV="$REPO/data/v6_val_200.csv"
PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt"
OUT_BASE="$REPO/logs/v6_run"

mkdir -p "$OUT_BASE/phase1" "$OUT_BASE/phase2" "$OUT_BASE/phase3"

START_PHASE="${1:-1}"

echo "=================================================================="
echo " V6 Training Chain — starting from Phase $START_PHASE"
echo " Train CSV:   $TRAIN_CSV"
echo " Val CSV:     $VAL_CSV"
echo " Pretrained:  $PRETRAINED"
echo " Output base: $OUT_BASE"
echo " $(date)"
echo "=================================================================="

# ── Phase 1: Torsion + Score Heads (8 epochs) ─────────────────────────
if [ "$START_PHASE" -le 1 ]; then
  echo ""
  echo "=== PHASE 1 (epochs 1-8): tor_bb_bond_conv + score heads ==="
  echo "    lr=3e-6 cosine | grad_clip=0.5 | 980k trainable"
  echo "    $(date)"
  conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
    --v6-mode \
    --train-csv "$TRAIN_CSV" \
    --v6-val-csv "$VAL_CSV" \
    --checkpoint "$PRETRAINED" \
    --output-dir "$OUT_BASE/phase1" \
    --unfreeze-phase 1 \
    --n-epochs 8 \
    --lr 3e-6 \
    --lr-schedule cosine \
    --cosine-min-lr 5e-7 \
    --warmup-epochs 2 \
    --grad-clip-norm 0.5 \
    --grad-accum 4 \
    --save-every 2 \
    --save-every-after 5 \
    --esm-device cpu \
    --v6-guard-patience 3 \
    --v6-guard-threshold 0.3 \
    2>&1 | tee "$OUT_BASE/phase1.log"
  echo "=== Phase 1 complete $(date) ==="
fi

# ── Phase 2: Cross-Conv Specialization (27 epochs) ────────────────────
if [ "$START_PHASE" -le 2 ]; then
  P1_CKPT="$OUT_BASE/phase1/rapidock_finetuned_final.pt"
  if [ ! -f "$P1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $P1_CKPT"
    exit 1
  fi
  echo ""
  echo "=== PHASE 2 (epochs 9-35): +cross_convs, tier oversampling ==="
  echo "    lr=5e-6→5e-7 cosine | grad_clip=1.0 | 3.65M trainable | λ=3e-4"
  echo "    $(date)"
  conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
    --v6-mode \
    --train-csv "$TRAIN_CSV" \
    --v6-val-csv "$VAL_CSV" \
    --checkpoint "$P1_CKPT" \
    --output-dir "$OUT_BASE/phase2" \
    --unfreeze-phase 2 \
    --n-epochs 27 \
    --lr 5e-6 \
    --lr-schedule cosine \
    --cosine-min-lr 5e-7 \
    --warmup-epochs 3 \
    --grad-clip-norm 1.0 \
    --grad-accum 4 \
    --save-every 5 \
    --save-every-after 15 \
    --pretrained-reg-lambda 3e-4 \
    --esm-device cpu \
    --v6-guard-patience 3 \
    --v6-guard-threshold 0.3 \
    2>&1 | tee "$OUT_BASE/phase2.log"
  echo "=== Phase 2 complete $(date) ==="
fi

# ── Phase 3: Fine-Polish (10 epochs) ──────────────────────────────────
if [ "$START_PHASE" -le 3 ]; then
  # Prefer best_combined from P2; fall back to final
  P2_CKPT="$OUT_BASE/phase2/rapidock_finetuned_best_combined.pt"
  if [ ! -f "$P2_CKPT" ]; then
    P2_CKPT="$OUT_BASE/phase2/rapidock_finetuned_final.pt"
  fi
  if [ ! -f "$P2_CKPT" ]; then
    echo "ERROR: Phase 2 checkpoint not found"
    exit 1
  fi
  echo ""
  echo "=== PHASE 3 (epochs 36-45): fine-polish, uniform 1× ==="
  echo "    lr=5e-7 flat | grad_clip=1.0 | λ=3e-4"
  echo "    $(date)"
  conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
    --v6-mode \
    --train-csv "$TRAIN_CSV" \
    --v6-val-csv "$VAL_CSV" \
    --checkpoint "$P2_CKPT" \
    --output-dir "$OUT_BASE/phase3" \
    --unfreeze-phase 3 \
    --n-epochs 10 \
    --lr 5e-7 \
    --lr-schedule cosine \
    --cosine-min-lr 1e-7 \
    --warmup-epochs 0 \
    --grad-clip-norm 1.0 \
    --grad-accum 4 \
    --save-every 2 \
    --save-every-after 1 \
    --pretrained-reg-lambda 3e-4 \
    --esm-device cpu \
    --v6-guard-patience 3 \
    --v6-guard-threshold 0.3 \
    2>&1 | tee "$OUT_BASE/phase3.log"
  echo "=== Phase 3 complete $(date) ==="
fi

echo ""
echo "=================================================================="
echo " V6 TRAINING COMPLETE"
echo " Final model:  $OUT_BASE/phase3/rapidock_finetuned_final.pt"
echo " Best vl:      $OUT_BASE/phase2/rapidock_finetuned_best_very_long.pt"
echo " Best comb:    $OUT_BASE/phase2/rapidock_finetuned_best_combined.pt"
echo " $(date)"
echo "=================================================================="
