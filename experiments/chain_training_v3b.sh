#!/usr/bin/env bash
# chain_training_v3b.sh — V3B stable controlled specialization
#
# Design goal: fix v3's fatal oscillation instability while preserving its
# conceptual approach (progressive cross-conv unfreezing + representation reg).
#
# Root causes fixed vs v3:
#   - COSINE schedule in ALL phases (plateau caused undamped LR-fixed oscillations)
#   - Lower LR throughout: P1=1e-5, P2=3e-6, P3 peak=5e-6
#   - Tighter grad_clip=0.5 (vs v3's 1.0) — protects cross_convs from spike damage
#   - Slower EMA=0.9999 from Phase 1 (vs v3's 0.9995 P1 / 0.9997 P2)
#   - Adaptive spike LR: auto-halves LR for 2 epochs on norm spike detection
#
# Architecture differences vs v3:
#   P1: cross_convs.3 ONLY (vs v3's cross_convs.2+3) — one ring at a time
#   P2: + cross_convs.2 at 0.6× differential LR (vs v3's 0.70×, + intra_convs.3)
#       intra_convs entirely frozen in P1 and P2 (same as v4, not v3)
#   P3: full except ESM; standard layerwise 0.5/0.2/0.05 (same as v3)
#       pretrained-weight L2 reg: intra_convs + cross_convs.0/1
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash experiments/chain_training_v3b.sh > logs/chain_training_v3b.log 2>&1 &
#   tail -f logs/chain_training_v3b.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3b_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3b_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3b_phase3"

CONDA_ENV="rapidock"

echo "[v3b-chain] Repo:       $REPO"
echo "[v3b-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v3b-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + cross_convs.3 ONLY (same param set as V5-P2)
#   Cosine: 1e-5 peak → 2e-6 floor; warmup=6; EMA=0.9999; grad_clip=0.5
#   Adaptive spike LR: auto-halves LR 2 epochs on norm spike
#   Rationale: start with the single outermost receptor ring; no intra_convs
#   drift; build a stable score-head calibration before widening the unfreeze
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v3b-chain] Phase 1: score heads + cross_convs.3 ONLY  (epochs 1-20)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3b-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v3b-mode \
        --n-epochs       20 \
        --lr             1e-5 \
        --warmup-epochs  6 \
        --lr-schedule    cosine \
        --cosine-min-lr  2e-6 \
        --grad-clip-norm 0.5 \
        --ema-decay      0.9999 \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 15 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: adds cross_convs.2 at 0.6× differential LR; L2 pretrained-reg
#   Cosine: 3e-6 peak → 5e-7 floor; warmup=6; grad_clip=0.5; EMA=0.9999
#   2-tier diff LR: cross_convs.2=0.6×, everything else=1.0×
#   Pretrained-reg: intra_convs, early equivariant layers
#   Rationale: add the 2nd outermost cross-conv ring conservatively; weak L2
#   anchors deep intra_convs (unfrozen from P3 on) to pretrained prior
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v3b-chain] Phase 2: + cross_convs.2 (0.6×); pretrained-reg  (epochs 1-35)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3b-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v3b-mode \
        --n-epochs       35 \
        --lr             3e-6 \
        --warmup-epochs  6 \
        --lr-schedule    cosine \
        --cosine-min-lr  5e-7 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --pretrained-reg-lambda 2e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 rec_node_embedding pep_node_embedding \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 15 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P2_CKPT="$P2_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P2_CKPT" "p2"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full model except ESM; cosine; standard layerwise 0.5/0.2/0.05
#   Peak=5e-6; floor=1e-7; warmup=10; epochs=65; grad_clip=0.5
#   Save every epoch from ep15 for late-stage ensemble averaging
#   Pretrained-reg: intra_convs + cross_convs.0/1 (keep deep layers near prior)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v3b-chain] Phase 3: full model (ESM frozen); layerwise  (epochs 1-65)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3b-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v3b-mode \
        --n-epochs       65 \
        --lr             5e-6 \
        --warmup-epochs  10 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-7 \
        --layerwise-lr-decay \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --pretrained-reg-lambda 1e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 rec_node_embedding pep_node_embedding \
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
echo "[v3b-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late checkpoints (every epoch from ep15): $P3_OUT/rapidock_finetuned_epoch*.pt"
echo ""
echo "  Benchmark multiple late-stable checkpoints for robustness:"
echo "  python3 scripts/benchmark_inference_multi.py --model-label v3b ..."
echo "=================================================================="
