#!/usr/bin/env bash
# chain_training_v4new.sh — V4N (v4 new) careful mechanistic probe
#
# Design goal: mechanistic probe of cross_conv adaptation limits with a more
# conservative hyperparameter envelope than v3b.  Identical cosine+adaptive-spike
# stability stack as v3b, but slower LR and tighter differential-LR tiers.
#
# Differences vs v3b:
#   P1: LR=8e-6 (vs 1e-5), 18 epochs (vs 20)
#   P2: LR=2e-6 (vs 3e-6), cross_convs.2=0.50× (vs 0.60×), 30 epochs (vs 35)
#   P3: LR=5e-6, layerwise 0.40/0.15/0.03 (vs standard 0.50/0.20/0.05), 60 epochs (vs 65)
#
# Shares with v3b:
#   - COSINE schedule ALL phases (no plateau)
#   - grad_clip=0.5
#   - EMA=0.9999 from P1
#   - Adaptive spike LR (auto-halves for 2 epochs on val tr_norm spike >10×)
#   - P1 unfreeze: score heads + cross_convs.3 ONLY
#   - P2 unfreeze: + cross_convs.2 (at 0.50×, tighter than v3b's 0.60×)
#   - P3: full except ESM; pretrained-reg on intra_convs + early cross_convs
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash scripts/chain_training_v4new.sh > logs/chain_training_v4new.log 2>&1 &
#   tail -f logs/chain_training_v4new.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4n_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4n_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4n_phase3"

CONDA_ENV="rapidock"

echo "[v4n-chain] Repo:       $REPO"
echo "[v4n-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v4n-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + cross_convs.3 ONLY (same param set as V5-P2 / v3b-P1)
#   Cosine: 8e-6 peak → 2e-6 floor; warmup=6; EMA=0.9999; grad_clip=0.5
#   Adaptive spike LR: auto-halves LR 2 epochs on norm spike
#   Rationale: slower entry LR than v3b (8e-6 vs 1e-5) for extra conservatism
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v4n-chain] Phase 1: score heads + cross_convs.3 ONLY  (epochs 1-18)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4n-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v4n-mode \
        --n-epochs       18 \
        --lr             8e-6 \
        --warmup-epochs  6 \
        --lr-schedule    cosine \
        --cosine-min-lr  2e-6 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-6 \
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
# Phase 2: adds cross_convs.2 at 0.50× differential LR; L2 pretrained-reg
#   Cosine: 2e-6 peak → 5e-7 floor; warmup=6; grad_clip=0.5; EMA=0.9999
#   2-tier diff LR: cross_convs.2=0.50×, everything else=1.0×
#   Pretrained-reg: intra_convs, early equivariant layers
#   Rationale: tighter 0.50× (vs v3b's 0.60×) = more conservative CC.2 entry
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v4n-chain] Phase 2: + cross_convs.2 (0.50×); pretrained-reg  (epochs 1-30)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4n-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v4n-mode \
        --n-epochs       30 \
        --lr             2e-6 \
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
# Phase 3: full model except ESM; careful layerwise 0.40/0.15/0.03
#   Peak=5e-6; floor=1e-7; warmup=10; epochs=60; grad_clip=0.5
#   Layerwise: 0.40 (late convs) / 0.15 (middle) / 0.03 (early) — tighter than
#              standard 0.50/0.20/0.05; looser than v5's 0.30/0.10/0.02
#   Save every epoch from ep15 for late-stage ensemble averaging
#   Pretrained-reg: intra_convs + cross_convs.0/1
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v4n-chain] Phase 3: full model (ESM frozen); careful layerwise  (epochs 1-60)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4n-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v4n-mode \
        --n-epochs       60 \
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
echo "[v4n-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late checkpoints (every epoch from ep15): $P3_OUT/rapidock_finetuned_epoch*.pt"
echo ""
echo "  Benchmark multiple late-stable checkpoints for robustness:"
echo "  python3 scripts/benchmark_inference_multi.py --model-label v4n ..."
echo "=================================================================="
