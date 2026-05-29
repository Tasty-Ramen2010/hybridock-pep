#!/usr/bin/env bash
# chain_training_v5new.sh — V5N (v5 new) ultra-conservative manifold preservation
#
# Design goal: test whether the pretrained RAPiDock diffusion prior is already
# near-optimal, and that only extremely gentle adaptation is needed to specialize
# to PepPC receptors without reducing exploration diversity.
#
# Core philosophy: preserve pretrained manifold, apply only minimal controlled biasing.
#
# Architecture differences vs v4n and v3b:
#   P1: score heads + output convs ONLY (even more conservative than all prior P1s)
#       Excludes center_edge_embedding, pep_a_node_embedding, final_edge_embedding
#   P2: + cross_convs.3 at uniform LR (same conv set as v3b P1)
#   P3: full except ESM; ultra-conservative layerwise 1.0/0.25/0.08/0.02
#       (far tighter than v4n's 0.40/0.15/0.03 and v3b's 0.50/0.20/0.05)
#
# Shared with v3b/v4n:
#   - Cosine LR ALL phases (no plateau)
#   - grad_clip=0.5
#   - EMA decay=0.9999 from P1
#   - Adaptive spike LR (auto-halves for 2 epochs on val tr_norm spike >10×)
#
# v5n-specific:
#   - EMA skip on spike: pause EMA updates for 2 epochs after spike detection
#     so that unstable spike-epoch weights don't leak into EMA checkpoint
#
# Training schedule (most conservative ever attempted):
#   P1: LR=5e-6, floor=1e-6, warmup=8, epochs=18, wd=1e-6
#   P2: LR=1e-6, floor=2e-7, warmup=8, epochs=25, wd=1e-5
#   P3: LR=3e-6, floor=1e-7, warmup=12, epochs=55, wd=1e-5
#
# Resumable: re-running this script is safe.
#   - Each phase is skipped if its final checkpoint already exists.
#   - Within a phase, --resume loads the latest epoch checkpoint and continues.
#
# Usage:
#   nohup bash scripts/chain_training_v5new.sh > logs/chain_training_v5new.log 2>&1 &
#   tail -f logs/chain_training_v5new.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5n_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5n_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5n_phase3"

CONDA_ENV="rapidock"

echo "[v5n-chain] Repo:       $REPO"
echo "[v5n-chain] Pretrained: $PRETRAINED_CKPT"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v5n-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + output convolution layers ONLY
#   Most conservative P1 of all experiments — no embedding layers, no cross_convs
#   Cosine: 5e-6 peak → 1e-6 floor; warmup=8; EMA=0.9999; grad_clip=0.5
#   Adaptive spike LR + EMA skip (v5n-specific)
#   Rationale: calibrate score output magnitudes to PepPC scale without
#   touching receptor-peptide interaction layers at all
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v5n-chain] Phase 1: score heads + output convs ONLY  (epochs 1-18)"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5n-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v5n-mode \
        --n-epochs       18 \
        --lr             5e-6 \
        --warmup-epochs  8 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-6 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-6 \
        --ema-decay      0.9999 \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 12 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: adds cross_convs.3 at uniform LR; L2 pretrained-reg
#   Cosine: 1e-6 peak → 2e-7 floor; warmup=8; grad_clip=0.5; EMA=0.9999
#   Uniform LR for all unfrozen params (no differential — too conservative to need it)
#   Pretrained-reg: anchors deeper layers to pretrained prior
#   Rationale: single outermost cross-conv ring at very gentle LR (1e-6 vs v3b's 3e-6)
#   provides minimal receptor recognition adaptation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v5n-chain] Phase 2: + cross_convs.3; pretrained-reg  (epochs 1-25)"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5n-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v5n-mode \
        --n-epochs       25 \
        --lr             1e-6 \
        --warmup-epochs  8 \
        --lr-schedule    cosine \
        --cosine-min-lr  2e-7 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --pretrained-reg-lambda 2e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 cross_convs.2 rec_node_embedding pep_node_embedding \
        --grad-accum     4 \
        --save-every     5 \
        --save-every-after 12 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P2_CKPT="$P2_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P2_CKPT" "p2"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full model except ESM; ultra-conservative layerwise 1.0/0.25/0.08/0.02
#   Peak=3e-6; floor=1e-7; warmup=12; epochs=55; grad_clip=0.5
#   Pretrained-reg: intra_convs + cross_convs.0/1 (keep deep geometry near prior)
#   Save every epoch from ep12 — many checkpoints for ensemble averaging
#   Rationale: the 1.0/0.25/0.08/0.02 layerwise tiers ensure that even in P3,
#   early equivariant layers (02×) barely move relative to pretrained init
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v5n-chain] Phase 3: full model (ESM frozen); ultra-conservative layerwise  (epochs 1-55)"
echo "=================================================================="

if [[ -f "$P3_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5n-chain] Phase 3: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P2_CKPT" \
        --output-dir     "$P3_OUT" \
        --unfreeze-phase 3 \
        --v5n-mode \
        --n-epochs       55 \
        --lr             3e-6 \
        --warmup-epochs  12 \
        --lr-schedule    cosine \
        --cosine-min-lr  1e-7 \
        --grad-clip-norm 0.5 \
        --weight-decay   1e-5 \
        --ema-decay      0.9999 \
        --pretrained-reg-lambda 1e-4 \
        --pretrained-reg-patterns intra_convs cross_convs.0 cross_convs.1 rec_node_embedding pep_node_embedding \
        --grad-accum     4 \
        --save-every     10 \
        --save-every-after 12 \
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
echo "[v5n-chain] ALL 3 PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "  P3 best: $P3_CKPT"
echo ""
echo "  Late checkpoints (every epoch from ep12): $P3_OUT/rapidock_finetuned_epoch*.pt"
echo ""
echo "  Benchmark multiple late-stable checkpoints for ensemble averaging:"
echo "  python3 scripts/benchmark_inference_multi.py --model-label v5n ..."
echo ""
echo "  Compare v3b/v4n/v5n to test manifold-preservation hypothesis:"
echo "  python3 scripts/analyze_training.py \\"
echo "    --phase-dirs $P3_OUT \\"
echo "    --compare-dirs third_party/RAPiDock_finetuned/finetune_peppc_v3b_phase3 \\"
echo "                   third_party/RAPiDock_finetuned/finetune_peppc_v4n_phase3 \\"
echo "    --out-dir logs/analysis_v3b_v4n_v5n"
echo "=================================================================="
