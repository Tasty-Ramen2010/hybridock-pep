#!/usr/bin/env bash
# chain_training_v5c.sh — V5C: ultra-minimal diversity-preserving recalibration
#
# PRIMARY goal: preservation of pretrained exploration diversity and score-field
# geometry.  The evidence from v1–v4c shows the pretrained RAPiDock manifold is
# extraordinarily well-calibrated and easy to damage.  V5C does almost nothing.
#
# Core findings driving this design:
#   - pretrained outperforms ALL finetuned on 7/8 benchmark complexes
#   - diversity collapses from 0.71 → 0.17–0.35 with any finetuning
#   - long peptides depend on broad exploration; even 0.08× cross_conv touch (v4c)
#     may compress the manifold
#   - useful adaptation: first few epochs only; subsequent epochs plateau/regress
#   - instability from repeated optimisation energy injection (cosine rebounds)
#
# V5C vs v3c/v4c:
#   - Phase 1: ONLY score heads (v3c/v4c had output convs in P1 too)
#   - Phase 2: + output convs at 0.5× (NO cross_conv update in ANY phase ever)
#   - LR ~2.5× lower than v3c: 2e-6→2e-7 (P1), 5e-7→5e-8 (P2)
#   - Slower EMA: 0.99998 (vs 0.99997)
#   - Tighter grad_clip: 0.2 (vs 0.25)
#   - Weaker pretrained-reg: λ=1e-4 (P1), λ=2e-4 (P2)  — "weak-to-moderate"
#   - Save every epoch after ep4 for stable-checkpoint averaging
#
# Phase 1: ONLY final score heads (tr/rot/tor_bb/tor_sc final_layer)
#   - LR: 2e-6 → 2e-7 (exponential, warmup=12, epochs=10)
#   - EMA=0.99998, grad_clip=0.2, weight_decay=2e-5
#   - Single-tier optimizer (1.0× — all heads treated equally)
#   - Pretrained-reg (lambda=1e-4) on the score heads themselves
#
# Phase 2: + output convolutions at 0.5× LR
#   - LR: 5e-7 → 5e-8 (exponential, warmup=10, epochs=12)
#   - EMA=0.99998, grad_clip=0.2, weight_decay=3e-5
#   - 2-tier diff LR: output_convs=0.5×, heads=1.0×
#   - Pretrained-reg (lambda=2e-4) on score heads + output_convs
#   - Save every epoch from ep4 (stable checkpoint for averaging)
#
# Resumable: re-running this script is safe (--resume + phase skip if final.pt exists).
#
# Usage:
#   nohup bash scripts/chain_training_v5c.sh > logs/chain_training_v5c.log 2>&1 &
#   tail -f logs/chain_training_v5c.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2"

CONDA_ENV="rapidock"

echo "[v5c-chain] Repo:       $REPO"
echo "[v5c-chain] Pretrained: $PRETRAINED_CKPT"
echo "[v5c-chain] Philosophy: ultra-minimal diversity-preserving recalibration, NO cross_conv ever"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v5c-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: ONLY final score heads
#   Monotone exponential: 2e-6 → 2e-7; warmup=3; epochs=10
#   Single-tier optimizer: all heads at 1.0× (no diff LR needed)
#   Pretrained-reg: lambda=1e-4 (weak) on score head patterns
#   Rationale: smallest possible adaptation — just score magnitude calibration.
#              Output convs, cross_convs, intra_convs: completely frozen.
#   NOTE: warmup was originally 12 (> n_epochs=10) — bug fixed to 3.
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v5c-chain] Phase 1: ONLY final score heads  (epochs 1-10)"
echo "           Monotone exponential 2e-6→2e-7, warmup=3"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5c-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v5c-mode \
        --n-epochs       10 \
        --lr             2e-6 \
        --warmup-epochs  3 \
        --lr-schedule    exponential \
        --cosine-min-lr  2e-7 \
        --grad-clip-norm 0.2 \
        --weight-decay   2e-5 \
        --ema-decay      0.99998 \
        --pretrained-reg-lambda 1e-4 \
        --pretrained-reg-patterns \
            tr_final_layer rot_final_layer tor_bb_final_layer tor_sc_final_layer \
        --grad-accum     4 \
        --save-every     1 \
        --save-every-after 4 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: + output convolutions at 0.5× LR
#   Monotone exponential: 5e-7 → 5e-8; warmup=2; epochs=12
#   NOTE: warmup was originally 10 (leaving only 2 post-warmup epochs) — bug fixed to 2.
#   2-tier diff LR: output_convs=0.5×, heads=1.0×
#   Pretrained-reg: lambda=2e-4 on score heads + output_convs
#   Save every epoch from ep4 — stable checkpoints for post-hoc averaging
#   Rationale: minimal geometry-adjacent update; no cross_conv touch ever.
#              0.5× for output_convs ≈ half-energy input vs score heads.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v5c-chain] Phase 2: + output convolutions at 0.5×  (epochs 1-12)"
echo "           Monotone exponential 5e-7→5e-8, warmup=2"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v5c-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v5c-mode \
        --n-epochs       12 \
        --lr             5e-7 \
        --warmup-epochs  2 \
        --lr-schedule    exponential \
        --cosine-min-lr  5e-8 \
        --grad-clip-norm 0.2 \
        --weight-decay   3e-5 \
        --ema-decay      0.99998 \
        --pretrained-reg-lambda 2e-4 \
        --pretrained-reg-patterns \
            tr_final_layer rot_final_layer tor_bb_final_layer tor_sc_final_layer \
            final_conv tor_bb_bond_conv tor_sc_bond_conv \
        --grad-accum     4 \
        --save-every     1 \
        --save-every-after 4 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P2_CKPT="$P2_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P2_CKPT" "p2"

echo ""
echo "=================================================================="
echo "[v5c-chain] ALL PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo ""
echo "  All P2 epoch checkpoints (save-every=1 from ep4):"
ls "$P2_OUT"/rapidock_finetuned_epoch*.pt 2>/dev/null | wc -l | xargs -I{} echo "    {} checkpoints"
echo ""
echo "  Recommended next steps:"
echo "  1. Benchmark v5c vs v3c/v4c and pretrained on all 8 benchmark complexes:"
echo "       python3 scripts/benchmark_inference_multi.py \\"
echo "           --benchmark-csv  data/inference_benchmark_set.csv \\"
echo "           --pretrained     $PRETRAINED_CKPT \\"
echo "           --finetuned      $P2_CKPT \\"
echo "           --also-compare \\"
echo "               $P1_CKPT \\"
echo "           --n-samples 20 --seed 42 \\"
echo "           --out-dir logs/benchmark_v5c"
echo ""
echo "  2. KEY metric: diversity_ratio — target ≥0.60 (vs pretrained ~0.71)"
echo "     If v5c drops diversity to <0.50, pure pretrained is still better."
echo "  3. Checkpoint averaging: avg best 3 P2 stable epochs:"
echo "       python3 -c \\"
echo "           \"import torch; ..."  # see ai_training_guide_peppc.md (Zenodo archive) §5
echo "=================================================================="
