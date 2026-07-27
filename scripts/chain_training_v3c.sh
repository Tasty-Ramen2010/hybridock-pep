#!/usr/bin/env bash
# chain_training_v3c.sh — V3C: minimal low-energy recalibration
#
# Core philosophy: the pretrained RAPiDock diffusion manifold is near-optimal.
# All adaptation energy beyond score-head calibration risks narrowing diversity.
#
# Key differences vs all prior runs:
#   - MONOTONE exponential decay only (WarmupThenExponential) — LR never increases
#   - Spike events permanently reduce LR (no cooldown-then-resume)
#   - Phase 1: score heads + output convs ONLY  (no cross_convs at all)
#   - Phase 2: + cross_convs.3 at 0.1× LR (minimal receptor recognition shift)
#   - NO Phase 3 — 100% of equivariant geometry priors preserved
#   - Strong pretrained-reg on all structural layers
#
# Design choices from v3b analysis (May 2026):
#   - v3b cosine repeatedly re-injected instability as LR climbed back up
#   - Stable operating window: ~3e-6 to 5e-6
#   - Useful adaptation: ep4–9 only
#   - Model RECOVERS after spikes → manifold intact; just need to stop disturbing it
#
# Phase 1: score heads + output convs ONLY
#   - LR: 4e-6 → 8e-7 (exponential, warmup=10, epochs=14)
#   - EMA=0.99997, grad_clip=0.25, weight_decay=1e-5
#   - 2-tier diff LR: output_convs=0.5×, heads=1.0×
#   - Pretrained-reg (lambda=3e-4) on equivariant + intra_convs + cc.0/1/2
#
# Phase 2: + cross_convs.3 at 0.1× LR
#   - LR: 1e-6 → 1e-7 (exponential, warmup=8, epochs=16)
#   - EMA=0.99997, grad_clip=0.25, weight_decay=2e-5
#   - 3-tier diff LR: cc.3=0.1×, output_convs=0.5×, heads=1.0×
#   - Pretrained-reg (lambda=5e-4) on equivariant + intra_convs + cc.0/1/2
#   - Save every epoch (all checkpoints; benchmark for best late-stable)
#
# Resumable: re-running this script is safe (--resume + phase skip if final.pt exists).
#
# Usage:
#   nohup bash scripts/chain_training_v3c.sh > logs/chain_training_v3c.log 2>&1 &
#   tail -f logs/chain_training_v3c.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v3c_phase2"

CONDA_ENV="rapidock"

echo "[v3c-chain] Repo:       $REPO"
echo "[v3c-chain] Pretrained: $PRETRAINED_CKPT"
echo "[v3c-chain] Philosophy: minimal low-energy recalibration, monotone LR, NO phase 3"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v3c-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + output convs ONLY
#   Monotone exponential: 4e-6 → 8e-7; warmup=10; epochs=14
#   2-tier diff LR: output_convs=0.5×, heads=1.0×
#   Strong pretrained-reg: lambda=3e-4 on all structural layers
#   Rationale: calibrate score magnitude without touching any geometry
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v3c-chain] Phase 1: score heads + output convs ONLY  (epochs 1-14)"
echo "           Monotone exponential 4e-6→8e-7, warmup=10"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3c-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v3c-mode \
        --n-epochs       14 \
        --lr             4e-6 \
        --warmup-epochs  10 \
        --lr-schedule    exponential \
        --cosine-min-lr  8e-7 \
        --grad-clip-norm 0.25 \
        --weight-decay   1e-5 \
        --ema-decay      0.99997 \
        --pretrained-reg-lambda 3e-4 \
        --pretrained-reg-patterns \
            final_conv tor_bb_bond_conv tor_sc_bond_conv \
        --grad-accum     4 \
        --save-every     2 \
        --save-every-after 6 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: + cross_convs.3 at 0.1× LR
#   Monotone exponential: 1e-6 → 1e-7; warmup=8; epochs=16
#   3-tier diff LR: cross_convs.3=0.1×, output_convs=0.5×, heads=1.0×
#   Stronger pretrained-reg: lambda=5e-4 (more structural layers unfrozen now)
#   Save every epoch from ep1 — all stable checkpoints kept for benchmarking
#   Rationale: minimal receptor recognition recalibration; 0.1× ≈ near-frozen
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v3c-chain] Phase 2: + cross_convs.3 at 0.1×  (epochs 1-16)"
echo "           Monotone exponential 1e-6→1e-7, warmup=8"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v3c-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v3c-mode \
        --n-epochs       16 \
        --lr             1e-6 \
        --warmup-epochs  8 \
        --lr-schedule    exponential \
        --cosine-min-lr  1e-7 \
        --grad-clip-norm 0.25 \
        --weight-decay   2e-5 \
        --ema-decay      0.99997 \
        --pretrained-reg-lambda 5e-4 \
        --pretrained-reg-patterns \
            final_conv tor_bb_bond_conv tor_sc_bond_conv cross_convs.3 \
        --grad-accum     4 \
        --save-every     1 \
        --save-every-after 1 \
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
echo "[v3c-chain] ALL PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo ""
echo "  All P2 epoch checkpoints (save-every=1 from ep1):"
ls "$P2_OUT"/rapidock_finetuned_epoch*.pt 2>/dev/null | wc -l | xargs -I{} echo "    {} checkpoints"
echo ""
echo "  Recommended next steps:"
echo "  1. Benchmark all P2 checkpoints for best late-stable model:"
echo "       python3 scripts/benchmark_inference_multi.py \\"
echo "           --benchmark-csv  data/inference_benchmark_set.csv \\"
echo "           --pretrained     $PRETRAINED_CKPT \\"
echo "           --finetuned      $P2_CKPT \\"
echo "           --also-compare \\"
echo "               $P1_CKPT \\"
echo "           --n-samples 20 --seed 42 \\"
echo "           --out-dir logs/benchmark_v3c"
echo ""
echo "  2. Compare diversity + RMSD vs pretrained baseline"
echo "  3. If diversity preserved (≥0.50), consider extending P2 by 8 more epochs"
echo "=================================================================="
