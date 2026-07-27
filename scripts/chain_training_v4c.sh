#!/usr/bin/env bash
# chain_training_v4c.sh — V4C: careful cross_conv adaptation test
#
# Core philosophy: test whether TINY cross_conv learning can improve
# receptor-specific interactions without destabilising the pretrained
# exploration manifold.  All optimisation energy minimised; structural
# layers held to near-pretrained via strong reg + very low LR multipliers.
#
# Key differences vs v3c:
#   - Phase 1 also includes cross_convs.3 at 0.15× LR (v3c had cc.3 in P2 only)
#   - Phase 2 adds cross_convs.2 at 0.08× (even more conservative than cc.3)
#   - Slightly higher base LR (5e-6 vs 4e-6) to compensate for 0.15× cc.3 scaling
#   - grad_clip=0.3 (slightly looser than v3c's 0.25 to accommodate cc.3 in P1)
#   - NO phase 3 — equivariant + intra_conv geometry priors preserved
#
# Design choices from v3b/v3c analysis (May 2026):
#   - Pretrained manifold is near-optimal; specialisation must be near-zero energy
#   - cc.3 = final cross-attention layer; most specific to binding geometry
#   - cc.2 = penultimate; tiny update at 0.08× ≈ near-zero shift
#   - Monotone LR only; spike → PERMANENT 50% reduction, never recovers
#
# Phase 1: score heads + output convs + cross_convs.3 at 0.15×
#   - LR: 5e-6 → 1e-6 (exponential, warmup=10, epochs=14)
#   - EMA=0.99997, grad_clip=0.3, weight_decay=1e-5
#   - 2-tier diff LR: cc.3=0.15×, heads+output_convs=1.0×
#   - Pretrained-reg (lambda=3e-4) on output_convs + cc.3
#
# Phase 2: + cross_convs.2 at 0.08×
#   - LR: 1.2e-6 → 1e-7 (exponential, warmup=8, epochs=18)
#   - EMA=0.99997, grad_clip=0.3, weight_decay=2e-5
#   - 3-tier diff LR: cc.2=0.08×, cc.3=0.15×, heads+output_convs=1.0×
#   - Pretrained-reg (lambda=5e-4) on output_convs + cc.3 + cc.2
#   - Save every epoch from ep1 (all checkpoints for late-stable benchmarking)
#
# Resumable: re-running this script is safe (--resume + phase skip if final.pt exists).
#
# Usage:
#   nohup bash scripts/chain_training_v4c.sh > logs/chain_training_v4c.log 2>&1 &
#   tail -f logs/chain_training_v4c.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v4c_phase2"

CONDA_ENV="rapidock"

echo "[v4c-chain] Repo:       $REPO"
echo "[v4c-chain] Pretrained: $PRETRAINED_CKPT"
echo "[v4c-chain] Philosophy: tiny cross_conv adaptation test, monotone LR, NO phase 3"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[v4c-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads + output convs + cross_convs.3 at 0.15×
#   Monotone exponential: 5e-6 → 1e-6; warmup=10; epochs=14
#   2-tier diff LR: cc.3=0.15×, heads+output_convs=1.0×
#   Pretrained-reg: lambda=3e-4 on output_convs + cc.3
#   Rationale: calibrate score magnitude + minimal cc.3 receptor shift
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[v4c-chain] Phase 1: score heads + output convs + cc.3 at 0.15×  (epochs 1-14)"
echo "           Monotone exponential 5e-6→1e-6, warmup=10"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4c-chain] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --v4c-mode \
        --n-epochs       14 \
        --lr             5e-6 \
        --warmup-epochs  10 \
        --lr-schedule    exponential \
        --cosine-min-lr  1e-6 \
        --grad-clip-norm 0.3 \
        --weight-decay   1e-5 \
        --ema-decay      0.99997 \
        --pretrained-reg-lambda 3e-4 \
        --pretrained-reg-patterns \
            final_conv tor_bb_bond_conv tor_sc_bond_conv cross_convs.3 \
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
# Phase 2: + cross_convs.2 at 0.08×
#   Monotone exponential: 1.2e-6 → 1e-7; warmup=8; epochs=18
#   3-tier diff LR: cc.2=0.08×, cc.3=0.15×, heads+output_convs=1.0×
#   Stronger pretrained-reg: lambda=5e-4 (more structural layers unfrozen now)
#   Save every epoch from ep1 — all stable checkpoints kept for benchmarking
#   Rationale: near-zero cc.2 update; 0.08× ≈ almost frozen in practice
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[v4c-chain] Phase 2: + cross_convs.2 at 0.08×  (epochs 1-18)"
echo "           Monotone exponential 1.2e-6→1e-7, warmup=8"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[v4c-chain] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 2 \
        --v4c-mode \
        --n-epochs       18 \
        --lr             1.2e-6 \
        --warmup-epochs  8 \
        --lr-schedule    exponential \
        --cosine-min-lr  1e-7 \
        --grad-clip-norm 0.3 \
        --weight-decay   2e-5 \
        --ema-decay      0.99997 \
        --pretrained-reg-lambda 5e-4 \
        --pretrained-reg-patterns \
            final_conv tor_bb_bond_conv tor_sc_bond_conv cross_convs.3 cross_convs.2 \
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
echo "[v4c-chain] ALL PHASES COMPLETE"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo ""
echo "  All P2 epoch checkpoints (save-every=1 from ep1):"
ls "$P2_OUT"/rapidock_finetuned_epoch*.pt 2>/dev/null | wc -l | xargs -I{} echo "    {} checkpoints"
echo ""
echo "  Recommended next steps:"
echo "  1. Benchmark all P2 checkpoints vs v3c for best cross_conv adaptation:"
echo "       python3 scripts/benchmark_inference_multi.py \\"
echo "           --benchmark-csv  data/inference_benchmark_set.csv \\"
echo "           --pretrained     $PRETRAINED_CKPT \\"
echo "           --finetuned      $P2_CKPT \\"
echo "           --also-compare \\"
echo "               $P1_CKPT \\"
echo "           --n-samples 20 --seed 42 \\"
echo "           --out-dir logs/benchmark_v4c"
echo ""
echo "  2. Compare diversity ratio vs v3c P2 and pretrained baseline"
echo "  3. If cc.3/cc.2 shift is detectable (diversity drops <0.10 vs v3c) — stop"
echo "  4. If neutral, this is the pattern to extend to more cc layers"
echo "=================================================================="
