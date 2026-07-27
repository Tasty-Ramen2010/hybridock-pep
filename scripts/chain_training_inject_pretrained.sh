#!/usr/bin/env bash
# chain_training_inject_pretrained.sh — injectable fixes applied from scratch (pretrained base)
#
# Same architectural fixes as inject-v5c but starting from the original pretrained weights.
# Runs the full v5c training schedule with inject-mode so the fixed architecture is trained
# from the beginning — gives a clean baseline for inject vs pure v5c comparison.
#
# Fixes injected into diffusion.py:
#   1. Tanh → SiLU in both torsion final layers
#   2. cross_type_embedding(4, ns) zero-initialized
#
# Schedule (mirrors v5c chain but with --inject-mode throughout):
#   P0 (3 ep):   torsion heads + cross_type_embedding only, ultra-low LR
#                — SiLU warmup from pretrained Tanh-trained weights
#   P1 (10 ep):  score heads only (like v5c-P1)
#   P2 (12 ep):  + output convs (like v5c-P2) + cross_type_embedding continues
#
# Resumable: re-running is safe (--resume, skip if final.pt exists).
#
# Usage:
#   nohup bash scripts/chain_training_inject_pretrained.sh > logs/chain_training_inject_pretrained.log 2>&1 &
#   tail -f logs/chain_training_inject_pretrained.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

PRETRAINED_CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P0_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_inject_pretrained_phase0"
P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_inject_pretrained_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_inject_pretrained_phase2"

CONDA_ENV="rapidock"

echo "[inject-pretrained] Repo:       $REPO"
echo "[inject-pretrained] Base ckpt:  $PRETRAINED_CKPT (rapidock_global.pt)"
echo "[inject-pretrained] Fixes:      Tanh→SiLU + cross_type_embedding(4,ns) zero-init"
echo ""

for f in "$PRETRAINED_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[inject-pretrained-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: torsion heads + cross_type_embedding only
#   LR: 1e-6 → 1e-7 (exponential, warmup=0, epochs=3)
#   Same warmup rationale as inject-v5c P0.
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[inject-pretrained] Phase 0: torsion heads + cross_type_embedding  (epochs 1-3)"
echo "                    Ultra-low LR 1e-6→1e-7, no warmup"
echo "=================================================================="

if [[ -f "$P0_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[inject-pretrained] Phase 0: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$PRETRAINED_CKPT" \
        --output-dir     "$P0_OUT" \
        --unfreeze-phase 0 \
        --inject-mode \
        --n-epochs       3 \
        --lr             1e-6 \
        --warmup-epochs  0 \
        --lr-schedule    exponential \
        --cosine-min-lr  1e-7 \
        --grad-clip-norm 0.1 \
        --weight-decay   0.0 \
        --ema-decay      0.99998 \
        --pretrained-reg-lambda 0 \
        --grad-accum     4 \
        --save-every     1 \
        --save-every-after 1 \
        --early-stop-patience 999 \
        --seed           42 \
        --esm-device     cuda \
        --bail-on-zero \
        --resume
fi

P0_CKPT="$P0_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P0_CKPT" "p0"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: score heads only  (mirrors v5c-P1)
#   LR: 2e-6 → 2e-7 (exponential, warmup=3, epochs=10)
#   cross_type_embedding included in INJECT_P1 patterns so it continues training
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[inject-pretrained] Phase 1: score heads only  (epochs 1-10)"
echo "                    LR 2e-6→2e-7, warmup=3"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[inject-pretrained] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P0_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --inject-mode \
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
# Phase 2: + output convs  (mirrors v5c-P2, uses INJECT_P1 patterns which already
#           include output convs — no separate INJECT_P2 needed)
#   LR: 5e-7 → 5e-8 (exponential, warmup=2, epochs=12)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[inject-pretrained] Phase 2: + output convs  (epochs 1-12)"
echo "                    LR 5e-7→5e-8, warmup=2"
echo "=================================================================="

if [[ -f "$P2_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[inject-pretrained] Phase 2: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P1_CKPT" \
        --output-dir     "$P2_OUT" \
        --unfreeze-phase 1 \
        --inject-mode \
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
echo "[inject-pretrained] ALL PHASES COMPLETE"
echo "  P0 best: $P0_CKPT"
echo "  P1 best: $P1_CKPT"
echo "  P2 best: $P2_CKPT"
echo "=================================================================="
