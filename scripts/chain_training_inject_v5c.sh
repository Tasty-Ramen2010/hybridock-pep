#!/usr/bin/env bash
# chain_training_inject_v5c.sh — injectable fixes applied on top of v5c-P2-best
#
# Two fixes injected into the ARCHITECTURE (in diffusion.py):
#   1. Tanh → SiLU in both torsion final layers  (unbounded; no saturating gradients)
#   2. cross_type_embedding(4, ns) zero-initialized (CA-CA/tip-CA/CA-tip/tip-tip contact signal)
#
# Strategy (starting from v5c-P2-best, which already has calibrated score heads + output convs):
#   P0 (3 ep):  torsion heads + cross_type_embedding only at ultra-low LR
#               — lets SiLU adapt from the Tanh-trained weights and cross_type_embedding warm up
#               — tor_*_final_layer.3.weight pre-scaled ×0.1 on load to compensate for SiLU range
#   P1 (12 ep): all score heads + output convs + cross_type_embedding + embeddings
#               — standard v5c-style calibration over the injected features
#
# Resumable: re-running is safe (--resume, skip if final.pt exists).
#
# Usage:
#   nohup bash scripts/chain_training_inject_v5c.sh > logs/chain_training_inject_v5c.log 2>&1 &
#   tail -f logs/chain_training_inject_v5c.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"

V5C_CKPT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v5c_phase2/rapidock_finetuned_best.pt"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"

P0_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_inject_v5c_phase0"
P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_inject_v5c_phase1"

CONDA_ENV="rapidock"

echo "[inject-v5c] Repo:       $REPO"
echo "[inject-v5c] Base ckpt:  $V5C_CKPT (v5c-P2-best)"
echo "[inject-v5c] Fixes:      Tanh→SiLU + cross_type_embedding(4,ns) zero-init"
echo ""

for f in "$V5C_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }
done

ckpt_info() {
    local path="$1" label="$2"
    [[ -f "$path" ]] || { echo "ERROR: checkpoint not found: $path"; exit 1; }
    local sz; sz=$(du -sh "$path" | cut -f1)
    echo "[inject-v5c-${label}] checkpoint: $path ($sz)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: torsion heads + cross_type_embedding only
#   LR: 1e-6 → 1e-7 (exponential, warmup=0, epochs=3)
#   Rationale: v5c weights trained with Tanh in torsion final layer; SiLU is
#   mathematically different in activation shape. 3-epoch warmup at tiny LR
#   lets gradient landscape re-equilibrate before touching anything else.
#   tor_*_final_layer.3.weight pre-scaled ×0.1 in load_model_for_inject to
#   avoid output blow-up (SiLU unbounded vs Tanh's [-1,1]).
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "[inject-v5c] Phase 0: torsion heads + cross_type_embedding  (epochs 1-3)"
echo "             Ultra-low LR 1e-6→1e-7, no warmup, SiLU adaptation"
echo "=================================================================="

if [[ -f "$P0_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[inject-v5c] Phase 0: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$V5C_CKPT" \
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
# Phase 1: all score heads + output convs + cross_type_embedding + embeddings
#   LR: 2e-6 → 2e-7 (exponential, warmup=2, epochs=12)
#   Rationale: v5c-style calibration but now including cross_type_embedding
#   in the unfrozen set so it can contribute signal beyond torsion.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "[inject-v5c] Phase 1: full inject set  (epochs 1-12)"
echo "             LR 2e-6→2e-7, warmup=2"
echo "=================================================================="

if [[ -f "$P1_OUT/rapidock_finetuned_final.pt" ]]; then
    echo "[inject-v5c] Phase 1: ALREADY COMPLETE — skipping"
else
    conda run --no-capture-output -n "$CONDA_ENV" python3 -u "$TRAIN_SCRIPT" \
        --train-csv      "$TRAIN_CSV" \
        --val-csv        "$VAL_CSV" \
        --checkpoint     "$P0_CKPT" \
        --output-dir     "$P1_OUT" \
        --unfreeze-phase 1 \
        --inject-mode \
        --n-epochs       12 \
        --lr             2e-6 \
        --warmup-epochs  2 \
        --lr-schedule    exponential \
        --cosine-min-lr  2e-7 \
        --grad-clip-norm 0.2 \
        --weight-decay   2e-5 \
        --ema-decay      0.99998 \
        --pretrained-reg-lambda 1e-4 \
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

P1_CKPT="$P1_OUT/rapidock_finetuned_best.pt"
ckpt_info "$P1_CKPT" "p1"

echo ""
echo "=================================================================="
echo "[inject-v5c] ALL PHASES COMPLETE"
echo "  P0 best: $P0_CKPT"
echo "  P1 best: $P1_CKPT"
echo "=================================================================="
