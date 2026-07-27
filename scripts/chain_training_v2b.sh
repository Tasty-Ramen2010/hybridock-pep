#!/usr/bin/env bash
# chain_training_v2b.sh — full 3-phase v2 run from pretrained, no intervention.
# Runs alongside current v2/v1. Output dirs: finetune_peppc_v2b_phase{1,2,3}
# Analysis runs automatically at completion.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_CSV="$REPO/datasets/training_formatted_peppc/combined_train_curated.csv"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"
PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
SCRIPT="$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py"
CONDA="conda run --no-capture-output -n rapidock"

P1_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2b_phase1"
P2_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2b_phase2"
P3_OUT="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2b_phase3"
P1_BEST="$P1_OUT/rapidock_finetuned_best.pt"
P2_BEST="$P2_OUT/rapidock_finetuned_best.pt"

mkdir -p "$P1_OUT" "$P2_OUT" "$P3_OUT" "$REPO/logs"

# Phase 1 — score heads (27.9%), lr=2e-5, warmup=5, 25 epochs
$CONDA python3 -u "$SCRIPT" \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
    --checkpoint "$PRETRAINED" --output-dir "$P1_OUT" \
    --unfreeze-phase 1 --n-epochs 25 --lr 2e-5 --warmup-epochs 5 \
    --lr-schedule plateau --grad-accum 4 --grad-clip-norm 0.5 \
    --save-every 5 --early-stop-patience 12 --seed 42 \
    --esm-device cuda --bail-on-zero

# Phase 2 — +cross_convs.2/3 +intra_convs.3, lr=5e-6, warmup=8, 45 epochs
$CONDA python3 -u "$SCRIPT" \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
    --checkpoint "$P1_BEST" --output-dir "$P2_OUT" \
    --unfreeze-phase 2 --n-epochs 45 --lr 5e-6 --warmup-epochs 8 \
    --lr-schedule plateau --grad-accum 4 --grad-clip-norm 1.0 \
    --save-every 5 --early-stop-patience 15 --seed 42 \
    --esm-device cuda --bail-on-zero

# Phase 3 — full 7.5M, cosine, layerwise LR decay
$CONDA python3 -u "$SCRIPT" \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
    --checkpoint "$P2_BEST" --output-dir "$P3_OUT" \
    --unfreeze-phase 3 --n-epochs 100 --lr 1e-5 --warmup-epochs 10 \
    --lr-schedule cosine --cosine-min-lr 1e-7 \
    --grad-accum 4 --grad-clip-norm 1.0 --layerwise-lr-decay \
    --save-every 10 --early-stop-patience 20 --seed 42 \
    --esm-device cuda --bail-on-zero

# Analysis — training curves
$CONDA python3 -u "$REPO/scripts/analyze_training.py" \
    --phase-dirs "$P1_OUT" "$P2_OUT" "$P3_OUT" \
    --out-dir "$REPO/logs/analysis_v2b"

# Inference quality comparison — pretrained vs v2b Phase 3
echo ""
echo "========================================================================"
echo "[v2b] Inference quality comparison: pretrained vs v2b Phase 3"
echo "========================================================================"
P3_BEST="$P3_OUT/rapidock_finetuned_best.pt"
P3_FINAL="$P3_OUT/rapidock_finetuned_final.pt"
V2B_CKPT="$P3_BEST"
if [ ! -f "$V2B_CKPT" ] && [ -f "$P3_FINAL" ]; then
    V2B_CKPT="$P3_FINAL"
fi

if [ -f "$V2B_CKPT" ]; then
    conda run --no-capture-output -n score-env python3 -u \
        "$REPO/scripts/compare_rapidock_vs_finetuned.py" \
        --receptor   "$REPO/data/pdbs/1YCR_mdm2.pdb" \
        --peptide    "ETFSDLWKLLPE" \
        --reference  "$REPO/data/pdbs/1YCR_peptide.pdb" \
        --pretrained "$PRETRAINED" \
        --finetuned  "$V2B_CKPT" \
        --n-samples  20 \
        --seed       42 \
        --out-dir    "$REPO/logs/inference_comparison_v2b" \
        2>&1 | tee "$REPO/logs/inference_comparison_v2b.log" || \
        echo "[v2b] WARNING: inference comparison failed — see logs/inference_comparison_v2b.log"
    echo "[v2b] Inference comparison saved: logs/inference_comparison_v2b/"
else
    echo "[v2b] WARNING: no Phase 3 checkpoint found — skipping inference comparison"
fi

echo ""
echo "========================================================================"
echo "v2b COMPLETE."
echo "  Training analysis:      logs/analysis_v2b/"
echo "  Inference comparison:   logs/inference_comparison_v2b/"
echo "========================================================================"
