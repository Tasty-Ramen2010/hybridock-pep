#!/usr/bin/env bash
# run_finetune_and_compare.sh
#
# Runs last-layer fine-tuning then the original-vs-finetuned comparison.
# Aborts if the OpenMM FEP simulations are still running.
#
# Usage:
#   bash scripts/run_finetune_and_compare.sh
#   (or via the Sunday cron job)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
LOG="$REPO/runs/finetune_and_compare.log"
mkdir -p "$REPO/runs"

echo "[$(date)] Starting run_finetune_and_compare.sh" | tee -a "$LOG"

# --- FEP guard ----------------------------------------------------------
if pgrep -f "fep_complex_leg\|fep_solvent_leg" > /dev/null 2>&1; then
    echo "[$(date)] ERROR: FEP simulations still running. Aborting to avoid GPU contention." | tee -a "$LOG"
    exit 1
fi
echo "[$(date)] FEP check passed — simulations not running." | tee -a "$LOG"

# --- Training -----------------------------------------------------------
CKPT="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
OUTDIR="$REPO/third_party/RAPiDock_finetuned/finetune_out"

if [ ! -f "$CKPT" ]; then
    echo "[$(date)] ERROR: Checkpoint not found at $CKPT" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Launching last-layer fine-tuning (30 epochs)..." | tee -a "$LOG"

# Remove stale checkpoints from any previous (broken) run so we don't accidentally
# pick up pretrained-weight checkpoints as if they were trained ones.
rm -f "$OUTDIR"/rapidock_finetuned_*.pt
echo "[$(date)] Stale checkpoints cleared from $OUTDIR" | tee -a "$LOG"

/home/igem/miniconda3/envs/rapidock/bin/python -u \
    "$REPO/third_party/RAPiDock_finetuned/train_lastlayer.py" \
    --train-csv "$REPO/datasets/training_formatted/training_data.csv" \
    --val-csv   "$REPO/datasets/training_formatted/val_data.csv" \
    --checkpoint "$CKPT" \
    --output-dir "$OUTDIR" \
    --n-epochs 50 \
    --lr 1e-4 \
    --ppii-weight 4 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Training complete." | tee -a "$LOG"

# --- Comparison ---------------------------------------------------------
# Use the FINAL checkpoint (most trained weights), not "best" (which would
# be epoch 1 if val_loss is 0.0 due to eval-mode compute_loss issue).
# Fall back to best if final doesn't exist (e.g. training was interrupted).
FINAL_CKPT="$OUTDIR/rapidock_finetuned_final.pt"
BEST_CKPT="$OUTDIR/rapidock_finetuned_best.pt"
if [ -f "$FINAL_CKPT" ]; then
    USE_CKPT="$FINAL_CKPT"
    echo "[$(date)] Using final trained checkpoint: $FINAL_CKPT" | tee -a "$LOG"
elif [ -f "$BEST_CKPT" ]; then
    USE_CKPT="$BEST_CKPT"
    echo "[$(date)] WARNING: final checkpoint not found, using best checkpoint (may be epoch 1)" | tee -a "$LOG"
else
    echo "[$(date)] ERROR: No fine-tuned checkpoint found after training. Check $LOG" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Launching model comparison..." | tee -a "$LOG"
PATH="/home/igem/miniconda3/envs/rapidock/bin:$PATH" \
    bash "$REPO/scripts/compare_rapidock_models.sh" \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done. Results in runs/model_comparison/" | tee -a "$LOG"
