#!/usr/bin/env bash
# run_after_current_chain.sh — waits for the current Phase 2/3 chain (v1) to
# finish, then runs the comparison test, then launches chain_v2 from the
# original pretrained checkpoint.
#
# Start this now in background:
#   bash scripts/run_after_current_chain.sh 2>&1 | tee logs/pipeline_v2.log &
#   echo "Post-chain pipeline PID: $!"

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="conda run --no-capture-output -n rapidock"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAINED="$REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
V1_P3_BEST="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase3/rapidock_finetuned_best.pt"
V1_P3_FINAL="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase3/rapidock_finetuned_final.pt"
VAL_CSV="$REPO/datasets/training_formatted_peppc/combined_val_curated.csv"
V2_P3_BEST="$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3/rapidock_finetuned_best.pt"

mkdir -p "$REPO/logs"

# ─── Step 1: Wait for current chain (PID 1597156) to exit ────────────────────
CHAIN_PID=1597156
echo ""
echo "========================================================================"
echo "[pipeline] Waiting for current v1 chain (PID $CHAIN_PID) to complete..."
echo "========================================================================"

while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 60
    echo "[pipeline] $(date '+%H:%M:%S') — chain still running (PID $CHAIN_PID)..."
done

echo "[pipeline] $(date) — v1 chain exited."

# ─── Step 2: Run comparison (pretrained vs v1 Phase 3) ────────────────────────
echo ""
echo "========================================================================"
echo "[pipeline] Running comparison: pretrained vs v1 Phase 3 best"
echo "========================================================================"

# Use the best checkpoint if it exists, else fall back to final
V1_BEST_OR_FINAL="$V1_P3_BEST"
if [ ! -f "$V1_P3_BEST" ] && [ -f "$V1_P3_FINAL" ]; then
    echo "[pipeline] No best.pt — using final.pt as comparison target"
    V1_BEST_OR_FINAL="$V1_P3_FINAL"
fi

if [ -f "$V1_BEST_OR_FINAL" ]; then
    $CONDA python3 -u "$REPO/scripts/compare_finetuned.py" \
        --pretrained  "$PRETRAINED" \
        --finetuned   "$V1_BEST_OR_FINAL" \
        --val-csv     "$VAL_CSV" \
        --out         "$REPO/logs/comparison_v1.json" \
        --device      cuda \
        2>&1 | tee "$REPO/logs/comparison_v1.log"
    echo "[pipeline] Comparison complete — see logs/comparison_v1.json"
else
    echo "[pipeline] WARNING: v1 Phase 3 checkpoint not found ($V1_P3_BEST)"
    echo "  Skipping comparison. Check chain log for Phase 3 failure."
fi

# ─── Step 3: Analyse v1 training history ─────────────────────────────────────
echo ""
echo "[pipeline] Analysing v1 training history..."
$CONDA python3 -u "$REPO/scripts/analyze_training.py" \
    --phase-dirs \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase1" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase2" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase3" \
    --out-dir "$REPO/logs/analysis_v1" \
    2>&1 | tee "$REPO/logs/analysis_v1.log" || true   # don't abort pipeline on analysis failure

# ─── Step 4 (SKIPPED): v2 chain was started manually before v1 finished ───────
# v2 is already running in parallel. This pipeline just waits for v2 to complete
# so it can do the final comparison.
echo "[pipeline] v2 chain was already started in parallel — waiting for it to finish..."
V2_PID_FILE="$REPO/logs/chain_v2.pid"
if [ -f "$V2_PID_FILE" ]; then
    V2_PID=$(cat "$V2_PID_FILE")
    echo "[pipeline] v2 PID: $V2_PID"
    while kill -0 "$V2_PID" 2>/dev/null; do
        sleep 120
        echo "[pipeline] $(date '+%H:%M:%S') — v2 still running..."
    done
    echo "[pipeline] v2 chain done."
else
    echo "[pipeline] WARNING: no v2 PID file found. Checking for v2 Phase 3 checkpoint..."
    # Just wait for the v2 Phase 3 output to appear
    while [ ! -f "$V2_P3_BEST" ] && [ ! -f "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3/rapidock_finetuned_final.pt" ]; do
        sleep 300
        echo "[pipeline] $(date '+%H:%M:%S') — waiting for v2 Phase 3 checkpoint..."
    done
fi

# ─── Step 6: Post-v2 comparison ──────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[pipeline] Running final comparison: pretrained vs v2 Phase 3"
echo "========================================================================"

if [ -f "$V2_P3_BEST" ]; then
    $CONDA python3 -u "$REPO/scripts/compare_finetuned.py" \
        --pretrained   "$PRETRAINED" \
        --finetuned    "$V2_P3_BEST" \
        --also-compare "$V1_BEST_OR_FINAL" \
        --val-csv      "$VAL_CSV" \
        --out          "$REPO/logs/comparison_v1_vs_v2.json" \
        --device       cuda \
        2>&1 | tee "$REPO/logs/comparison_v1_vs_v2.log"
    echo "[pipeline] Final comparison saved: logs/comparison_v1_vs_v2.json"
fi

# ─── Step 7: Analyse v2 and compare ──────────────────────────────────────────
echo "[pipeline] Analysing v2 and comparing to v1..."
$CONDA python3 -u "$REPO/scripts/analyze_training.py" \
    --phase-dirs \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase1" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase2" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3" \
    --compare-dirs \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase1" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase2" \
        "$REPO/third_party/RAPiDock_finetuned/finetune_peppc_phase3" \
    --out-dir "$REPO/logs/analysis_v1_vs_v2" \
    2>&1 | tee "$REPO/logs/analysis_comparison.log" || true

# ─── Step 8: Inference quality comparison (pretrained vs v1 vs v2 Phase 3) ──
echo ""
echo "========================================================================"
echo "[pipeline] Inference quality comparison: pretrained vs v1 vs v2 Phase 3"
echo "========================================================================"

V1_P3_BEST_OR_FINAL="$V1_P3_BEST"
if [ ! -f "$V1_P3_BEST" ] && [ -f "$V1_P3_FINAL" ]; then
    V1_P3_BEST_OR_FINAL="$V1_P3_FINAL"
fi

ALSO_COMPARE_ARG=""
if [ -f "$V2_P3_BEST" ]; then
    ALSO_COMPARE_ARG="--also-compare $V2_P3_BEST"
fi

if [ -f "$V1_P3_BEST_OR_FINAL" ]; then
    conda run --no-capture-output -n score-env python3 -u \
        "$REPO/scripts/compare_rapidock_vs_finetuned.py" \
        --receptor   "$REPO/data/pdbs/1YCR_mdm2.pdb" \
        --peptide    "ETFSDLWKLLPE" \
        --reference  "$REPO/data/pdbs/1YCR_peptide.pdb" \
        --pretrained "$PRETRAINED" \
        --finetuned  "$V1_P3_BEST_OR_FINAL" \
        $ALSO_COMPARE_ARG \
        --n-samples  20 \
        --seed       42 \
        --out-dir    "$REPO/logs/inference_comparison_v1_v2" \
        2>&1 | tee "$REPO/logs/inference_comparison_v1_v2.log" || \
        echo "[pipeline] WARNING: inference comparison failed — see logs/inference_comparison_v1_v2.log"
    echo "[pipeline] Inference comparison saved: logs/inference_comparison_v1_v2/"
else
    echo "[pipeline] WARNING: no Phase 3 checkpoint found — skipping inference comparison"
fi

echo ""
echo "========================================================================"
echo "[pipeline] FULL PIPELINE COMPLETE."
echo "  v1 comparison:            logs/comparison_v1.json"
echo "  v1 vs v2 comparison:      logs/comparison_v1_vs_v2.json"
echo "  v2 analysis:              logs/analysis_v1_vs_v2/"
echo "  Inference quality report: logs/inference_comparison_v1_v2/"
echo "========================================================================"
