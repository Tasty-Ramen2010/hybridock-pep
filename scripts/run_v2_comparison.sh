#!/usr/bin/env bash
# run_v2_comparison.sh — One command to test V2 RAPiDock improvements.
#
# Steps:
#   1. Wait for Exp E to finish (if still running)
#   2. Extract V2 features from bench300 (V2ConfidenceModel: cross-attn +
#      sidechain proxy + receptor SS proxy)
#   3. Train and compare V1 (96-dim) vs V2 (115-dim) heads via 5-fold CV
#   4. Print final Δτ verdict
#
# Usage:
#   bash scripts/run_v2_comparison.sh [--skip-wait] [--device cuda] [--workers 8]
#
# Options:
#   --skip-wait   Don't wait for Exp E; run V2 extraction immediately
#   --device DEV  cuda or cpu (default: cuda)
#   --workers N   Parallel workers for head training (default: 8)
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAPIDOCK_PY="/home/igem/miniconda3/envs/rapidock/bin/python3"
LOG_DIR="$REPO/logs"
EXP_E_LOG="$LOG_DIR/training_campaign/campaign_exp_e_rerun.log"
V2_FEAT="$LOG_DIR/diagnosis/feats_bench300_v2.pkl"
V2_EXTRACT_LOG="$LOG_DIR/extract_v2_features.log"
V2_TRAIN_LOG="$LOG_DIR/v2_comparison/train_v2_head.log"

SKIP_WAIT=0
DEVICE="cuda"
WORKERS=8

for arg in "$@"; do
    case $arg in
        --skip-wait) SKIP_WAIT=1 ;;
        --device=*)  DEVICE="${arg#*=}" ;;
        --device)    shift; DEVICE="$1" ;;
        --workers=*) WORKERS="${arg#*=}" ;;
        --workers)   shift; WORKERS="$1" ;;
    esac
done

echo "================================================"
echo "  V2 RAPiDock Feature Comparison Pipeline"
echo "  $(date)"
echo "================================================"

# ── Step 1: Wait for Exp E ────────────────────────────────────────────────────
if [[ $SKIP_WAIT -eq 0 ]]; then
    if pgrep -f "confidence_training_campaign" > /dev/null 2>&1; then
        echo ""
        echo "[1/3] Exp E is running — waiting for it to complete..."
        echo "      (use --skip-wait to bypass this step)"
        until grep -q "Saved.*all_results\|exp_e_rerun.*done\|E_finetune" "$EXP_E_LOG" 2>/dev/null && \
              ! pgrep -f "confidence_training_campaign" > /dev/null 2>&1; do
            sleep 30
            LAST=$(tail -1 "$EXP_E_LOG" 2>/dev/null || echo "(no output yet)")
            echo "      Exp E status: $LAST"
        done
        echo "      Exp E finished."
    else
        echo "[1/3] Exp E not running (already done or skipped)."
    fi
else
    echo "[1/3] --skip-wait set; skipping Exp E wait."
fi

# ── Step 2: Extract V2 features ───────────────────────────────────────────────
echo ""
if [[ -f "$V2_FEAT" ]]; then
    N_FEATS=$(python3 -c "import pickle; d=pickle.load(open('$V2_FEAT','rb')); print(len(d))" 2>/dev/null || echo "?")
    echo "[2/3] V2 features already exist ($N_FEATS entries). Skipping extraction."
    echo "      Delete $V2_FEAT to re-extract."
else
    echo "[2/3] Extracting V2 features (ESM ~2 min + inference ~10 min)..."
    mkdir -p "$LOG_DIR/v2_comparison"
    PYTHONPATH="$REPO" $RAPIDOCK_PY -u \
        "$REPO/scripts/extract_features_v2.py" \
        --device "$DEVICE" \
        --batch-size 8 \
        2>&1 | tee "$V2_EXTRACT_LOG"
    if [[ ! -f "$V2_FEAT" ]]; then
        echo "ERROR: V2 feature extraction failed. Check $V2_EXTRACT_LOG"
        exit 1
    fi
    echo "      V2 features saved → $V2_FEAT"
fi

# ── Step 3: Train and compare ─────────────────────────────────────────────────
echo ""
echo "[3/3] Training V1 vs V2 heads (5-fold CV × 3 seeds, epochs=50)..."
mkdir -p "$LOG_DIR/v2_comparison"
PYTHONPATH="$REPO" $RAPIDOCK_PY -u \
    "$REPO/scripts/train_v2_head.py" \
    --epochs 50 \
    --workers "$WORKERS" \
    2>&1 | tee "$V2_TRAIN_LOG"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  COMPARISON COMPLETE"
echo "================================================"
SUMMARY="$LOG_DIR/v2_comparison/v2_comparison_summary.csv"
if [[ -f "$SUMMARY" ]]; then
    python3 - <<'PYEOF'
import pandas as pd, sys
from pathlib import Path
df = pd.read_csv(Path(__file__).resolve().parent.parent / "logs/v2_comparison/v2_comparison_summary.csv" if False else "logs/v2_comparison/v2_comparison_summary.csv")
for _, r in df.iterrows():
    print(f"  {r['label']:15s}  mean_τ={r['mean_tau']:.4f} ± {r['std_tau']:.4f}  max_τ={r['max_tau']:.4f}  top1_RMSD={r['mean_top1']:.3f}Å")
rows = df.set_index("label")
if "V2_115dim" in rows.index and "V1_96dim" in rows.index:
    delta = rows.loc["V2_115dim","mean_tau"] - rows.loc["V1_96dim","mean_tau"]
    verdict = "V2 IMPROVES SIGNIFICANTLY" if delta > 0.010 else \
              "V2 marginal gain" if delta > 0.002 else \
              "No improvement — re-check features" if delta >= 0 else \
              "V1 is better — V2 changes need review"
    print(f"\n  Δτ = {delta:+.4f}  →  {verdict}")
PYEOF
fi
echo ""
echo "Full results → logs/v2_comparison/"
echo "Exp E results → logs/training_campaign/campaign_exp_e_rerun.log"
