#!/usr/bin/env bash
# monitor_and_launch.sh
#
# Monitors v3b training (all 3 phases run by chain_training_v3b.sh),
# reports GPU + training status every hour, and alerts on errors.
#
# v3b chain is already running independently (launched directly).
# This script just watches and reports progress.
#
# Usage (run in a persistent tmux session):
#   tmux new-session -d -s training_monitor
#   tmux send-keys -t training_monitor \
#       'bash scripts/monitor_and_launch.sh 2>&1 | tee -a logs/monitor_and_launch.log' Enter
#
# Check status any time:
#   tail -50 logs/monitor_and_launch.log
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FINETUNED="$REPO/third_party/RAPiDock_finetuned"
LOGS="$REPO/logs"

V3B_LOG="$LOGS/chain_training_v3b.log"

POLL_INTERVAL=3600   # 1 hour between status checks

# ─── helpers ──────────────────────────────────────────────────────────────────

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
separator() { echo ""; echo "══════════════════════════════════════════════════════"; }

gpu_status() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi \
            --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{
            printf "  GPU: %s | util=%s%% | mem=%sMiB/%sMiB | temp=%s°C\n",
                   $1,$2,$3,$4,$5
          }' || echo "  GPU: nvidia-smi error"
    else
        echo "  GPU: nvidia-smi not found"
    fi
}

is_running() {
    local out_dir="$1"
    pgrep -f "output-dir.*$(basename "$out_dir")" &>/dev/null
}

is_complete() {
    local out_dir="$1"
    [[ -f "$out_dir/rapidock_finetuned_final.pt" ]]
}

# Comprehensive v3b chain status
v3b_chain_status() {
    echo ""
    echo "  ── v3b chain progress ──"
    local any_running=false
    for phase in 1 2 3; do
        local pout="$FINETUNED/finetune_peppc_v3b_phase${phase}"
        local pfinal="$pout/rapidock_finetuned_final.pt"
        local pbest="$pout/rapidock_finetuned_best.pt"
        local nbest; nbest=$(ls "$pout"/rapidock_finetuned_epoch*.pt 2>/dev/null | wc -l)
        if [[ -f "$pfinal" ]]; then
            echo "  P${phase}: COMPLETE ✓  ($(du -sh $pfinal | cut -f1))"
        elif is_running "$pout"; then
            any_running=true
            # Parse last epoch from log
            local last; last=$(grep "^Epoch" "$V3B_LOG" 2>/dev/null | \
                grep "phase${phase}" 2>/dev/null || \
                grep "^Epoch" "$V3B_LOG" 2>/dev/null | tail -1 || echo "(starting)")
            local best_line; best_line=$(grep "✓ New best" "$V3B_LOG" 2>/dev/null | tail -1 || echo "(no best yet)")
            echo "  P${phase}: RUNNING"
            echo "           Last:   $(grep "^Epoch" "$V3B_LOG" 2>/dev/null | tail -1 || echo "(starting)")"
            echo "           Best:   $best_line"
            echo "           Saved epoch ckpts: $nbest"
            # Spike LR events
            grep "SPIKE LR" "$V3B_LOG" 2>/dev/null | tail -2 | sed 's/^/           SPIKE: /' || true
        elif [[ -f "$pbest" ]]; then
            echo "  P${phase}: STOPPED (best checkpoint exists; process not found)"
            echo "           $(ls -lh $pbest | awk '{print $5, $7, $8}')"
        else
            echo "  P${phase}: waiting (not started yet)"
        fi
    done
}

# ─── main ─────────────────────────────────────────────────────────────────────

separator
log "monitor_and_launch.sh starting — monitoring v3b chain"
log "Repo: $REPO"
log "Poll interval: $((POLL_INTERVAL / 60)) minutes"
log "Log: $V3B_LOG"
echo ""
echo "  History:"
echo "    v2/v2b: terminated ep50, best from ep22 (val~47.57)"
echo "    v3:     terminated: P1 complete (best ep16 val=39.005); P2 killed ep18 (best ep16 val=39.698)"
echo "    v3b:    RUNNING — stable controlled spec (cosine + adaptive spike LR)"
echo ""

# Initial status
separator
log "INITIAL STATUS"
gpu_status
v3b_chain_status

# ── Monitor loop ───────────────────────────────────────────────────────────────
while true; do
    # Check if v3b P3 is complete — we're done
    if is_complete "$FINETUNED/finetune_peppc_v3b_phase3"; then
        separator
        log "V3B ALL 3 PHASES COMPLETE ✓"
        gpu_status
        echo ""
        echo "  Checkpoints:"
        for p in 1 2 3; do
            ckpt="$FINETUNED/finetune_peppc_v3b_phase${p}/rapidock_finetuned_best.pt"
            [[ -f "$ckpt" ]] && echo "    ✓ P${p}: $ckpt" || echo "    ✗ P${p}: MISSING"
        done
        echo ""
        echo "  Next steps:"
        echo "    1. Compare v3 vs v3b: python3 scripts/analyze_training.py \\"
        echo "         --phase-dirs $FINETUNED/finetune_peppc_v3b_phase{1,2,3} \\"
        echo "         --compare-dirs $FINETUNED/finetune_peppc_v3_phase{1,2,3} \\"
        echo "         --out-dir logs/analysis_v3_v3b"
        echo "    2. Run benchmark: python3 scripts/benchmark_inference_multi.py"
        echo "    3. Update Excel:  python3 scripts/update_training_excel.py"
        log "monitor_and_launch.sh complete."
        exit 0
    fi

    sleep "$POLL_INTERVAL"

    separator
    log "STATUS CHECK — v3b training"
    gpu_status
    v3b_chain_status

    # Show recent log
    echo ""
    echo "  Recent v3b log (last 8 lines):"
    tail -8 "$V3B_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (no log yet)"

    # Check if chain process died unexpectedly
    # (check if any v3b phase is running; if not and P3 not complete → crash)
    v3b_any_active=false
    for phase in 1 2 3; do
        is_running "$FINETUNED/finetune_peppc_v3b_phase${phase}" && v3b_any_active=true && break || true
    done
    # Also check the chain_training_v3b.sh process itself
    pgrep -f "chain_training_v3b" &>/dev/null && v3b_any_active=true || true

    if ! $v3b_any_active; then
        if is_complete "$FINETUNED/finetune_peppc_v3b_phase3"; then
            : # handled above at top of loop next iteration
        else
            log "WARNING: v3b chain process not found but P3 not complete — may have crashed!"
            log "  Check $V3B_LOG for errors."
            log "  To resume: nohup bash scripts/chain_training_v3b.sh >> $V3B_LOG 2>&1 &"
        fi
    fi

    log "Next check in $((POLL_INTERVAL / 60)) minutes..."
done
