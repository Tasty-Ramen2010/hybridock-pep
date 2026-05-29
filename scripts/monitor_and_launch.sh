#!/usr/bin/env bash
# monitor_and_launch.sh
#
# Monitors v3 P3 training, reports GPU + training status every hour,
# then automatically launches v3b when v3 finishes.
#
# Sequence:
#   v3  — running independently (chain_training_v3.sh, all phases chained)
#   v3b — launched here after v3 P3 completes (chain_training_v3b.sh)
#
# Note: v2/v2b were terminated at ep50 (best checkpoints at ep22, val~47.57).
#       Their P3 final.pt files exist as tombstones (copied from best.pt).
#       v4/v5 chains are on hold pending hyperparameter fixes.
#
# Usage (run in a persistent tmux session):
#   tmux new-session -d -s training_monitor
#   tmux send-keys -t training_monitor \
#       'bash scripts/monitor_and_launch.sh 2>&1 | tee logs/monitor_and_launch.log' Enter
#
# Check status any time:
#   tail -50 logs/monitor_and_launch.log
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FINETUNED="$REPO/third_party/RAPiDock_finetuned"
LOGS="$REPO/logs"
SCRIPTS="$REPO/scripts"

V3_P3_OUT="$FINETUNED/finetune_peppc_v3_phase3"
V3_LOG="$LOGS/chain_training_v3.log"

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

# Return the last epoch line from a training log
last_epoch_line() {
    local log="$1"
    grep "^Epoch" "$log" 2>/dev/null | tail -1 || echo "  (no epoch data)"
}

# Return the best val loss line from a training log
best_val_line() {
    local log="$1"
    grep "✓ New best" "$log" 2>/dev/null | tail -1 || echo "  (no best yet)"
}

# Check if a chain is still running by matching the output-dir argument in ps
is_running() {
    local out_dir="$1"
    pgrep -f "output-dir.*$(basename "$out_dir")" &>/dev/null
}

# Check if a chain has completed (final checkpoint saved)
is_complete() {
    local out_dir="$1"
    [[ -f "$out_dir/rapidock_finetuned_final.pt" ]]
}

# Print a compact status summary for one training run
training_status() {
    local label="$1"
    local out_dir="$2"
    local log="$3"

    echo ""
    echo "  ── $label ──"

    if is_complete "$out_dir"; then
        echo "  Status : COMPLETE ✓"
        best_val_line "$log" | sed 's/^/  Best   : /'
        last_epoch_line "$log" | sed 's/^/  Last   : /'
    elif is_running "$out_dir"; then
        echo "  Status : RUNNING"
        last_epoch_line "$log" | sed 's/^/  Epoch  : /'
        best_val_line "$log" | sed 's/^/  Best   : /'
        # Norm alerts in last 5 lines
        grep "NORM ALERT\|KILL\|fatal\|ABORTING" "$log" 2>/dev/null | tail -3 \
            | sed 's/^/  ALERT  : /' || true
    else
        if [[ -f "$out_dir/rapidock_finetuned_best.pt" ]]; then
            echo "  Status : STOPPED (process not found; best checkpoint exists)"
        else
            echo "  Status : NOT STARTED or CRASHED (no checkpoint found)"
        fi
        last_epoch_line "$log" 2>/dev/null | sed 's/^/  Last   : /' || true
    fi
}

# Print full v3 chain phase status
v3_chain_status() {
    echo ""
    echo "  ── v3 chain (all 3 phases) ──"
    for phase in 1 2 3; do
        local pout="$FINETUNED/finetune_peppc_v3_phase${phase}"
        local pfinal="$pout/rapidock_finetuned_final.pt"
        local pbest="$pout/rapidock_finetuned_best.pt"
        if [[ -f "$pfinal" ]]; then
            echo "  P${phase}: COMPLETE ✓"
        elif [[ -f "$pbest" ]]; then
            local is_run; pgrep -f "output-dir.*v3_phase${phase}" &>/dev/null && is_run="RUNNING" || is_run="STOPPED (best exists)"
            echo "  P${phase}: $is_run"
        else
            echo "  P${phase}: not started yet"
        fi
    done
}

# Wait until v3 P3 is complete
wait_for_v3() {
    while true; do
        if is_complete "$V3_P3_OUT"; then
            log "V3 P3 COMPLETE. Proceeding to launch v3b."
            return 0
        fi

        # Status report
        separator
        log "STATUS CHECK — waiting for v3 P3 to finish"
        gpu_status
        v3_chain_status
        echo ""
        # Show last few log lines
        echo "  Recent v3 log:"
        tail -5 "$V3_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (no log)"

        if ! is_running "$V3_P3_OUT" && ! is_complete "$V3_P3_OUT"; then
            # Check if P2 is still running (chain may not have reached P3 yet)
            if is_running "$FINETUNED/finetune_peppc_v3_phase2" || \
               is_running "$FINETUNED/finetune_peppc_v3_phase1"; then
                log "  (v3 P2/P1 still in progress — P3 not started yet)"
            else
                log "WARNING: v3 P3 process not found but not complete — may have crashed!"
                log "  Check $V3_LOG for errors."
            fi
        fi

        log "Next check in $((POLL_INTERVAL / 60)) minutes..."
        sleep "$POLL_INTERVAL"
    done
}

# Launch a chain and wait for it to finish
launch_and_wait() {
    local label="$1"
    local chain_script="$2"
    local p3_out_dir="$3"
    local chain_log="$4"

    separator
    log "LAUNCHING $label"
    log "Script: $chain_script"
    log "Log:    $chain_log"
    echo ""

    # Launch in background, capturing to log
    bash "$chain_script" >> "$chain_log" 2>&1 &
    local pid=$!
    log "$label started with PID $pid"

    # Poll until complete
    while kill -0 "$pid" 2>/dev/null || ! is_complete "$p3_out_dir"; do
        sleep "$POLL_INTERVAL"

        separator
        log "STATUS CHECK — $label running"
        gpu_status
        echo ""

        local vbase="finetune_peppc_$(echo "$label" | tr '[:upper:]' '[:lower:]')"
        for phase in 1 2 3; do
            local pout="$FINETUNED/${vbase}_phase${phase}"
            local pfinal="$pout/rapidock_finetuned_final.pt"
            local pbest="$pout/rapidock_finetuned_best.pt"
            if [[ -f "$pfinal" ]]; then
                echo "  P${phase}: COMPLETE ✓"
            elif [[ -f "$pbest" ]]; then
                echo "  P${phase}: IN PROGRESS (best checkpoint exists)"
            else
                echo "  P${phase}: not started yet"
            fi
        done

        # Show last few log lines
        echo ""
        echo "  Recent log:"
        tail -8 "$chain_log" 2>/dev/null | sed 's/^/    /'

        # Check if process died unexpectedly
        if ! kill -0 "$pid" 2>/dev/null && ! is_complete "$p3_out_dir"; then
            log "WARNING: $label process $pid exited but P3 not complete!"
            log "Check $chain_log for errors."
            log "Waiting one more cycle before proceeding..."
            sleep "$POLL_INTERVAL"
            break
        fi

        log "Next check in $((POLL_INTERVAL / 60)) minutes..."
    done

    separator
    log "$label FINISHED"
    if is_complete "$p3_out_dir"; then
        log "Final checkpoint confirmed: $p3_out_dir/rapidock_finetuned_final.pt"
    else
        log "WARNING: final checkpoint not found — $label may have failed!"
    fi
}

# ─── main ─────────────────────────────────────────────────────────────────────

separator
log "monitor_and_launch.sh starting"
log "Repo: $REPO"
log "Poll interval: $((POLL_INTERVAL / 60)) minutes"
echo ""
echo "  Sequence:"
echo "    1. Wait for v3 P3 to complete (chain_training_v3.sh running independently)"
echo "    2. Launch v3b (chain_training_v3b.sh)"
echo ""
echo "  Status: v2/v2b terminated at ep50 — best checkpoints saved from ep22."
echo "          v4/v5 on hold pending hyperparameter review."
echo ""

# Initial status check
separator
log "INITIAL STATUS"
gpu_status
v3_chain_status

# ── Step 1: wait for v3 P3 ────────────────────────────────────────────────────
wait_for_v3

# ── Step 2: launch v3b ────────────────────────────────────────────────────────
launch_and_wait \
    "v3b" \
    "$SCRIPTS/chain_training_v3b.sh" \
    "$FINETUNED/finetune_peppc_v3b_phase3" \
    "$LOGS/chain_training_v3b.log"

# ── Done ──────────────────────────────────────────────────────────────────────
separator
log "ALL CHAINS COMPLETE (v3 + v3b)"
echo ""
echo "  Checkpoints:"
for v in v3 v3b; do
    for p in 1 2 3; do
        ckpt="$FINETUNED/finetune_peppc_${v}_phase${p}/rapidock_finetuned_best.pt"
        [[ -f "$ckpt" ]] && echo "    ✓ $ckpt" || echo "    ✗ MISSING: $ckpt"
    done
done
echo ""
echo "  Next steps:"
echo "    1. Compare v3 vs v3b: python3 scripts/analyze_training.py \\"
echo "         --phase-dirs $FINETUNED/finetune_peppc_v3b_phase{1,2,3} \\"
echo "         --compare-dirs $FINETUNED/finetune_peppc_v3_phase{1,2,3} \\"
echo "         --out-dir logs/analysis_v3_v3b"
echo "    2. Run benchmark: python3 scripts/benchmark_inference_multi.py"
echo "    3. Update Excel:  python3 scripts/update_training_excel.py"
echo ""
log "monitor_and_launch.sh complete."
