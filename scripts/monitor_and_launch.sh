#!/usr/bin/env bash
# monitor_and_launch.sh
#
# Monitors v2 and v2b P3 training, reports GPU + training status every hour,
# then automatically launches v3 → v4 → v5 in sequence when each finishes.
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

V2_P3_OUT="$FINETUNED/finetune_peppc_v2_phase3"
V2B_P3_OUT="$FINETUNED/finetune_peppc_v2b_phase3"
V2_LOG="$LOGS/chain_training_v2.log"
V2B_LOG="$LOGS/chain_training_v2b.log"

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

# Wait until both v2 and v2b P3 are complete
wait_for_v2_v2b() {
    while true; do
        local v2_done=false v2b_done=false

        is_complete "$V2_P3_OUT"  && v2_done=true
        is_complete "$V2B_P3_OUT" && v2b_done=true

        if $v2_done && $v2b_done; then
            log "Both v2 P3 and v2b P3 COMPLETE. Proceeding to launch v3."
            return 0
        fi

        # Status report
        separator
        log "STATUS CHECK — waiting for v2 + v2b to finish"
        gpu_status
        training_status "v2  P3" "$V2_P3_OUT"  "$V2_LOG"
        training_status "v2b P3" "$V2B_P3_OUT" "$V2B_LOG"
        echo ""

        if ! $v2_done && ! is_running "$V2_P3_OUT"; then
            log "WARNING: v2 P3 process not found but not complete — may have crashed!"
        fi
        if ! $v2b_done && ! is_running "$V2B_P3_OUT"; then
            log "WARNING: v2b P3 process not found but not complete — may have crashed!"
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

        local all_outs=("$FINETUNED/finetune_peppc_$(echo "$label" | tr '[:upper:]' '[:lower:]')_phase1"
                        "$FINETUNED/finetune_peppc_$(echo "$label" | tr '[:upper:]' '[:lower:]')_phase2"
                        "$FINETUNED/finetune_peppc_$(echo "$label" | tr '[:upper:]' '[:lower:]')_phase3")
        for phase in 1 2 3; do
            local pout="${all_outs[$((phase-1))]}"
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
echo "  Waiting for:  v2 P3  ($V2_P3_OUT)"
echo "               v2b P3 ($V2B_P3_OUT)"
echo ""
echo "  NOTE: v3 is assumed to already be running independently."
echo "  Then launching in sequence:"
echo "    1. v4 ($SCRIPTS/chain_training_v4.sh)"
echo "    2. v5 ($SCRIPTS/chain_training_v5.sh)"
echo ""

# Initial status check
separator
log "INITIAL STATUS"
gpu_status
training_status "v2  P3" "$V2_P3_OUT"  "$V2_LOG"
training_status "v2b P3" "$V2B_P3_OUT" "$V2B_LOG"

# ── Step 1: wait for v2 + v2b ─────────────────────────────────────────────────
wait_for_v2_v2b

# ── Step 2: launch v4 ─────────────────────────────────────────────────────────
# (v3 is already running independently in a separate process)
launch_and_wait \
    "v4" \
    "$SCRIPTS/chain_training_v4.sh" \
    "$FINETUNED/finetune_peppc_v4_phase3" \
    "$LOGS/chain_training_v4.log"

# ── Step 3: launch v5 ─────────────────────────────────────────────────────────
launch_and_wait \
    "v5" \
    "$SCRIPTS/chain_training_v5.sh" \
    "$FINETUNED/finetune_peppc_v5_phase3" \
    "$LOGS/chain_training_v5.log"

# ── Done ──────────────────────────────────────────────────────────────────────
separator
log "ALL CHAINS COMPLETE (v4 + v5)"
echo ""
echo "  Checkpoints:"
for v in v3 v4 v5; do
    for p in 1 2 3; do
        ckpt="$FINETUNED/finetune_peppc_${v}_phase${p}/rapidock_finetuned_best.pt"
        [[ -f "$ckpt" ]] && echo "    ✓ $ckpt" || echo "    ✗ MISSING: $ckpt"
    done
done
echo ""
echo "  Next steps:"
echo "    1. Run benchmark:  python3 scripts/benchmark_inference_multi.py"
echo "    2. Update Excel:   python3 scripts/update_training_excel.py"
echo "    3. Update report:  logs/finetuning_analysis_report.md"
echo ""
log "monitor_and_launch.sh complete."
