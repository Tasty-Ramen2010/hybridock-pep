#!/usr/bin/env bash
# monitor_and_launch.sh
#
# Sequential pipeline: v3b → v4n → v5n
#   Monitors each chain, auto-launches next when previous P3 completes.
#   Reports GPU + training status every hour.
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
V4N_LOG="$LOGS/chain_training_v4new.log"
V5N_LOG="$LOGS/chain_training_v5new.log"

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

# Per-experiment chain status display
chain_status() {
    local label="$1"   # e.g. "v3b"
    local log_file="$2"
    local chain_pattern="$3"  # pgrep pattern for the chain script
    shift 3
    local phase_dirs=("$@")  # array of output dirs

    echo ""
    echo "  ── ${label} chain progress ──"
    for i in "${!phase_dirs[@]}"; do
        local phase=$((i + 1))
        local pout="${phase_dirs[$i]}"
        local pfinal="$pout/rapidock_finetuned_final.pt"
        local pbest="$pout/rapidock_finetuned_best.pt"
        local nbest; nbest=$(ls "$pout"/rapidock_finetuned_epoch*.pt 2>/dev/null | wc -l)
        if [[ -f "$pfinal" ]]; then
            echo "  P${phase}: COMPLETE ✓  ($(du -sh "$pfinal" | cut -f1))"
        elif is_running "$pout"; then
            local best_line; best_line=$(grep "✓ New best" "$log_file" 2>/dev/null | tail -1 || echo "(no best yet)")
            echo "  P${phase}: RUNNING"
            echo "           Last:   $(grep "^Epoch" "$log_file" 2>/dev/null | tail -1 || echo "(starting)")"
            echo "           Best:   $best_line"
            echo "           Saved epoch ckpts: $nbest"
            grep "SPIKE LR" "$log_file" 2>/dev/null | tail -2 | sed 's/^/           SPIKE: /' || true
        elif [[ -f "$pbest" ]]; then
            echo "  P${phase}: STOPPED (best checkpoint exists; process not found)"
            echo "           $(ls -lh "$pbest" | awk '{print $5, $7, $8}')"
        else
            echo "  P${phase}: waiting (not started yet)"
        fi
    done
}

# ─── main ─────────────────────────────────────────────────────────────────────

separator
log "monitor_and_launch.sh starting — v3b → v4n sequential pipeline"
log "Repo: $REPO"
log "Poll interval: $((POLL_INTERVAL / 60)) minutes"
log "v3b log: $V3B_LOG"
log "v4n log: $V4N_LOG"
echo ""
echo "  History:"
echo "    v2/v2b: terminated ep50, best from ep22 (val~47.57)"
echo "    v3:     terminated: P1 complete (best ep16 val=39.005); P2 killed ep18 (best ep16 val=39.698)"
echo "    v3b:    RUNNING — stable controlled spec (cosine + adaptive spike LR)"
echo "    v4n:    queued — careful mechanistic probe (LR 8e-6/2e-6/5e-6; layerwise 0.40/0.15/0.03)"
echo "    v5n:    queued — ultra-conservative manifold preservation (LR 5e-6/1e-6/3e-6; layerwise 1.0/0.25/0.08/0.02)"
echo ""

# Initial status
separator
log "INITIAL STATUS"
gpu_status
chain_status "v3b" "$V3B_LOG" "chain_training_v3b" \
    "$FINETUNED/finetune_peppc_v3b_phase1" \
    "$FINETUNED/finetune_peppc_v3b_phase2" \
    "$FINETUNED/finetune_peppc_v3b_phase3"

# If v4n already started show it too
if [[ -d "$FINETUNED/finetune_peppc_v4n_phase1" ]] || \
   pgrep -f "chain_training_v4new" &>/dev/null; then
    chain_status "v4n" "$V4N_LOG" "chain_training_v4new" \
        "$FINETUNED/finetune_peppc_v4n_phase1" \
        "$FINETUNED/finetune_peppc_v4n_phase2" \
        "$FINETUNED/finetune_peppc_v4n_phase3"
fi

# If v5n already started show it too
if [[ -d "$FINETUNED/finetune_peppc_v5n_phase1" ]] || \
   pgrep -f "chain_training_v5new" &>/dev/null; then
    chain_status "v5n" "$V5N_LOG" "chain_training_v5new" \
        "$FINETUNED/finetune_peppc_v5n_phase1" \
        "$FINETUNED/finetune_peppc_v5n_phase2" \
        "$FINETUNED/finetune_peppc_v5n_phase3"
fi

# ── Monitor loop ───────────────────────────────────────────────────────────────
while true; do

    # ── Check if everything is done ─────────────────────────────────────────
    if is_complete "$FINETUNED/finetune_peppc_v5n_phase3"; then
        separator
        log "ALL COMPLETE: v3b + v4n + v5n ✓"
        gpu_status
        echo ""
        for exp in v3b v4n v5n; do
            echo "  ${exp} checkpoints:"
            for p in 1 2 3; do
                ckpt="$FINETUNED/finetune_peppc_${exp}_phase${p}/rapidock_finetuned_best.pt"
                [[ -f "$ckpt" ]] && echo "    ✓ P${p}: $ckpt" || echo "    ✗ P${p}: MISSING"
            done
            echo ""
        done
        echo "  Next steps:"
        echo "    1. Compare v3b / v4n / v5n:"
        echo "         python3 scripts/analyze_training.py \\"
        echo "           --phase-dirs $FINETUNED/finetune_peppc_v5n_phase{1,2,3} \\"
        echo "           --compare-dirs $FINETUNED/finetune_peppc_v3b_phase{1,2,3} \\"
        echo "                          $FINETUNED/finetune_peppc_v4n_phase{1,2,3} \\"
        echo "           --out-dir logs/analysis_v3b_v4n_v5n"
        echo "    2. Benchmark: python3 scripts/benchmark_inference_multi.py"
        echo "    3. Update AI doc: append to docs/ai_training_guide_peppc.md"
        log "monitor_and_launch.sh complete."
        exit 0
    fi

    sleep "$POLL_INTERVAL"

    separator
    log "STATUS CHECK"
    gpu_status

    # ── v3b status ────────────────────────────────────────────────────────────
    chain_status "v3b" "$V3B_LOG" "chain_training_v3b" \
        "$FINETUNED/finetune_peppc_v3b_phase1" \
        "$FINETUNED/finetune_peppc_v3b_phase2" \
        "$FINETUNED/finetune_peppc_v3b_phase3"

    echo ""
    echo "  Recent v3b log (last 5 lines):"
    tail -5 "$V3B_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (no log yet)"

    # ── Launch v4n if v3b just finished ──────────────────────────────────────
    if is_complete "$FINETUNED/finetune_peppc_v3b_phase3"; then
        if ! pgrep -f "chain_training_v4new" &>/dev/null && \
           ! is_complete "$FINETUNED/finetune_peppc_v4n_phase3"; then
            separator
            log "V3B COMPLETE ✓ — launching v4n now"
            log "  Log: $V4N_LOG"
            nohup bash "$REPO/experiments/chain_training_v4new.sh" \
                >> "$V4N_LOG" 2>&1 &
            log "  v4n PID: $!"
        fi
    fi

    # ── v4n status (if started) ───────────────────────────────────────────────
    if [[ -d "$FINETUNED/finetune_peppc_v4n_phase1" ]] || \
       pgrep -f "chain_training_v4new" &>/dev/null; then
        chain_status "v4n" "$V4N_LOG" "chain_training_v4new" \
            "$FINETUNED/finetune_peppc_v4n_phase1" \
            "$FINETUNED/finetune_peppc_v4n_phase2" \
            "$FINETUNED/finetune_peppc_v4n_phase3"

        echo ""
        echo "  Recent v4n log (last 5 lines):"
        tail -5 "$V4N_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (no log yet)"

        # crash detection for v4n
        v4n_any_active=false
        for phase in 1 2 3; do
            is_running "$FINETUNED/finetune_peppc_v4n_phase${phase}" && \
                v4n_any_active=true && break || true
        done
        pgrep -f "chain_training_v4new" &>/dev/null && v4n_any_active=true || true

        if ! $v4n_any_active && ! is_complete "$FINETUNED/finetune_peppc_v4n_phase3"; then
            log "WARNING: v4n chain process not found but P3 not complete — may have crashed!"
            log "  Check $V4N_LOG for errors."
            log "  To resume: nohup bash experiments/chain_training_v4new.sh >> $V4N_LOG 2>&1 &"
        fi
    fi

    # ── Launch v5n if v4n just finished ──────────────────────────────────────
    if is_complete "$FINETUNED/finetune_peppc_v4n_phase3"; then
        if ! pgrep -f "chain_training_v5new" &>/dev/null && \
           ! is_complete "$FINETUNED/finetune_peppc_v5n_phase3"; then
            separator
            log "V4N COMPLETE ✓ — launching v5n now"
            log "  Log: $V5N_LOG"
            nohup bash "$REPO/experiments/chain_training_v5new.sh" \
                >> "$V5N_LOG" 2>&1 &
            log "  v5n PID: $!"
        fi
    fi

    # ── v5n status (if started) ───────────────────────────────────────────────
    if [[ -d "$FINETUNED/finetune_peppc_v5n_phase1" ]] || \
       pgrep -f "chain_training_v5new" &>/dev/null; then
        chain_status "v5n" "$V5N_LOG" "chain_training_v5new" \
            "$FINETUNED/finetune_peppc_v5n_phase1" \
            "$FINETUNED/finetune_peppc_v5n_phase2" \
            "$FINETUNED/finetune_peppc_v5n_phase3"

        echo ""
        echo "  Recent v5n log (last 5 lines):"
        tail -5 "$V5N_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (no log yet)"

        # crash detection for v5n
        v5n_any_active=false
        for phase in 1 2 3; do
            is_running "$FINETUNED/finetune_peppc_v5n_phase${phase}" && \
                v5n_any_active=true && break || true
        done
        pgrep -f "chain_training_v5new" &>/dev/null && v5n_any_active=true || true

        if ! $v5n_any_active && ! is_complete "$FINETUNED/finetune_peppc_v5n_phase3"; then
            log "WARNING: v5n chain process not found but P3 not complete — may have crashed!"
            log "  Check $V5N_LOG for errors."
            log "  To resume: nohup bash experiments/chain_training_v5new.sh >> $V5N_LOG 2>&1 &"
        fi
    fi

    # crash detection for v3b (only while v3b still in progress)
    if ! is_complete "$FINETUNED/finetune_peppc_v3b_phase3"; then
        v3b_any_active=false
        for phase in 1 2 3; do
            is_running "$FINETUNED/finetune_peppc_v3b_phase${phase}" && \
                v3b_any_active=true && break || true
        done
        pgrep -f "chain_training_v3b" &>/dev/null && v3b_any_active=true || true

        if ! $v3b_any_active; then
            log "WARNING: v3b chain process not found but P3 not complete — may have crashed!"
            log "  Check $V3B_LOG for errors."
            log "  To resume: nohup bash experiments/chain_training_v3b.sh >> $V3B_LOG 2>&1 &"
        fi
    fi

    log "Next check in $((POLL_INTERVAL / 60)) minutes..."
done
