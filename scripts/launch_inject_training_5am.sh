#!/usr/bin/env bash
# launch_inject_training_5am.sh — GPU-gated sequential launch of inject training runs
#
# Checks GPU occupancy via nvidia-smi before proceeding.
# Polls every 5 min for up to 90 min. If GPU stays busy, aborts with an error log.
# If GPU is free, launches inject-v5c then inject-pretrained sequentially.
#
# Scheduled via system crontab for 05:00 local time.
# Log: logs/inject_training_5am.log
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/logs/inject_training_5am.log"
mkdir -p "$REPO/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Inject training launcher started ==="
log "Checking GPU availability..."

GPU_FREE=0
MAX_WAIT_MIN=90
POLL_INTERVAL_SEC=300
WAITED=0

while true; do
    # Check if any process is using the GPU (memory > 0 MiB in-use)
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)

    if [[ "$GPU_PROCS" -eq 0 && "$GPU_MEM_USED" -lt 500 ]]; then
        log "GPU is free (${GPU_MEM_USED} MiB used, 0 compute processes). Proceeding."
        GPU_FREE=1
        break
    fi

    log "GPU busy (${GPU_MEM_USED} MiB used, ${GPU_PROCS} compute process(es)). Waited ${WAITED}/${MAX_WAIT_MIN} min."
    if [[ "$WAITED" -ge "$MAX_WAIT_MIN" ]]; then
        log "ERROR: GPU still busy after ${MAX_WAIT_MIN} min. Aborting inject training launch."
        log "Re-run manually: nohup bash scripts/launch_inject_training_5am.sh &"
        exit 1
    fi

    sleep "$POLL_INTERVAL_SEC"
    WAITED=$(( WAITED + POLL_INTERVAL_SEC / 60 ))
done

# ─────────────────────────────────────────────────────────────────────────────
# Run 1: inject-v5c  (starts from v5c-P2-best)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log ">>> Launching inject-v5c  (logs/chain_training_inject_v5c.log)"
nohup bash "$REPO/scripts/chain_training_inject_v5c.sh" \
    >> "$REPO/logs/chain_training_inject_v5c.log" 2>&1
log "<<< inject-v5c COMPLETE (exit $?)"

# ─────────────────────────────────────────────────────────────────────────────
# Run 2: inject-pretrained  (starts from rapidock_global.pt)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log ">>> Launching inject-pretrained  (logs/chain_training_inject_pretrained.log)"
nohup bash "$REPO/scripts/chain_training_inject_pretrained.sh" \
    >> "$REPO/logs/chain_training_inject_pretrained.log" 2>&1
log "<<< inject-pretrained COMPLETE (exit $?)"

log ""
log "=== All inject training runs complete ==="
log "Best checkpoints:"
log "  inject-v5c:         $REPO/third_party/RAPiDock_finetuned/finetune_inject_v5c_phase1/rapidock_finetuned_best.pt"
log "  inject-pretrained:  $REPO/third_party/RAPiDock_finetuned/finetune_inject_pretrained_phase2/rapidock_finetuned_best.pt"
