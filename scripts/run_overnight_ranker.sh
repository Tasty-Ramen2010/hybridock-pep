#!/usr/bin/env bash
# Waits for confidence_training_campaign.py to finish, then runs overnight
# ranker comparison on GPU with VRAM capped at 90%.
#
# Usage: nohup bash scripts/run_overnight_ranker.sh &
#        tail -f logs/training_campaign/ranker_comparison_overnight.log

LOG=logs/training_campaign/ranker_comparison_overnight.log
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "$(date '+%H:%M:%S') Waiting for confidence_training_campaign.py to finish..." | tee "$LOG"

# Poll until campaign process is gone
while pgrep -f "confidence_training_campaign" > /dev/null; do
    sleep 60
done

echo "$(date '+%H:%M:%S') Campaign finished. GPU should be free." | tee -a "$LOG"
sleep 10  # brief settle time

# Check GPU free
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

echo "$(date '+%H:%M:%S') Launching overnight ranker comparison (GPU, --overnight)..." | tee -a "$LOG"

PYTHONPATH="$REPO" \
    "$HOME/miniconda3/envs/rapidock/bin/python" \
    scripts/ranker_comparison.py \
    --overnight \
    --device auto \
    2>&1 | tee -a "$LOG"

echo "$(date '+%H:%M:%S') Done." | tee -a "$LOG"
