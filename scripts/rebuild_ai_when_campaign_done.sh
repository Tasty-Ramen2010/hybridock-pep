#!/bin/bash
# Auto-rebuild the AI/deployment affinity model once the real-pose campaigns finish writing training data.
# The AI model (data/affinity_ai_nofix.joblib) trains on e93 + e154 + e176_{short,long,vlong} real poses;
# the short campaign is the last one still appending. When it (and any vlong finisher) stop, rebuild + log.
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
LOG=runs/ai_rebuild.log
echo "$(date) === AI-rebuild watcher START (waiting for real-pose campaigns) ===" >> $LOG
# wait until no e176 real-pose queue worker remains
while pgrep -f "e176_realpose_queue.py" >/dev/null 2>&1; do sleep 120; done
echo "$(date) campaigns done — rebuilding AI model" >> $LOG
$PY scripts/e204_build_ai_nofix.py >> $LOG 2>&1
echo "$(date) === AI model rebuilt (data/affinity_ai_nofix.joblib) ===" >> $LOG
# also refresh the crystal model is NOT needed (fixed 925); selectivity e193 dock continues separately.
