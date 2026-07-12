#!/usr/bin/env bash
# Chain the charged PPIKB cloud campaign (e324) to start when the PDBbind one (e323) frees the GPU.
# Single RTX 5070 → strictly sequential (CLAUDE.md). Poll for the e323 process to exit, then launch e324.
set -u
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/e324_ppikb_clouds.log

echo "[chain] waiting for e323 (PDBbind charged clouds) to finish before starting e324 (PPIKB)..." >> "$LOG"
while pgrep -f e323_charged_cloud_campaign.py >/dev/null 2>&1; do
    sleep 120
done
echo "[chain] e323 finished at $(date -u +%FT%TZ); launching e324 PPIKB charged clouds" >> "$LOG"
OMP_NUM_THREADS=1 nohup "$PY" experiments/e324_ppikb_charged_clouds.py >> "$LOG" 2>&1 &
echo "[chain] e324 launched PID $!" >> "$LOG"
