#!/usr/bin/env bash
# Queue the NEUTRAL control campaign (e326) after both GPU campaigns (e323 PDBbind-charged, e324 PPIKB-charged)
# finish. Single RTX 5070 → strictly sequential. Poll until neither is running, then launch e326.
set -u
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/e326_neutral_clouds.log

echo "[chain] waiting for e323 + e324 to finish before starting e326 (neutral control)..." >> "$LOG"
while pgrep -f 'e323_charged_cloud_campaign.py|e324_ppikb_charged_clouds.py' >/dev/null 2>&1; do
    sleep 120
done
echo "[chain] charged campaigns done at $(date -u +%FT%TZ); launching e326 neutral control" >> "$LOG"
OMP_NUM_THREADS=1 nohup "$PY" experiments/e326_neutral_pdbbind_clouds.py >> "$LOG" 2>&1 &
echo "[chain] e326 launched PID $!" >> "$LOG"
