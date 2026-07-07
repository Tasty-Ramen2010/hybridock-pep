#!/usr/bin/env bash
# Run the relative charge-morph FEP (e333) after the corrected decouple FEP (e332). Single RTX 5070 → sequential.
set -u
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/openmm-env/bin/python
LOG=logs/e333_relative.log
echo "[chain] waiting for e332 to finish before starting e333..." >> "$LOG"
while pgrep -f e332_g1_charged_corrected.py >/dev/null 2>&1; do sleep 60; done
echo "[chain] e332 done at $(date -u +%FT%TZ); launching e333 relative charge-morph" >> "$LOG"
nohup "$PY" scripts/e333_relative_charge_morph.py >> "$LOG" 2>&1 &
echo "[chain] e333 launched PID $!" >> "$LOG"
