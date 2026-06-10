#!/bin/bash
# N=500 generation on the 57 diverse gen_n100 complexes — density-hypothesis test.
# GPU-polite: yields to any live PfLDH production dock. Resume-safe.
set -uo pipefail
cd /home/igem/unknown_software
SCORE=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/gen_n500/run.log
mkdir -p logs/gen_n500
echo "==== gen_n500 start $(date) ====" | tee -a "$LOG"

# Yield GPU to any live PfLDH production dock before starting.
while pgrep -f "runs/pfldh_lisdaeleaifeadc" >/dev/null 2>&1; do
  echo "[gpu] PfLDH production running, waiting $(date)" | tee -a "$LOG"; sleep 60
done
echo "[gpu] free at $(date)" | tee -a "$LOG"

# N=500, timeout 1500s/complex (5x the N=100 budget). Resume-safe (skips done).
$SCORE scripts/generate_confidence_poses.py \
  --input data/gen_n500_57.csv \
  --out-dir logs/gen_n500 \
  --n-samples 500 \
  --timeout 1500 \
  --seed 42 >> "$LOG" 2>&1

echo "==== gen_n500 done $(date) ====" | tee -a "$LOG"
