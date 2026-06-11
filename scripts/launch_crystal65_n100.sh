#!/bin/bash
# Generate RAPiDock N=100 poses for the 65 Kd-labeled crystal complexes.
# Deployment-faithful: lets us recompute E18/E19 features on REAL diffusion poses
# instead of crystal oracle poses. GPU-polite (yields to PfLDH production dock).
set -uo pipefail
cd /home/igem/unknown_software
SCORE=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/crystal65_n100/gen.log
mkdir -p logs/crystal65_n100
echo "==== crystal65 N=100 gen start $(date) ====" | tee -a "$LOG"
# yield GPU to any live PfLDH production dock
while pgrep -f "runs/pfldh_lisdaeleaifeadc" >/dev/null 2>&1; do echo "[gpu] waiting for PfLDH $(date)" | tee -a "$LOG"; sleep 60; done
echo "[gpu] free at $(date)" | tee -a "$LOG"
$SCORE scripts/generate_confidence_poses.py \
  --input data/crystal65_gen.csv \
  --out-dir logs/crystal65_n100 \
  --n-samples 100 >> "$LOG" 2>&1
echo "==== crystal65 N=100 gen done $(date) ====" | tee -a "$LOG"
