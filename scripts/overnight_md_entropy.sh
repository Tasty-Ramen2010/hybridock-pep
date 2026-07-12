#!/bin/bash
# OVERNIGHT: test Ram's 100ps Ramachandran/MD-entropy hypothesis on all 65 crystal complexes.
# GPU-polite: waits for crystal65 pose generation (and any PfLDH dock) to finish first.
set -uo pipefail
cd /home/igem/unknown_software
RAPID=/home/igem/miniconda3/envs/rapidock/bin/python   # has CUDA + openmm
SCORE=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/crystal65_n100/overnight_md.log
mkdir -p logs/crystal65_n100
echo "==== overnight MD-entropy start $(date) ====" | tee -a "$LOG"
# wait for pose generation to finish (frees GPU)
while ! grep -q "gen done" logs/crystal65_n100/gen.log 2>/dev/null; do sleep 120; done
# yield to PfLDH production dock if it appears
while pgrep -f "runs/pfldh_lisdaeleaifeadc" >/dev/null 2>&1; do sleep 120; done
echo "[gpu] free, running 100ps MD entropy on 65 crystal complexes $(date)" | tee -a "$LOG"
# e18v2_features computes ds_dih + rmsf_ratio (100ps bound+free MD), checkpoint-safe, crystal mode
$SCORE experiments/e18v2_features.py 100 cr >> "$LOG" 2>&1
echo "[md] features done $(date)" | tee -a "$LOG"
# eval: does MD-entropy add to geometry?
echo "==== MD-ENTROPY EVAL $(date) ====" | tee -a "$LOG"
$SCORE experiments/e18v2_md_eval.py 2>/dev/null | tee -a "$LOG"
echo "==== overnight MD-entropy DONE $(date) ====" | tee -a "$LOG"
