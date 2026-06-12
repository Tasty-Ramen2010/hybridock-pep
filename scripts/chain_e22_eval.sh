#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
export PYTHONPATH="/home/igem/unknown_software/src:$PYTHONPATH"
LOG=logs/crystal65_n100/e22_ensemble.log
while ! grep -q "^done:" logs/crystal65_n100/e22_vina.log 2>/dev/null; do sleep 30; done
echo "==== ensemble eval (real poses) $(date) ====" | tee -a "$LOG"
/home/igem/miniconda3/envs/score-env/bin/python scripts/e22_ensemble_eval.py 2>/dev/null | tee -a "$LOG"
echo "==== done $(date) ====" | tee -a "$LOG"
