#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
LOG=logs/crystal65_n100/eval.log
# wait for generation to finish (the launcher writes the 'gen done' line)
while ! grep -q "crystal65 N=100 gen done" logs/crystal65_n100/gen.log 2>/dev/null; do sleep 60; done
echo "==== gen complete, running deployment-faithful eval $(date) ====" | tee -a "$LOG"
/home/igem/miniconda3/envs/score-env/bin/python experiments/e19_realpose_eval.py >> "$LOG" 2>&1
echo "==== eval done $(date) ====" | tee -a "$LOG"
