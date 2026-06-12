#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/crystal65_n100/multimodal_eval.log
while [ ! -f /tmp/esm_affinity.json ]; do sleep 30; done
sleep 5
echo "==== multimodal/nonlinear eval $(date) ====" | tee -a "$LOG"
$PY scripts/e20_multimodal_eval.py 2>/dev/null | tee -a "$LOG"
echo "==== done $(date) ====" | tee -a "$LOG"
