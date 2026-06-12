#!/bin/bash
# Fair FlexPepDock-style comparison: ref2015 interface energy AFTER FastRelax (CPU).
set -uo pipefail
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
LOG=logs/crystal65_n100/rosetta_relax.log
cp -f /tmp/rosetta_ref2015.json /tmp/rosetta_ref2015_norelax.json 2>/dev/null || true
rm -f /tmp/rosetta_ref2015.json
echo "==== Rosetta ref2015 + FastRelax (fair) start $(date) ====" | tee -a "$LOG"
$PY scripts/rosetta_ref2015_eval.py --relax >> "$LOG" 2>&1
echo "==== done $(date) ====" | tee -a "$LOG"
