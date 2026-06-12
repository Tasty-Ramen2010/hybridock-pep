#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
export PYTHONPATH="/home/igem/unknown_software/src:$PYTHONPATH"
PY=/home/igem/miniconda3/envs/score-env/bin/python
while ! grep -q "^done" logs/crystal65_n100/e28_reextract.log 2>/dev/null; do sleep 20; done
echo "98 extracted, running Rosetta per-term (cropped relax)..." 
$PY scripts/e43_rosetta_terms.py b98 > logs/crystal65_n100/e43_98.log 2>&1
