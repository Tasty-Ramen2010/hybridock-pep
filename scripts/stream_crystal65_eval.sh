#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
PY=/home/igem/miniconda3/envs/score-env/bin/python
SLOG=logs/crystal65_n100/eval_stream.log
for target in 35 50 65; do
  while [ "$(grep -c 'best=' logs/crystal65_n100/gen.log)" -lt "$target" ]; do
    if grep -q "gen done" logs/crystal65_n100/gen.log; then break; fi
    sleep 60
  done
  echo "==== eval @ $(grep -c 'best=' logs/crystal65_n100/gen.log) complexes  $(date) ====" >> "$SLOG"
  $PY experiments/e19_realpose_eval.py 2>/dev/null | grep -vE "processed" >> "$SLOG"
done
echo "==== STREAM DONE $(date) ====" >> "$SLOG"
