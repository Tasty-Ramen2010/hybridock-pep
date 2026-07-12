#!/bin/bash
PY=/home/igem/miniconda3/envs/score-env/bin/python
cd /home/igem/unknown_software
for cx in 1PPF_E_I 1CHO_EFG_I 1R0R_E_I 3SGB_E_I 1AO7_ABC_DE; do
  echo "### $cx $(date +%H:%M) ###"
  $PY experiments/e51_skempi_ddg.py $cx 130 2>&1 | grep -vi "warn\|core\.\|basic\.\|protocols\.\|apps\."
done
echo "### CAMPAIGN DONE $(date +%H:%M) ###"
