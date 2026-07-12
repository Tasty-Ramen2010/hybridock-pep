#!/bin/bash
set -uo pipefail
cd /home/igem/unknown_software
export PATH="/home/igem/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
export PYTHONPATH="/home/igem/unknown_software/src:$PYTHONPATH"
/home/igem/miniconda3/envs/score-env/bin/python experiments/e22_vina_realpose.py 1 > logs/crystal65_n100/e22_vina.log 2>&1
