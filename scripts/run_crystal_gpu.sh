#!/usr/bin/env bash
# GPU run of the remaining MM-GBSA variants (run only when the GPU is free —
# never contend with a production dock). Baseline already completed on CPU.
set -u
cd /home/igem/unknown_software
PY=python3
LOG=/tmp/crystal_gpu.log
: > "$LOG"

run() {
  local name="$1"; shift
  echo "==================== $name  $(date '+%H:%M:%S') ====================" | tee -a "$LOG"
  $PY -u scripts/score_crystal_benchmark.py --gpu \
      --out "data/benchmark_crystal_scored_${name}.json" "$@" >> "$LOG" 2>&1
  echo "-------------------- $name done $(date '+%H:%M:%S') --------------------" | tee -a "$LOG"
}

run 3traj   --3traj
run ie      --ie
run ie3traj --ie --3traj
echo "GPU VARIANTS COMPLETE $(date '+%H:%M:%S')" | tee -a "$LOG"
