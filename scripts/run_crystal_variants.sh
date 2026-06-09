#!/usr/bin/env bash
# Sequential crystal-benchmark scoring across MM-GBSA variants (overhaul validation).
# Runs CPU-only, one variant at a time (no CPU oversubscription), each writing its
# own incremental JSON so a crash/kill loses at most the in-flight variant.
# Order = increasing cost: baseline → 3traj (cheap) → IE (trajectory, slow) → IE+3traj.
set -u
cd /home/igem/unknown_software
PY=python3
DATA=data
LOG=/tmp/crystal_variants.log

run() {
  local name="$1"; shift
  echo "==================== $name  $(date '+%H:%M:%S') ====================" | tee -a "$LOG"
  $PY -u scripts/score_crystal_benchmark.py --out "$DATA/benchmark_crystal_scored_${name}.json" "$@" \
      >> "$LOG" 2>&1
  echo "-------------------- $name done $(date '+%H:%M:%S') --------------------" | tee -a "$LOG"
}

: > "$LOG"
run baseline                       # εin=1, single-traj, no entropy
run 3traj      --3traj             # three-trajectory (unbound relaxation)
run ie         --ie                # interaction entropy
run ie3traj    --ie --3traj        # full SOTA-style config
echo "ALL VARIANTS COMPLETE $(date '+%H:%M:%S')" | tee -a "$LOG"
