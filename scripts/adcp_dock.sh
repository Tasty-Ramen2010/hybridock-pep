#!/usr/bin/env bash
# ADCP docking on the leak-free RecentSet benchmark (the baseline RAPiDock's paper cites most).
# Settings: 20 replicas x 2.5M steps (ADCP's standard; 171 s for a 10-mer on 8 cores).
# Waits for the GPU benchmark and the prep to finish first: running beside the GPU arms
# slowed them from 27 to ~45 s/complex, and they are the headline result.
# Output: runs/adcp/<name>/dock_ranked_N.pdb (ADCP's own affinity ranking -> a real top-1).
# Usage: adcp_dock.sh [parallel_jobs]   (default 3 x 8 cores = 24)
set -u
R=/home/igem/unknown_software
LOG=$R/logs/adcp_dock.log
P=${1:-3}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

if [ "${NO_GPU_WAIT:-0}" = 1 ]; then
  # CPU-only; run beside the GPU arms at low parallelism instead of idling for ~6 h.
  say "NO_GPU_WAIT=1: waiting only for ADCP prep"
  while pgrep -f 'adcp_prep\.sh' >/dev/null; do sleep 120; done
else
  say "waiting for GPU benchmark + ADCP prep to finish"
  while pgrep -f 'recentset_bench\.sh$' >/dev/null || pgrep -f 'adcp_prep\.sh' >/dev/null; do
    sleep 120
  done
fi
say "starting ADCP: $P parallel jobs"

dock_one() {
  A=/home/igem/ADFRsuite_x86_64Linux_1.0/bin
  IFS=$'\t' read -r NAME REC PEP SEQ <<< "$1"
  d=/home/igem/unknown_software/runs/adcp/$NAME
  [ -s "$d/tgt.trg" ] || { echo "  $NAME: no target, skipped"; return 0; }
  [ -s "$d/dock_ranked_1.pdb" ] && { echo "  $NAME: already docked"; return 0; }
  s=$(date +%s)
  ( cd "$d" && "$A/adcp" -t tgt.trg -s "$(echo "$SEQ" | tr 'A-Z' 'a-z')" \
      -N 20 -n 2500000 -c 8 -o dock -O > dock.log 2>&1 )
  if [ -s "$d/dock_ranked_1.pdb" ]; then
    echo "  $NAME: ok $(ls "$d"/dock_ranked_*.pdb | wc -l) modes in $(( $(date +%s)-s ))s"
  else
    echo "  $NAME: adcp FAILED"
  fi
}
export -f dock_one

xargs -a /tmp/claude-1000/adcp_jobs.tsv -d '\n' -P "$P" -I{} bash -c 'dock_one "$@"' _ {} >> "$LOG" 2>&1
say "ADCP_DONE ok=$(grep -c ': ok' "$LOG") failed=$(grep -c 'FAILED' "$LOG")"
