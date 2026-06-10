#!/bin/bash
# Auto-launch the BUNS test once MM-GBSA finishes (avoids CPU contention).
set -uo pipefail
cd /home/igem/unknown_software
RAPID=/home/igem/miniconda3/envs/rapidock/bin/python
LOG=/tmp/buns_test.log

echo "[buns-chain] waiting for MM-GBSA to finish $(date)"
# Wait for the MM-GBSA completion marker (or its process to exit).
while ! grep -q "Saved → " /tmp/mmgbsa_rerank.log 2>/dev/null; do
  pgrep -f mmgbsa_rerank_top5.py >/dev/null 2>&1 || {
    # process gone but no Saved marker → still proceed after a grace check
    sleep 10
    grep -q "Saved → " /tmp/mmgbsa_rerank.log 2>/dev/null && break
    echo "[buns-chain] mmgbsa process exited without Saved marker; proceeding anyway"
    break
  }
  sleep 60
done

echo "[buns-chain] MM-GBSA done, launching BUNS $(date)"
PYTHONUNBUFFERED=1 $RAPID -u scripts/buns_test.py > "$LOG" 2>&1
echo "[buns-chain] BUNS done $(date)"
