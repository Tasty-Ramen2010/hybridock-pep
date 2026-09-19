#!/usr/bin/env bash
# Protect the GPU finetune from the CPU benchmark, NOT the other way round.
#
# scripts/memguard.sh kills `train_lastlayer.py` when `free -m` available drops below
# 3000 MB. Each CPU inference worker costs 3.4-4.4 GB (ESM-2 650M + diffusion model on
# CPU), so a couple of extra workers can trip memguard and destroy a 4.5-hour training
# run to keep a benchmark alive. This guard fires FIRST, at a higher threshold, and kills
# the youngest CPU bench worker instead -- the bench is restartable, the training is not.
#
# Kill patterns are bracketed so this script never matches its own command line, and the
# trainer is never a target here.
set -u
THRESH=${BENCHGUARD_THRESH_MB:-4800}   # fire well above memguard's 3000
LOG=/home/igem/unknown_software/logs/benchguard.log
while true; do
  a=$(free -m | awk '/Mem:/{print $7}')
  if [ "$a" -lt "$THRESH" ]; then
    # youngest worker first: least progress lost
    victim=$(ps -eo pid,etimes,cmd --sort=etimes | grep '[i]nference.py' | head -1 | awk '{print $1}')
    if [ -n "${victim:-}" ]; then
      echo "[benchguard $(date '+%m-%d %H:%M:%S')] avail ${a}MB < ${THRESH} -> killing bench worker pid=$victim" >> $LOG
      kill "$victim" 2>/dev/null
      sleep 20
    else
      echo "[benchguard $(date '+%m-%d %H:%M:%S')] avail ${a}MB < ${THRESH} but no bench worker to kill (trainer is NOT a target)" >> $LOG
      sleep 30
    fi
  fi
  sleep 10
done
