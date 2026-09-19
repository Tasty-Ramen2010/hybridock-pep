#!/usr/bin/env bash
# Kill the local trainer (wrapper first) if free RAM drops below 3GB.
# The box took a hypervisor crash; a runaway RSS must not take it down again.
while true; do
  a=$(free -m | awk "/Mem:/{print \$7}")
  if [ "$a" -lt 3000 ]; then
    echo "[memguard $(date +%H:%M:%S)] avail ${a}MB < 3000 -> killing trainer" >> /home/igem/unknown_software/logs/xtal850_local.log
    pkill -f "[l]ongft_local.sh"; sleep 1; pkill -f "[t]rain_lastlayer.py"
    exit 0
  fi
  sleep 10
done
