#!/usr/bin/env bash
set -u
D="$1"; E="${2:-10}"; R="${3:-5}"
while true; do
  mapfile -t e < <(ls "$D" 2>/dev/null | grep -oE "epoch[0-9]+" | grep -oE "[0-9]+" | sort -n | uniq)
  n=${#e[@]}
  for i in "${!e[@]}"; do
    x=${e[$i]}
    [ $((10#$x % E)) -eq 0 ] && continue
    [ $((n - i)) -le $R ] && continue
    rm -f "$D"/*epoch"$x".pt
  done
  sleep 600
done
