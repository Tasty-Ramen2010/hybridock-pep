#!/usr/bin/env bash
# switch_to_adapter.sh — atomically hand the GPU from the x0 run to the frozen-base ADAPTER run.
# Both are systemd --user services (survive disconnect via linger). Only ONE trains at a time
# (RTX 5070 12GB can't hold two full-model trainings). Run this when ready to switch.
set -e
echo "[1/4] stopping x0 run (keeps its checkpoints)..."
systemctl --user stop rapidock-x0.service rapidock-x0-meter.service 2>/dev/null || true
sleep 5
echo "[2/4] GPU after stop:"; nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo "[3/4] enabling + starting ADAPTER (frozen-base, zero-init, lr 1e-5, BN-freeze)..."
systemctl --user enable --now rapidock-adapter.service 2>/dev/null
systemctl --user enable --now rapidock-adapter-meter.service 2>/dev/null || true
sleep 8
echo "[4/4] adapter status:"; systemctl --user is-active rapidock-adapter.service | sed 's/^/  service: /'
echo "  log: tail -f logs/adapter_lowlr.log"
echo "  (x0 left disabled+stopped; re-enable with: systemctl --user start rapidock-x0.service)"
