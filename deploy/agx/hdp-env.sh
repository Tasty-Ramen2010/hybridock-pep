#!/bin/bash
# The environment of the HybriDock-Pep demo server on the AGX. Sourced by hdp-serve.sh EVERY time it (re)starts the server, so an edit
# here (or in /mnt/data/agx-setup/hdp.conf, which is sourced last) applies on the next restart without restarting anything else.
HDP=/mnt/hdp/hybridock-pep
CONDA=/mnt/hdp/miniforge3
export PATH="$CONDA/envs/score-env/bin:$CONDA/bin:/usr/local/cuda-11.4/bin:$PATH"
export HYBRIDOCK_TIMING_FILE="$HDP/src/hybridock_pep/web/static/data/timing.json"
export HYBRIDOCK_TIMING_HISTORY="$HDP/runs/timing_history.jsonl"      # how long each peptide really took here, quoted back next time
export HYBRIDOCK_JOB_HOOK="$HDP/deploy/agx/hdp-job-hook"
export HYBRIDOCK_NICE="${HYBRIDOCK_NICE:--10}"          # a prediction outranks everything else on the board
export HYBRIDOCK_IDLE_SECONDS="${HYBRIDOCK_IDLE_SECONDS:-900}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# GPU sampling: the Jetson build of PyTorch (Python 3.8, CUDA 11.4) in its own environment, plus the CUDA libraries the chroot lacks
# (unpacked under cuda-extra by deploy/agx/install-gpu.sh; nothing is installed system-wide).
X=/mnt/hdp/cuda-extra
export LD_LIBRARY_PATH="$X/usr/local/cuda-11.4/targets/aarch64-linux/lib:$X/usr/lib/aarch64-linux-gnu:/usr/local/tegra-libs:/usr/local/cuda-11.4/lib64:${LD_LIBRARY_PATH:-}"
if [ -x "$CONDA/envs/rapidock-gpu/bin/python" ] && [ -e /dev/nvhost-gpu ]; then
  export RAPIDOCK_PYTHON="$CONDA/envs/rapidock-gpu/bin/python"
  export HYBRIDOCK_ACCELERATOR="cuda:NVIDIA Jetson AGX Orin"
fi
export HDP_PORT=${HDP_PORT:-8765}
[ -f /mnt/data/agx-setup/hdp.conf ] && . /mnt/data/agx-setup/hdp.conf
