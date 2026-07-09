#!/usr/bin/env bash
# HybriDock-Pep terminal UI launcher (macOS / Linux / WSL).
#
#   ./launch_ui.sh            # full-screen interactive UI
#   ./launch_ui.sh --demo     # full-screen UI + auto-run a simulated pipeline (no GPU needed)
#   ./launch_ui.sh --cli      # plain step-by-step wizard (SSH / dumb terminals)
#   ./launch_ui.sh --print    # print the hybridock-pep command without running
#
# Needs a Python with prompt_toolkit (the score-env conda env has it). Also works if you have
# already run `conda activate score-env` — it just uses the active `python`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"

has_pt() { "$1" -c "import prompt_toolkit" >/dev/null 2>&1; }

PY=""
# 1) active interpreter, 2) common score-env locations across OSes, 3) plain python3
for cand in \
    python python3 \
    "$HOME/miniconda3/envs/score-env/bin/python" \
    "$HOME/anaconda3/envs/score-env/bin/python" \
    "$HOME/miniforge3/envs/score-env/bin/python" \
    "$HOME/mambaforge/envs/score-env/bin/python" \
    "/opt/homebrew/Caskroom/miniconda/base/envs/score-env/bin/python" \
    "/opt/miniconda3/envs/score-env/bin/python"
do
    if command -v "$cand" >/dev/null 2>&1 && has_pt "$cand"; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
    echo "Could not find a Python with prompt_toolkit." >&2
    echo "Fix:  conda activate score-env   (or:  pip install prompt_toolkit )" >&2
    exit 1
fi

exec "$PY" -m hybridock_pep.ui.tui "$@"
