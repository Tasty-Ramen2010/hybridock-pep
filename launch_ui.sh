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
# An explicitly *activated* score-env, when that is what we are already in.
ACTIVE_SCORE_PY=""
if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "score-env" ] \
    && [ -x "$CONDA_PREFIX/bin/python" ]; then
    ACTIVE_SCORE_PY="$CONDA_PREFIX/bin/python"
fi

# Order matters, and it used to be wrong. `python`/`python3` came first, on the
# reasoning that an activated environment should win. But the only test applied
# is "does this interpreter have prompt_toolkit", and plenty of interpreters
# that are NOT score-env pass it: a conda *base* env usually has it, and Google
# Colab's system python3 has it via IPython. Either gets picked ahead of a
# perfectly good score-env, and then `from hybridock_pep.output import art`
# pulls in hybridock_pep/__init__.py -> pydantic and the scoring stack, out of
# an environment that does not have them.
#
# So: an activated score-env first, then score-env wherever it is normally
# installed, and only then whatever `python` happens to be on PATH.
# 1) activated score-env, 2) known score-env locations, 3) plain python/python3
for cand in \
    ${ACTIVE_SCORE_PY:+"$ACTIVE_SCORE_PY"} \
    "$HOME/miniconda3/envs/score-env/bin/python" \
    "$HOME/anaconda3/envs/score-env/bin/python" \
    "$HOME/miniforge3/envs/score-env/bin/python" \
    "$HOME/mambaforge/envs/score-env/bin/python" \
    "/opt/homebrew/Caskroom/miniconda/base/envs/score-env/bin/python" \
    "/opt/miniconda3/envs/score-env/bin/python" \
    "/opt/conda/envs/score-env/bin/python" \
    python python3
do
    if command -v "$cand" >/dev/null 2>&1 && has_pt "$cand"; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
    echo "Could not find a Python with prompt_toolkit." >&2
    echo "Fix:  conda activate score-env   (or:  pip install prompt_toolkit )" >&2
    exit 1
fi

# $PY may have been found by full path (a candidate above) rather than via
# PATH — which means score-env's bin/ (where the `hybridock-pep` console
# script actually lives) is not necessarily on PATH for the process this
# starts. The TUI shells out to `hybridock-pep` by bare name for every real
# run, so without this it launches fine and then every run fails with
# "'hybridock-pep' not on PATH" — the UI *looks* broken right after a fresh
# install even though the install itself succeeded. tui.py also has its own
# sys.executable-relative fallback for this, but fixing PATH here means any
# other tool on that same bin/ (vina, etc.) resolves correctly too.
export PATH="$(dirname "$PY"):$PATH"

exec "$PY" -m hybridock_pep.ui.tui "$@"
