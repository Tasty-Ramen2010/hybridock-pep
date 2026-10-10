#!/usr/bin/env bash
# HybriDock-Pep web app launcher (macOS / Linux / WSL2): starts the app and opens it in your browser.
#
#   ./launch_web.sh                    # opens http://127.0.0.1:8000
#   ./launch_web.sh --port 9000        # pick a port
#   ./launch_web.sh --no-browser       # for a server or WSL2: just print the address
#
# It finds the score-env that install.sh built, so you do not have to `conda activate` anything first.
# Close the window (or press Ctrl-C) to stop the app.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BIN=""
if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "score-env" ] && [ -x "$CONDA_PREFIX/bin/hybridock-pep" ]; then
    BIN="$CONDA_PREFIX/bin"
fi
if [ -z "$BIN" ]; then
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" \
                "/opt/homebrew/Caskroom/miniconda/base" "/opt/homebrew/Caskroom/miniforge/base" "/opt/miniconda3" "/opt/conda"; do
        if [ -x "$base/envs/score-env/bin/hybridock-pep" ]; then BIN="$base/envs/score-env/bin"; break; fi
    done
fi
if [ -n "$BIN" ]; then
    export PATH="$BIN:$PATH"
elif ! command -v hybridock-pep >/dev/null 2>&1; then
    echo "HybriDock-Pep is not installed yet (no score-env found)." >&2
    echo "Run the installer first:   ./install.sh" >&2
    exit 1
fi

echo "Starting HybriDock-Pep. Close this window or press Ctrl-C to stop it."
exec hybridock-pep serve "$@"
