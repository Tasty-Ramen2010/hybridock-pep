#!/usr/bin/env bash
# Launch HybriDock-Pep Streamlit UI and expose a public URL via cloudflared.
# Usage: bash scripts/launch_ui.sh [port]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP="$REPO_ROOT/src/hybridock_pep/ui/app.py"
PORT="${1:-8501}"

# ── start Streamlit in score-env ──────────────────────────────────────────────
echo "Starting HybriDock-Pep UI on port $PORT …"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
SCORE_ENV_STREAMLIT="$CONDA_BASE/envs/score-env/bin/streamlit"

if [ -x "$SCORE_ENV_STREAMLIT" ]; then
    "$SCORE_ENV_STREAMLIT" run "$APP" \
        --server.port "$PORT" \
        --server.address "0.0.0.0" \
        --server.headless true \
        --browser.gatherUsageStats false \
        &
else
    # Fallback: use whatever streamlit is on PATH
    streamlit run "$APP" \
        --server.port "$PORT" \
        --server.address "0.0.0.0" \
        --server.headless true \
        --browser.gatherUsageStats false \
        &
fi
STREAMLIT_PID=$!
echo "  Streamlit PID: $STREAMLIT_PID"

# Give Streamlit a moment to start
sleep 2

# ── try to get a public URL ────────────────────────────────────────────────────
PUBLIC_URL=""

if command -v cloudflared &>/dev/null; then
    echo "Starting Cloudflare tunnel (free, no account needed)…"
    CF_LOG="$(mktemp /tmp/cloudflared_XXXXXX.log)"
    cloudflared tunnel --url "http://localhost:$PORT" >"$CF_LOG" 2>&1 &
    CF_PID=$!

    # Wait up to 12 s for the tunnel URL to appear
    for i in $(seq 1 24); do
        sleep 0.5
        PUBLIC_URL=$(grep -oP 'https://[^\s]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1 || true)
        if [ -n "$PUBLIC_URL" ]; then
            break
        fi
    done
fi

# ── print access info ─────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│           HybriDock-Pep is running!                      │"
if [ -n "$PUBLIC_URL" ]; then
echo "│  Public URL  : $PUBLIC_URL"
echo "│  Local URL   : http://localhost:$PORT                     │"
else
echo "│  Local URL   : http://localhost:$PORT                     │"
echo "│                                                            │"
echo "│  No public URL: install cloudflared for a shareable link  │"
echo "│    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \\│"
echo "│      | sudo tee /usr/share/keyrings/cloudflare-main.gpg   │"
echo "│    or: sudo snap install cloudflared                       │"
echo "│                                                            │"
echo "│  VS Code tunnel users: forward port $PORT in the          │"
echo "│  PORTS panel (bottom bar) to get a browser-accessible URL │"
fi
echo "└──────────────────────────────────────────────────────────┘"
echo ""
echo "Press Ctrl-C to stop."

# ── wait and forward signals ──────────────────────────────────────────────────
trap 'echo "Stopping…"; kill "$STREAMLIT_PID" 2>/dev/null; [ -n "${CF_PID:-}" ] && kill "$CF_PID" 2>/dev/null; exit 0' SIGINT SIGTERM

wait "$STREAMLIT_PID"
