#!/usr/bin/env bash
# scripts/smoke_test.sh — validate HybriDock-Pep environment dependencies (TEST-01).
#
# Checks:
#   1. CUDA compute capability >= 12.0 (Blackwell/RTX 5070). Warns on macOS ARM.
#   2. ADFRsuite prepare_receptor on PATH.
#   3. AutoDock Vina >= 1.2.5 on PATH.
#
# Exit codes:
#   0  — all mandatory checks passed (warnings OK)
#   1  — one or more checks failed

set -euo pipefail

PASS=0; WARN=0; FAIL=0
pass()  { echo "[PASS] $*"; PASS=$((PASS+1)); }
warn()  { echo "[WARN] $*"; WARN=$((WARN+1)); }
fail()  { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

# --- 1. CUDA compute capability >= 12.0 ---
if command -v nvidia-smi &>/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "${CC:-}" ]; then
        fail "nvidia-smi present but compute_cap query returned empty"
    else
        MAJOR=${CC%.*}
        MINOR=${CC#*.}
        if [ "$MAJOR" -gt 12 ] 2>/dev/null || { [ "$MAJOR" -eq 12 ] 2>/dev/null && [ "$MINOR" -ge 0 ] 2>/dev/null; }; then
            pass "CUDA compute capability $CC >= 12.0 (Blackwell-compatible)"
        else
            fail "CUDA compute capability $CC < 12.0 — RTX 5070 requires CC 12.0+; upgrade GPU driver or use a Blackwell card"
        fi
    fi
else
    warn "nvidia-smi not found — skipping CUDA check (expected on macOS ARM / non-NVIDIA machines)"
fi

# --- 2. ADFRsuite prepare_receptor on PATH ---
if command -v prepare_receptor &>/dev/null; then
    pass "prepare_receptor found on PATH"
else
    fail "prepare_receptor not on PATH — install ADFRsuite from https://ccsb.scripps.edu/adfrsuite/"
fi

# --- 3. Vina Python API >= 1.2.5 ---
# The pipeline uses the vina Python package API (from vina import Vina), not the standalone
# CLI binary. Check score-env's Python (works whether or not score-env is active).
SCORE_PY="$HOME/miniconda3/envs/score-env/bin/python3"
if [ ! -x "$SCORE_PY" ]; then
    SCORE_PY="python3"  # fall back to active env if score-env not found
fi
VINA_CHECK=$("$SCORE_PY" - <<'PYEOF' 2>&1
import importlib.metadata, sys
try:
    ver = importlib.metadata.version("vina")
except importlib.metadata.PackageNotFoundError:
    print("NOT_FOUND")
    sys.exit(0)
parts = [int(x) for x in ver.split(".")[:3]]
ok = parts > [1, 2, 4]  # >= 1.2.5
print(f"{ver} {'OK' if ok else 'OLD'}")
PYEOF
)
if echo "$VINA_CHECK" | grep -q "NOT_FOUND"; then
    fail "vina Python package not installed in score-env — install via: pip install 'vina>=1.2.5'"
elif echo "$VINA_CHECK" | grep -q " OK$"; then
    VINA_VER=$(echo "$VINA_CHECK" | awk '{print $1}')
    pass "AutoDock Vina Python API $VINA_VER >= 1.2.5 (score-env)"
else
    VINA_VER=$(echo "$VINA_CHECK" | awk '{print $1}')
    fail "vina Python API $VINA_VER < 1.2.5 — upgrade via: pip install 'vina>=1.2.5'"
fi

echo ""
echo "Results: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
