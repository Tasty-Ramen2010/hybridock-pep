#!/usr/bin/env bash
# scripts/smoke_test.sh — validate HybriDock-Pep environment dependencies (TEST-01).
#
# Checks:
#   1. Device availability: CUDA (Linux/WSL2) or MPS (macOS Apple Silicon).
#      Warns on macOS Intel (CPU-only) and Linux without nvidia-smi.
#   2. RAPiDock-Reloaded submodule present + importable in rapidock env.
#   3. ADFRsuite prepare_receptor on PATH.
#   4. AutoDock Vina >= 1.2.5 Python API in score-env.
#   5. OpenMM importable (for MM-GBSA).
#
# Exit codes:
#   0  — all mandatory checks passed (warnings/info OK)
#   1  — one or more checks failed

set -euo pipefail

PASS=0; WARN=0; FAIL=0; INFO=0
pass() { echo "[PASS] $*"; PASS=$((PASS+1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN+1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "[INFO] $*"; INFO=$((INFO+1)); }

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"

# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
if [ "$PLATFORM" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        # Check MPS availability via Python
        RAPIDOCK_PY="$HOME/miniconda3/envs/rapidock/bin/python3"
        [ -x "$RAPIDOCK_PY" ] || RAPIDOCK_PY="$(command -v python3 2>/dev/null || echo '')"
        if [ -n "$RAPIDOCK_PY" ] && [ -x "$RAPIDOCK_PY" ]; then
            MPS_CHECK=$("$RAPIDOCK_PY" -c "
import torch, sys
if not hasattr(torch.backends, 'mps'):
    print('NO_MPS_MODULE')
elif not torch.backends.mps.is_available():
    print('MPS_NOT_AVAILABLE')
else:
    print('MPS_OK')
" 2>/dev/null || echo "PYTHON_ERROR")
            case "$MPS_CHECK" in
                MPS_OK)            pass "MPS backend available (macOS Apple Silicon)" ;;
                MPS_NOT_AVAILABLE) warn "MPS module present but not available — check macOS version >= 12.3" ;;
                NO_MPS_MODULE)     warn "torch.backends.mps missing — PyTorch too old or CPU-only build" ;;
                *)                 info "No CUDA GPU — MPS (Apple Silicon) or CPU will be used for Stage 1" ;;
            esac
        else
            info "rapidock env not found — skipping MPS check (install Step 2)"
        fi
    else
        info "macOS Intel detected — Stage 1 will use CPU (slow). Use --n-samples 10 or --input-poses bypass."
    fi
elif command -v nvidia-smi &>/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "${CC:-}" ]; then
        fail "nvidia-smi present but compute_cap query returned empty — check GPU driver"
    else
        MAJOR="${CC%.*}"
        MINOR="${CC#*.}"
        if { [ "$MAJOR" -gt 12 ] 2>/dev/null; } || \
           { [ "$MAJOR" -eq 12 ] 2>/dev/null && [ "$MINOR" -ge 0 ] 2>/dev/null; }; then
            pass "CUDA compute capability $CC >= 12.0 (Blackwell-compatible)"
        elif { [ "$MAJOR" -ge 8 ] 2>/dev/null; }; then
            pass "CUDA compute capability $CC >= 8.0 (Ampere/Turing — compatible)"
        else
            fail "CUDA compute capability $CC < 8.0 — may not be compatible with PyTorch 2.7"
        fi
    fi
else
    warn "nvidia-smi not found — no CUDA GPU detected. Stage 1 will use CPU."
fi

# ---------------------------------------------------------------------------
# 2. RAPiDock-Reloaded submodule
# ---------------------------------------------------------------------------
RDOCK_DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/RAPiDock"
if [ ! -f "$RDOCK_DIR/inference.py" ]; then
    fail "RAPiDock-Reloaded not found at $RDOCK_DIR — run: git submodule update --init --recursive"
else
    pass "RAPiDock-Reloaded submodule present ($RDOCK_DIR)"
fi

# ---------------------------------------------------------------------------
# 3. ADFRsuite prepare_receptor on PATH
# ---------------------------------------------------------------------------
if command -v prepare_receptor &>/dev/null; then
    pass "prepare_receptor found on PATH ($(command -v prepare_receptor))"
elif [ "$PLATFORM" = "Darwin" ]; then
    warn "prepare_receptor not on PATH — install ADFRsuite (Rosetta 2) from https://ccsb.scripps.edu/adfrsuite/"
else
    fail "prepare_receptor not on PATH — install ADFRsuite from https://ccsb.scripps.edu/adfrsuite/"
fi

# ---------------------------------------------------------------------------
# 4. Vina Python API >= 1.2.5
# ---------------------------------------------------------------------------
SCORE_PY="$HOME/miniconda3/envs/score-env/bin/python3"
[ -x "$SCORE_PY" ] || SCORE_PY="$(command -v python3 2>/dev/null || echo '')"

if [ -n "$SCORE_PY" ] && [ -x "$SCORE_PY" ]; then
    VINA_CHECK=$("$SCORE_PY" - <<'PYEOF' 2>&1
import importlib.metadata, sys
try:
    ver = importlib.metadata.version("vina")
except importlib.metadata.PackageNotFoundError:
    print("NOT_FOUND")
    sys.exit(0)
parts = [int(x) for x in ver.split(".")[:3]]
ok = parts > [1, 2, 4]
print(f"{ver} {'OK' if ok else 'OLD'}")
PYEOF
)
    if echo "$VINA_CHECK" | grep -q "NOT_FOUND"; then
        fail "vina Python package not in score-env — pip install 'vina>=1.2.5'"
    elif echo "$VINA_CHECK" | grep -q " OK$"; then
        VINA_VER=$(echo "$VINA_CHECK" | awk '{print $1}')
        pass "AutoDock Vina Python API $VINA_VER >= 1.2.5"
    else
        VINA_VER=$(echo "$VINA_CHECK" | awk '{print $1}')
        fail "vina Python API $VINA_VER < 1.2.5 — pip install 'vina>=1.2.5'"
    fi
else
    warn "score-env Python not found — skipping Vina check (install Step 1)"
fi

# ---------------------------------------------------------------------------
# 5. OpenMM importable (for MM-GBSA)
# ---------------------------------------------------------------------------
if [ -n "$SCORE_PY" ] && [ -x "$SCORE_PY" ]; then
    OPENMM_CHECK=$("$SCORE_PY" -c "
import importlib.metadata as m, sys
try:
    v = m.version('openmm')
    print(f'OK {v}')
except:
    print('NOT_FOUND')
" 2>/dev/null || echo "ERROR")
    if echo "$OPENMM_CHECK" | grep -q "^OK"; then
        OV=$(echo "$OPENMM_CHECK" | awk '{print $2}')
        pass "OpenMM $OV importable in score-env (MM-GBSA ready)"
    else
        warn "OpenMM not installed in score-env — MM-GBSA (--refine-topk) will be unavailable"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Results: ${PASS} passed, ${INFO} info, ${WARN} warnings, ${FAIL} failed"
echo "Platform: $PLATFORM $ARCH"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
