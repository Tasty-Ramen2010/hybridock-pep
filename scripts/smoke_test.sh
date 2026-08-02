#!/usr/bin/env bash
# scripts/smoke_test.sh — validate HybriDock-Pep environment dependencies (TEST-01).
#
# Checks:
#   1. Device availability: CUDA (Linux/WSL2) or MPS (macOS Apple Silicon).
#      Warns on macOS Intel (CPU-only) and Linux without nvidia-smi.
#   2. RAPiDock-Reloaded submodule present + importable in rapidock env.
#   3. Receptor-prep tooling: prepare_receptor -> mk_prepare_receptor.py ->
#      obabel (the real fallback chain), plus optional autogrid4.
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

# Locate a conda env's python3 regardless of which conda distro installed it
# (miniconda3, miniforge3, anaconda3, mambaforge, /opt/conda, ...). Falls back
# to the already-active `python3` on PATH if it belongs to that env, then to
# plain `python3` as a last resort.
find_conda_env_python() {
    local env_name="$1"
    if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "$env_name" ] \
        && [ -x "$CONDA_PREFIX/bin/python3" ]; then
        echo "$CONDA_PREFIX/bin/python3"
        return
    fi
    local conda_exe base
    conda_exe="$(command -v conda 2>/dev/null || true)"
    if [ -n "$conda_exe" ]; then
        base="$(cd "$(dirname "$conda_exe")/.." && pwd)"
        if [ -x "$base/envs/$env_name/bin/python3" ]; then
            echo "$base/envs/$env_name/bin/python3"
            return
        fi
    fi
    for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" \
                "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3"; do
        if [ -x "$base/envs/$env_name/bin/python3" ]; then
            echo "$base/envs/$env_name/bin/python3"
            return
        fi
    done
    command -v python3 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
if [ "$PLATFORM" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        # Check MPS availability via Python
        RAPIDOCK_PY="$(find_conda_env_python rapidock)"
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
# 3. Receptor-prep tooling
# ---------------------------------------------------------------------------
# Mirrors prep/receptor.py's actual fallback chain: ADFRsuite prepare_receptor
# -> meeko mk_prepare_receptor.py -> obabel. ADFRsuite has not been required
# since meeko took over, so demanding it here reported [FAIL] on installs that
# were completely fine — including every fresh one, which is the worst possible
# first impression.
#
# Tools installed by score-env.yml live in that env's bin/, not on the PATH of
# the shell running this script (installers and `ssh host ./install.sh` never
# have score-env activated), so look there too before concluding a tool is
# absent.
SCORE_BIN=""
SCORE_PY_PROBE="$(find_conda_env_python score-env)"
if [ -n "$SCORE_PY_PROBE" ]; then
    SCORE_BIN="$(dirname "$SCORE_PY_PROBE")"
fi

have_tool() {
    command -v "$1" &>/dev/null && { echo "$(command -v "$1")"; return 0; }
    [ -n "$SCORE_BIN" ] && [ -x "$SCORE_BIN/$1" ] && { echo "$SCORE_BIN/$1"; return 0; }
    return 1
}

if PREP_PATH="$(have_tool prepare_receptor)"; then
    pass "receptor prep: ADFRsuite prepare_receptor ($PREP_PATH)"
elif PREP_PATH="$(have_tool mk_prepare_receptor.py)"; then
    pass "receptor prep: meeko mk_prepare_receptor.py ($PREP_PATH) — ADFRsuite not needed"
elif PREP_PATH="$(have_tool obabel)"; then
    warn "receptor prep: only the obabel fallback found ($PREP_PATH) — install meeko for" \
         "better PDBQT fidelity:  pip install 'meeko>=0.7'"
else
    fail "no receptor-prep tool found (tried prepare_receptor, mk_prepare_receptor.py, obabel)" \
         "— install meeko:  pip install 'meeko>=0.7'"
fi

# autogrid4 is optional: --scoring ad4 is off by default, and conda-forge has
# no linux-aarch64 build at all, so its absence is information, not a failure.
if AG_PATH="$(have_tool autogrid4)"; then
    pass "AD4 grid maps: autogrid4 ($AG_PATH) — '--scoring ad4' available"
else
    info "autogrid4 not installed — '--scoring ad4' unavailable (off by default; the" \
         "reported ΔG comes from the affinity model, not AD4)"
fi

# ---------------------------------------------------------------------------
# 4. Vina Python API >= 1.2.5
# ---------------------------------------------------------------------------
SCORE_PY="$(find_conda_env_python score-env)"

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
