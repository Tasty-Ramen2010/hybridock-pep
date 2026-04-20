#!/usr/bin/env bash
# scripts/smoke_test.sh — validate HybriDock-Pep environment dependencies (TEST-01).
#
# Checks:
#   1. CUDA compute capability >= 12.0 (Blackwell/RTX 5070). Warns on macOS ARM.
#   2. ADFRsuite prepare_receptor4.py on PATH.
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

# --- 2. ADFRsuite prepare_receptor4.py on PATH ---
if command -v prepare_receptor4.py &>/dev/null; then
    pass "prepare_receptor4.py found on PATH"
else
    fail "prepare_receptor4.py not on PATH — install ADFRsuite from https://ccsb.scripps.edu/adfrsuite/"
fi

# --- 3. Vina >= 1.2.5 ---
if command -v vina &>/dev/null; then
    VINA_RAW=$(vina --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -z "${VINA_RAW:-}" ]; then
        fail "vina present but --version did not emit a semver"
    else
        V_MAJ=$(echo "$VINA_RAW" | cut -d. -f1)
        V_MIN=$(echo "$VINA_RAW" | cut -d. -f2)
        V_PAT=$(echo "$VINA_RAW" | cut -d. -f3)
        OK=0
        if [ "$V_MAJ" -gt 1 ]; then OK=1
        elif [ "$V_MAJ" -eq 1 ] && [ "$V_MIN" -gt 2 ]; then OK=1
        elif [ "$V_MAJ" -eq 1 ] && [ "$V_MIN" -eq 2 ] && [ "$V_PAT" -ge 5 ]; then OK=1
        fi
        if [ "$OK" -eq 1 ]; then
            pass "AutoDock Vina $VINA_RAW >= 1.2.5"
        else
            fail "vina version $VINA_RAW < 1.2.5 required — upgrade via: pip install 'vina>=1.2.5'"
        fi
    fi
else
    fail "vina not on PATH — install via: pip install 'vina>=1.2.5'"
fi

echo ""
echo "Results: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
