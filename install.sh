#!/usr/bin/env bash
# HybriDock-Pep — one-command installer (Linux, WSL2, macOS).
#
#   git clone --recursive https://github.com/Tasty-Ramen2010/hybridock-pep.git
#   cd hybridock-pep
#   ./install.sh
#
# What this does, in order:
#   1. Installs Miniforge (conda) automatically if you don't already have conda.
#   2. Initialises the RAPiDock-Reloaded git submodule.
#   3. Creates both conda environments (score-env, rapidock) with the right
#      PyTorch/PyG build for your OS + GPU — auto-detected.
#   4. Downloads the RAPiDock model weights (public Zenodo record, ~55 MB).
#   5. Checks receptor-prep tooling (meeko + autogrid). Fully automated: no
#      license click-through, no manual download. ADFRsuite is NOT required,
#      though it is used automatically if you already have it on PATH.
#   6. Runs the smoke test.
#   7. Launches the guided terminal UI so you can run your first dock without
#      memorizing any CLI flags.
#
# Flags:
#   --backend cuda|rocm|xpu|mps|cpu   Force a compute backend (default: auto-detect)
#   --force                            Recreate conda envs even if they already exist
#   --skip-rapidock                    Only set up score-env (Stage 2 scoring only)
#   --skip-scoring                     Only set up rapidock (Stage 1 sampling only)
#   --no-ui                            Don't auto-launch the terminal UI at the end
#   -h, --help                         Show this help
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

NO_UI=0
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-ui) NO_UI=1 ;;
        -h|--help)
            sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; YELLOW='\033[33m'; RESET='\033[0m'
step() { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"

# Google Colab: this script would install Miniforge into a $HOME that is wiped
# with the VM, pick the CUDA build for the wrong compute capability (Colab's T4
# is CC 7.5, not the 8.0+ assumed below), and finish by exec'ing an interactive
# terminal UI a notebook kernel cannot host. scripts/colab_setup.sh does the
# same work in a way that survives all three.
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d /content/sample_data ]; then
    printf "\n${BOLD}This looks like Google Colab.${RESET}\n"
    echo "  Use the Colab installer instead:"
    echo "      !bash scripts/colab_setup.sh"
    echo "  or open notebooks/HybriDock_Pep_Colab.ipynb, which drives the whole"
    echo "  pipeline end to end. See INSTALL.md → Google Colab."
    echo
    echo "  Continuing with ./install.sh anyway is not recommended; press Ctrl-C"
    echo "  now, or wait 15 s to proceed."
    sleep 15
fi

# ---------------------------------------------------------------------------
# 1. conda (install Miniforge automatically if missing)
# ---------------------------------------------------------------------------
step "Checking for conda"
# `conda` is only on PATH once a shell has been through `conda init` + a login
# shell. Non-interactive shells (ssh host './install.sh', CI, a terminal opened
# before conda init took effect) have a perfectly good conda installed and
# still fail `command -v conda`. Look for the installation itself before
# concluding there isn't one — otherwise we try to install Miniforge over an
# existing ~/miniforge3, the installer refuses ("File or directory already
# exists"), and set -e kills the whole install with a misleading message.
# Same prefixes setup_environment.py::_conda_base() checks, same order.
if ! command -v conda >/dev/null 2>&1; then
    for _base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_base/etc/profile.d/conda.sh" ]; then
            # conda's own shell-integration script isn't written to be
            # `set -u`-clean, and sourcing it can run OTHER packages'
            # activate.d/deactivate.d hooks as a side effect (found on a real
            # Windows CI run of uninstall.sh, same sourcing pattern: a
            # stale/partial nested-shell state made a khronos-opencl-icd-loader
            # deactivate hook reference a backup variable that was never set
            # in this process, aborting the whole script under -u). Not this
            # project's bug to fix -- relax nounset only around this one
            # third-party sourcing call.
            set +u
            # shellcheck disable=SC1091
            source "$_base/etc/profile.d/conda.sh"
            set -u
            break
        fi
    done
fi

if command -v conda >/dev/null 2>&1; then
    ok "conda found: $(command -v conda)"
else
    warn "conda not found — installing Miniforge (this is the only download in this" \
         "step, ~80 MB, takes under a minute)"
    case "$PLATFORM" in
        Darwin) MF_OS="MacOSX" ;;
        Linux)  MF_OS="Linux" ;;
        *) echo "Unsupported platform for auto Miniforge install: $PLATFORM" >&2
           echo "Install conda manually: https://github.com/conda-forge/miniforge/releases" >&2
           exit 1 ;;
    esac
    MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${MF_OS}-${ARCH}.sh"
    # Miniforge's installer refuses to run unless invoked as "$0" ending in
    # .sh (its own guard against being sourced), so the temp file needs that
    # extension.
    #
    # It cannot come from `mktemp <template>.sh`: BSD/macOS mktemp only expands
    # XXXXXX when it is the *trailing* part of the template. Given a suffix it
    # takes the name literally, creates /tmp/hdp-miniforge-installer.XXXXXX.sh
    # once, and then fails every later run with
    #     mktemp: mkstemp failed on ...XXXXXX.sh: File exists
    # which kills install.sh at step 1 under set -e. GNU mktemp has --suffix
    # for this; macOS does not.
    #
    # Make a temp *directory* instead — portable everywhere — and use a fixed
    # filename inside it.
    MF_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/hdp-miniforge.XXXXXX")"
    MF_INSTALLER="$MF_TMPDIR/miniforge-installer.sh"
    curl -fsSL "$MF_URL" -o "$MF_INSTALLER"
    bash "$MF_INSTALLER" -b -p "$HOME/miniforge3"
    rm -rf "$MF_TMPDIR"
    # See the nounset note on the sourcing call above -- same reasoning.
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    set -u
    ok "Miniforge installed at $HOME/miniforge3"
    echo "  (add 'source \$HOME/miniforge3/etc/profile.d/conda.sh' to your shell rc" \
         "for future sessions, or run 'conda init')"
fi

# ---------------------------------------------------------------------------
# 2. RAPiDock-Reloaded submodule
# ---------------------------------------------------------------------------
step "Initialising the RAPiDock-Reloaded submodule"
if [ -f "third_party/RAPiDock/inference.py" ]; then
    ok "submodule already present"
else
    git submodule update --init --recursive
    ok "submodule initialised"
fi

# ---------------------------------------------------------------------------
# 3. Conda environments (auto-detects OS/GPU backend)
# ---------------------------------------------------------------------------
step "Creating conda environments (score-env + rapidock)"
if [ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]; then
    python3 scripts/setup_environment.py "${PASSTHROUGH_ARGS[@]}"
else
    python3 scripts/setup_environment.py
fi

# ---------------------------------------------------------------------------
# 4. RAPiDock model weights (public Zenodo record — no login needed)
# ---------------------------------------------------------------------------
step "Downloading RAPiDock model weights"
WEIGHTS_DIR="third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
mkdir -p "$WEIGHTS_DIR"

# Not every macOS ships GNU sha256sum; `shasum -a 256` is always there. Pick one
# up front so the verify below can't silently no-op into a re-download loop.
if command -v sha256sum >/dev/null 2>&1; then
    _sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    _sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    _sha256() { echo "no-sha-tool"; }
fi

# Both checkpoints are required for a full install: rapidock_local.pt for
# ordinary site-directed docking, rapidock_global.pt for the --blind
# pocket-search pass (sampling/pocket_search.py hard-codes it). Skipping the
# global one leaves `dock --blind` dead on arrival with a torch.load failure
# ~70s into Stage 1, so it is fetched unconditionally, not on demand.
fetch_ckpt() {
    _name="$1"; _want="$2"; _dest="$WEIGHTS_DIR/$_name"
    if [ -f "$_dest" ] && [ "$(_sha256 "$_dest")" = "$_want" ]; then
        ok "$_name already present and verified"
        return 0
    fi
    curl -fsSL "https://zenodo.org/api/records/14193621/files/$_name/content" \
        -o "$_dest.part" || {
        warn "$_name download failed — '--blind' docking needs rapidock_global.pt;" \
             "ordinary docking needs rapidock_local.pt. Re-run ./install.sh to retry."
        rm -f "$_dest.part"
        return 0
    }
    # Only move into place once the bytes are complete: a half-written .pt is
    # indistinguishable from a good one to the runtime's existence check.
    mv "$_dest.part" "$_dest"
    if [ "$(_sha256 "$_dest")" = "$_want" ]; then
        ok "$_name downloaded and checksum-verified"
    else
        warn "$_name downloaded but checksum did not match — the file may be" \
             "corrupted or the Zenodo record was updated. Continuing, but re-check this."
    fi
}

fetch_ckpt rapidock_local.pt \
    d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693
fetch_ckpt rapidock_global.pt \
    a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5

# ---------------------------------------------------------------------------
# 5. Receptor-prep / grid tooling (no license click-through required)
# ---------------------------------------------------------------------------
# ADFRsuite is no longer required. meeko's mk_prepare_receptor.py handles
# receptor PDBQT and conda-forge autogrid handles AD4 grid maps; both install
# unattended and have native Apple Silicon builds. ADFRsuite is still used
# automatically if it happens to be on PATH, which keeps results identical for
# installs that already have it.
step "Checking receptor-prep tooling"
# These tools are installed *into score-env*, and this script never activates
# it, so `command -v` alone reports every one of them missing on a perfectly
# good install. Look inside the env too.
SCORE_ENV_BIN=""
for _base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [ -d "$_base/envs/score-env/bin" ]; then
        SCORE_ENV_BIN="$_base/envs/score-env/bin"
        break
    fi
done
# score-env first, then $PATH — the same order as hybridock_pep/toolpath.which(),
# so what this step reports is what the runtime will actually run. Checking $PATH
# first reports a conda *base* env's copy of a tool (base is on $PATH almost
# everywhere) while the runtime uses score-env's, which is how a broken base-env
# meeko passed this check and then failed at dock time.
find_tool() {
    [ -n "$SCORE_ENV_BIN" ] && [ -x "$SCORE_ENV_BIN/$1" ] && { echo "$SCORE_ENV_BIN/$1"; return 0; }
    command -v "$1" 2>/dev/null && return 0
    return 1
}

if PREP="$(find_tool prepare_receptor)"; then
    ok "ADFRsuite found — will be preferred: $PREP"
elif PREP="$(find_tool mk_prepare_receptor.py)"; then
    # Existence is not enough: meeko imports rdkit at module load, so a meeko
    # installed without it is on disk, executable, and dies on first use. Run it
    # once here so that surfaces now rather than mid-dock as an obabel fallback.
    if MEEKO_ERR="$("$PREP" --help 2>&1)"; then
        ok "meeko mk_prepare_receptor.py found, no ADFRsuite needed: $PREP"
    else
        warn "mk_prepare_receptor.py at $PREP is installed but not runnable —" \
             "receptor prep will fall back to obabel. Fix with:" \
             "conda install -n score-env -c conda-forge rdkit" \
             "Error was: $(echo "$MEEKO_ERR" | tail -n 1)"
    fi
else
    warn "No receptor-prep tool found. Install meeko:  pip install 'meeko>=0.7'"
fi

if AG="$(find_tool autogrid4)"; then
    ok "autogrid4 found — '--scoring ad4' available: $AG"
else
    warn "autogrid4 not available on this platform — '--scoring ad4' will not work." \
         "Everything else is unaffected: AD4 is off by default and the reported ΔG" \
         "comes from the affinity model, not AD4."
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------
step "Running smoke test"
SCORE_PY="$(python3 -c "
import json, subprocess, shutil
from pathlib import Path
exe = shutil.which('conda')
base = Path(exe).resolve().parent.parent if exe else Path.home() / 'miniforge3'
print(base / 'envs' / 'score-env' / 'bin' / 'python3')
" 2>/dev/null)"
if [ -x "$SCORE_PY" ]; then
    bash scripts/smoke_test.sh || warn "smoke test reported failures — see above"
else
    warn "could not locate score-env's python3 — skipping smoke test (run it manually" \
         "after 'conda activate score-env')"
fi

# ---------------------------------------------------------------------------
# 7. Launch the guided UI
# ---------------------------------------------------------------------------
if [ "$NO_UI" -eq 0 ] && [ -t 0 ] && [ -t 1 ]; then
    step "Launching the guided terminal UI"
    exec ./launch_ui.sh
else
    step "Setup complete"
    echo "  Run './launch_ui.sh' for the guided terminal UI, or:"
    echo "    conda activate score-env"
    echo "    hybridock-pep dock --peptide ETFSDLWKLLPE \\"
    echo "        --receptor data/pdbs/1YCR_mdm2.pdb --site 25.20 -25.61 -7.97 --box 30 \\"
    echo "        --n-samples 20 --output-dir runs/demo"
fi
