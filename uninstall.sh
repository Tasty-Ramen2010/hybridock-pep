#!/usr/bin/env bash
# HybriDock-Pep — uninstaller. Reverses what install.sh did.
#
#   ./uninstall.sh                interactive: removes the 2 conda envs +
#                                  downloaded model weights, keeps the folder
#   ./uninstall.sh --all          also deletes this entire hybridock-pep folder
#   ./uninstall.sh --envs-only    only remove the conda environments, leave
#                                  downloaded weights and the folder alone
#   ./uninstall.sh --yes          skip the confirmation prompt (scripts/CI)
#
# Windows: run this from the WSL2/Ubuntu terminal, same as install.sh — there
# is no separate native-Windows install to undo (see install.bat).
#
# What this does NOT touch, on purpose:
#   - Your conda/Miniforge installation itself. install.sh only installs
#     Miniforge if you didn't already have conda, but this script has no
#     reliable way to tell "installed by install.sh" apart from "you already
#     had conda for other projects" — removing someone else's conda install
#     out from under them is a much worse mistake than leaving two small
#     environments behind. Only the score-env and rapidock environments are
#     removed. To remove Miniforge entirely (only if you're sure nothing
#     else depends on it): rm -rf ~/miniforge3
#   - Anything outside this repo folder (PATH entries, shell rc edits from
#     `conda init`, etc.).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ALL=0
ASSUME_YES=0
ENVS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --all) ALL=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        --envs-only) ENVS_ONLY=1 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg (see --help)" >&2
            exit 1
            ;;
    esac
done

if [ "$ALL" -eq 1 ] && [ "$ENVS_ONLY" -eq 1 ]; then
    echo "--all and --envs-only are mutually exclusive." >&2
    exit 1
fi

BOLD='\033[1m'; RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; RESET='\033[0m'
step() { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }

# ---------------------------------------------------------------------------
# Locate conda (same search order install.sh uses).
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    for _base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_base/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# ---------------------------------------------------------------------------
# Show what's about to happen, then require explicit confirmation — this can
# delete real work (uncommitted changes, everything in runs/) if pointed at
# the wrong folder, so it never runs silently.
# ---------------------------------------------------------------------------
step "This will remove"
echo "  - conda environments: score-env, rapidock"
if [ "$ENVS_ONLY" -ne 1 ]; then
    echo "  - downloaded RAPiDock model weights (~55 MB)"
fi
if [ "$ALL" -eq 1 ]; then
    echo "  - this entire folder: $REPO_ROOT"
    if command -v git >/dev/null 2>&1 && [ -n "$(git status --porcelain 2>/dev/null || true)" ]; then
        warn "this folder has uncommitted changes — they will be permanently lost"
    fi
    if [ -d runs ] && [ -n "$(ls -A runs 2>/dev/null)" ]; then
        warn "runs/ has content in it (past dock results) — it will be permanently lost"
    fi
fi
echo "  Left alone: your conda/Miniforge installation itself (see --help)."

if [ "$ASSUME_YES" -ne 1 ]; then
    printf "\nContinue? [y/N] "
    read -r reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted — nothing removed."; exit 0 ;;
    esac
fi

# ---------------------------------------------------------------------------
# Conda environments
# ---------------------------------------------------------------------------
step "Removing conda environments"
if command -v conda >/dev/null 2>&1; then
    for env_name in score-env rapidock; do
        if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
            conda env remove -n "$env_name" -y >/dev/null
            ok "removed $env_name"
        else
            warn "$env_name not found — already removed, or never created"
        fi
    done
else
    warn "conda not found on this machine — skipping environment removal"
fi

if [ "$ENVS_ONLY" -eq 1 ]; then
    step "Done (--envs-only)"
    echo "  Downloaded weights and this folder were left in place."
    exit 0
fi

# ---------------------------------------------------------------------------
# Downloaded model weights
# ---------------------------------------------------------------------------
step "Removing downloaded model weights"
CKPT="third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
if [ -f "$CKPT" ]; then
    rm -f "$CKPT"
    ok "removed $CKPT"
else
    warn "$CKPT not found — already removed, or never downloaded"
fi

# ---------------------------------------------------------------------------
# The repo folder itself (only with --all)
# ---------------------------------------------------------------------------
if [ "$ALL" -eq 1 ]; then
    step "Removing the hybridock-pep folder"
    PARENT="$(dirname "$REPO_ROOT")"
    cd "$PARENT"
    rm -rf "$REPO_ROOT"
    ok "removed $REPO_ROOT"
    printf "${GREEN}Done.${RESET} hybridock-pep is fully uninstalled.\n"
else
    step "Done"
    echo "  Conda environments and downloaded weights are gone."
    echo "  This folder ($REPO_ROOT) was left in place — re-run with --all to remove it too,"
    echo "  or just: cd .. && rm -rf $(basename "$REPO_ROOT")"
fi
