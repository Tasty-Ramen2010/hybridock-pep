#!/usr/bin/env bash
# scripts/install_weights.sh — put the RAPiDock checkpoints where inference.py
# looks for them, without touching the network.
#
# The checkpoints are committed to this repository under weights/ (54 MB each,
# CC-BY-4.0 — see weights/README.md), so a `git clone` already carries them.
# This script only has to move them into the submodule's model directory, which
# is where RAPiDock resolves a bare --ckpt filename against.
#
# Zenodo is kept strictly as a fallback for the case where weights/ is missing
# (an incomplete clone, or a source tarball that dropped the large files). It is
# no longer the normal path: it was the one install step that had to reach a
# host outside GitHub/PyPI, and it is blocked on some school networks.
#
# Usage:
#   bash scripts/install_weights.sh [--model-dir DIR] [--lite] [--quiet]
#
#   --model-dir DIR  Destination (default: the RAPiDock submodule's
#                    train_models/CGTensorProductEquivariantModel/).
#   --lite           Install only rapidock_local.pt. Skips rapidock_global.pt,
#                    which only `dock --blind`'s exploratory pass reads.
#   --quiet          Only print on failure.
#
# Exit status is 0 if every requested checkpoint is present and verified.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/weights"
MODEL_DIR="$REPO_ROOT/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
LITE=0
QUIET=0
ZENODO_RECORD="14193621"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR="${2:?--model-dir needs a path}"; shift 2 ;;
        --lite)      LITE=1; shift ;;
        --quiet)     QUIET=1; shift ;;
        -h|--help)   sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^#\{1\} \{0,1\}//'; exit 0 ;;
        *)           echo "install_weights.sh: unknown flag $1" >&2; exit 2 ;;
    esac
done

# Git Bash/MSYS hands us --model-dir exactly as Windows wrote it (C:\Users\...).
# SRC_DIR comes from `pwd` so it is already POSIX, but MSYS tools (sha256sum,
# cp) cannot open a backslash path, and _sha256 swallows their error — so the
# destination never verified, and every run re-copied instead of reporting
# "already in place". Normalise once, here, where it is cheap.
if command -v cygpath >/dev/null 2>&1; then
    MODEL_DIR="$(cygpath -u "$MODEL_DIR")"
fi

say()  { [ "$QUIET" -eq 1 ] || printf '  %s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

# Checksums of the upstream Zenodo files. weights/SHA256SUMS carries the same
# two lines; these are duplicated here so a corrupted weights/ cannot vouch for
# itself.
sha_for() {
    case "$1" in
        rapidock_local.pt)  echo d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693 ;;
        rapidock_global.pt) echo a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5 ;;
        *) return 1 ;;
    esac
}

# Not every macOS ships GNU sha256sum; `shasum -a 256` is always there. If
# neither exists, say so once and stop pretending to check — a hasher that
# always answers "no" would send every run to Zenodo, and one that always
# answers "yes" would wave a truncated file through.
CAN_VERIFY=1
if command -v sha256sum >/dev/null 2>&1; then
    _sha256() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    _sha256() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }
else
    CAN_VERIFY=0
    _sha256() { echo unverifiable; }
fi

# True when $1 is a file we are willing to use as $2's checkpoint. With no
# hasher available, existence is all we have.
verified() {
    [ -f "$1" ] || return 1
    [ "$CAN_VERIFY" -eq 0 ] && return 0
    [ "$(_sha256 "$1")" = "$(sha_for "$2")" ]
}

install_ckpt() {
    name="$1"
    dest="$MODEL_DIR/$name"
    src="$SRC_DIR/$name"

    if verified "$dest" "$name"; then
        say "$name already in place and verified"
        return 0
    fi
    # A dest that exists but does not verify is worse than no dest at all: it
    # would be loaded and produce silent garbage. Replace it. (Unconditional
    # `rm -f`, not a `[ -e ] &&` guard — the guard's non-zero status on a
    # missing file is a set -e trap waiting for the first caller that does not
    # invoke this function on the left of `||`.)
    rm -f "$dest"

    if verified "$src" "$name"; then
        # Copy, not a hard link or a symlink. A link would save 54 MB, but it
        # makes the model directory and the committed weights/ file the same
        # bytes: anything that writes through the model-dir path silently
        # corrupts the repository's own copy, and `git status` then reports a
        # modified 54 MB binary with no obvious cause. Copy via .part so an
        # interrupted install leaves nothing that looks loadable.
        cp "$src" "$dest.part"
        mv "$dest.part" "$dest"
        # Verify what we just wrote, not just what we read. A copy that lands
        # short or unreadable used to be announced as a successful install and
        # only showed up as a re-copy on the next run (which is how the Windows
        # path bug above stayed hidden).
        if ! verified "$dest" "$name"; then
            warn "$name was copied from weights/ but does not verify at $dest" \
                 "— expected $(sha_for "$name"), got $(_sha256 "$dest")"
            return 1
        fi
        say "$name installed from weights/ (no download)"
        return 0
    elif [ -f "$src" ]; then
        warn "weights/$name is present but its checksum does not match the" \
             "upstream file — treating it as absent and falling back to Zenodo."
    else
        say "weights/$name not found in this checkout — falling back to Zenodo"
    fi

    url="https://zenodo.org/api/records/$ZENODO_RECORD/files/$name/content"
    if ! curl -fsSL "$url" -o "$dest.part"; then
        rm -f "$dest.part"
        warn "could not download $name from Zenodo, and weights/$name is not" \
             "usable. Some networks block zenodo.org — if that is the case," \
             "re-clone this repository (weights/ ships the file) or copy" \
             "$name into $MODEL_DIR by hand."
        return 1
    fi
    mv "$dest.part" "$dest"
    if verified "$dest" "$name"; then
        say "$name downloaded from Zenodo and checksum-verified"
    else
        warn "$name downloaded but the checksum does not match — the file is" \
             "probably truncated. Delete $dest and re-run."
        return 1
    fi
}

if [ "$CAN_VERIFY" -eq 0 ]; then
    warn "neither sha256sum nor shasum is on PATH — checkpoints will be"\
         "installed from weights/ but not checksum-verified"
fi

mkdir -p "$MODEL_DIR"

rc=0
install_ckpt rapidock_local.pt || rc=1

# rapidock_global.pt is read by exactly one code path: the exploratory
# whole-receptor pass that `dock --blind` runs when no --site is given
# (sampling/pocket_search.py). Site-directed docking never opens it.
if [ "$LITE" -eq 1 ]; then
    say "--lite: skipping rapidock_global.pt (54 MB) — 'dock --blind' will not work"
else
    install_ckpt rapidock_global.pt || rc=1
fi

exit "$rc"
