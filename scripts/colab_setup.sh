#!/usr/bin/env bash
# scripts/colab_setup.sh — HybriDock-Pep setup for Google Colab (T4 / L4 / A100).
#
#   !bash scripts/colab_setup.sh
#
# Why a separate script from ./install.sh:
#   Colab has no conda, no persistent home, and a GPU whose compute capability
#   decides which PyTorch build is even loadable (a T4 is CC 7.5 — the repo's
#   default cu128 wheels target CC 8.0+ / Blackwell and are the wrong choice
#   there). install.sh also ends by exec'ing an interactive terminal UI, which
#   a notebook kernel cannot host. This script does the same work, headless,
#   with micromamba instead of Miniforge (a ~5 MB static binary and a much
#   faster solver — Colab sessions are billed in wall-clock minutes).
#
# What it does:
#   1. Installs micromamba with its root prefix at /opt/conda, which is one of
#      the prefixes rapidock_runner.py and toolpath.py already probe, so the
#      two envs are found with no extra configuration.
#   2. Creates score-env (Python 3.11: Vina, OpenMM, RDKit, meeko) and rapidock
#      (Python 3.10: the diffusion stack) from the repo's own env files.
#   3. Installs the PyTorch + PyG build matching THIS runtime's GPU, then
#      proves it with a real CUDA matmul rather than trusting the version string.
#   4. Installs both RAPiDock checkpoints from weights/ (committed to the
#      repo, so no Zenodo download and nothing outside GitHub to reach).
#   5. Optionally redirects the torch/ESM cache at a persistent --cache-dir (a
#      Google Drive folder), so the ~2.5 GB ESM-2 download happens once per
#      account rather than once per session.
#   6. Writes /content/hybridock_env.sh + prints the paths the notebook needs.
#
# Flags:
#   --cache-dir DIR     Persist the ESM-2 / torch cache here (e.g. a mounted
#                       Drive folder). Default: session-local only. The RAPiDock
#                       checkpoints no longer need caching — they ship in the repo.
#   --backend cuda|cpu  Force a backend (default: auto-detect via nvidia-smi).
#   --skip-rapidock     score-env only (Stage 2 scoring / --input-poses runs).
#   --lite              Install only what plain docking needs. Skips the blind-mode
#                       checkpoint (54 MB), the AD4/obabel extras, and the Boost
#                       C++ headers once Vina has compiled (188 MB). Saves ~260 MB.
#                       Rules out `dock --blind` and `--scoring ad4`.
#   --force             Recreate envs that already exist.
#   -h, --help          Show this help.
#
# On "lite": the two costs that dominate an install are PyTorch and the 2.4 GB
# ESM-2 language model, and docking needs both — no flag can remove them. --lite
# trims the ~260 MB around them, which is real but is not the difference between
# fitting and not fitting. If you only need to SCORE existing poses, use
# --skip-rapidock instead: that one skips the whole sampling env and the ESM
# download, which is a multi-gigabyte saving, at the cost of not being able to dock.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR=""
FORCE=0
SKIP_RAPIDOCK=0
LITE=0
BACKEND=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cache-dir) CACHE_DIR="${2:?--cache-dir needs a path}"; shift 2 ;;
        --backend)   BACKEND="${2:?--backend needs cuda or cpu}"; shift 2 ;;
        --skip-rapidock) SKIP_RAPIDOCK=1; shift ;;
        --lite)      LITE=1; shift ;;
        --force)     FORCE=1; shift ;;
        -h|--help)   sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1 (try --help)" >&2; exit 2 ;;
    esac
done

BOLD='\033[1m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'
step() { printf "\n${BOLD}==> %s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
die()  { printf "  ${RED}✗${RESET} %s\n" "$*" >&2; exit 1; }

START_TS=$(date +%s)

# ---------------------------------------------------------------------------
# 0. Runtime sanity
# ---------------------------------------------------------------------------
step "Checking the Colab runtime"
[ "$(uname -s)" = "Linux" ] || die "this script is for Colab (Linux); use ./install.sh locally"

GPU_NAME=""
GPU_CC=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
    ok "GPU: $GPU_NAME (compute capability $GPU_CC)"
else
    warn "no GPU visible — in Colab: Runtime ▸ Change runtime type ▸ T4 GPU, then re-run this cell"
fi

if [ -z "$BACKEND" ]; then
    if [ -n "$GPU_CC" ]; then BACKEND="cuda"; else BACKEND="cpu"; fi
fi

# The CUDA build has to match the GPU, not the repo's default. Colab hands out
# T4 (7.5), L4 (8.9) and A100 (8.0) depending on tier and time of day, so this
# is decided per session:
#   - CC 8.0+  : torch 2.7.0 + cu128, the same build install.sh uses.
#   - CC 7.5   : torch 2.6.0 + cu124. PyTorch's cu128 wheels are built for
#                Blackwell-era arches and are not a safe bet on Turing; the
#                cu124 build unambiguously ships sm_75 kernels, and PyG
#                publishes matching torch-2.6.0+cu124 wheels (building
#                torch-scatter from source in a notebook takes ~20 min).
#
# Every spec below carries its "+cuXXX" local version, and that is load-bearing,
# not decoration. `conda-forge::e3nn` in rapidock-env.yml pulls conda-forge's
# own pytorch into the env, and micromamba resolves the __cuda virtual package
# from Colab's driver, so what lands is a working CUDA build. A bare
# `pip install torch==2.6.0` then compares equal to it, reports "Requirement
# already satisfied", and silently does nothing — leaving conda-forge's torch in
# place. conda-forge builds with _GLIBCXX_USE_CXX11_ABI=1 while the PyG wheels
# are compiled against pytorch.org's ABI-0 build, so torch_scatter dies on
# import with `undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb` about
# 10 s into Stage 1. "2.6.0+cu124" can never compare equal to "2.6.0", so pip
# actually performs the replacement.
TORCH_SPEC=""; TORCH_INDEX=""; PYG_FIND=""
if [ "$BACKEND" = "cuda" ]; then
    CC_MAJOR="${GPU_CC%%.*}"
    if [ "${CC_MAJOR:-0}" -ge 8 ] 2>/dev/null; then
        TORCH_SPEC="torch==2.7.0+cu128"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        PYG_FIND="https://data.pyg.org/whl/torch-2.7.0+cu128.html"
    else
        TORCH_SPEC="torch==2.6.0+cu124"
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        PYG_FIND="https://data.pyg.org/whl/torch-2.6.0+cu124.html"
    fi
    ok "PyTorch target: $TORCH_SPEC from ${TORCH_INDEX##*/}"
else
    # Same reasoning on CPU: take pytorch.org's build so its ABI matches the
    # PyG wheels, rather than whatever conda-forge left behind.
    TORCH_SPEC="torch==2.6.0+cpu"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    PYG_FIND="https://data.pyg.org/whl/torch-2.6.0+cpu.html"
    warn "CPU backend — Stage 1 sampling will be very slow; keep --n-samples ≤ 10"
fi

# ---------------------------------------------------------------------------
# 1. micromamba
# ---------------------------------------------------------------------------
step "Installing micromamba"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/conda}"
mkdir -p "$MAMBA_ROOT_PREFIX"
MAMBA_BIN="/usr/local/bin/micromamba"
if [ -x "$MAMBA_BIN" ]; then
    ok "micromamba already installed"
else
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local bin/micromamba
    ok "micromamba $("$MAMBA_BIN" --version) installed"
fi
export PATH="/usr/local/bin:$PATH"
mm() { "$MAMBA_BIN" "$@"; }

SCORE_PREFIX="$MAMBA_ROOT_PREFIX/envs/score-env"
RAPIDOCK_PREFIX="$MAMBA_ROOT_PREFIX/envs/rapidock"

# ---------------------------------------------------------------------------
# 2. Submodule
# ---------------------------------------------------------------------------
step "Initialising the RAPiDock-Reloaded submodule"
if [ -f "third_party/RAPiDock/inference.py" ]; then
    ok "submodule already present"
else
    git submodule update --init --recursive
    ok "submodule initialised"
fi

# ---------------------------------------------------------------------------
# 3. Persistent cache (optional)
# ---------------------------------------------------------------------------
# RAPiDock pulls esm2_t33_650M_UR50D (~2.5 GB) through torch.hub on the first
# dock of every session. Symlinking the cache directory — rather than exporting
# TORCH_HOME — means it survives into the rapidock subprocess no matter how the
# notebook's environment is set up.
if [ -n "$CACHE_DIR" ]; then
    step "Wiring the persistent cache at $CACHE_DIR"
    mkdir -p "$CACHE_DIR/torch"
    mkdir -p "$HOME/.cache"
    if [ -L "$HOME/.cache/torch" ] || [ ! -e "$HOME/.cache/torch" ]; then
        rm -f "$HOME/.cache/torch"
        ln -s "$CACHE_DIR/torch" "$HOME/.cache/torch"
        ok "~/.cache/torch → $CACHE_DIR/torch (ESM-2 weights download once, not once per session)"
    else
        warn "~/.cache/torch already exists as a real directory — leaving it alone"
    fi
fi

# ---------------------------------------------------------------------------
# 4. score-env
# ---------------------------------------------------------------------------
# micromamba does not honour a yml's `pip:` block unless pip is itself a conda
# dependency, and score-env.yml does not list it. Rather than duplicating the
# package list here — which would rot the moment envs/*.yml changes — split the
# file at read time: conda deps go to micromamba, the pip block is installed
# afterwards with the env's own pip.
python3 -c "import yaml" 2>/dev/null || pip install -q pyyaml

split_yml() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys, yaml
src, conda_out, pip_out = sys.argv[1:4]
spec = yaml.safe_load(open(src))
conda_deps, pip_deps = [], []
for item in spec.get("dependencies", []):
    (pip_deps.extend(item["pip"]) if isinstance(item, dict) and "pip" in item
     else conda_deps.append(item))
if "pip" not in [str(d).split("=")[0].split("::")[-1] for d in conda_deps]:
    conda_deps.append("pip")
spec["dependencies"] = conda_deps
yaml.safe_dump(spec, open(conda_out, "w"), sort_keys=False)
open(pip_out, "w").write("\n".join(pip_deps) + ("\n" if pip_deps else ""))
PY
}

create_env() {
    _name="$1"; _yml="$2"; _prefix="$3"
    if [ -d "$_prefix" ] && [ "$FORCE" -eq 0 ]; then
        ok "$_name already exists (pass --force to recreate)"
        return 0
    fi
    [ "$FORCE" -eq 1 ] && [ -d "$_prefix" ] && mm env remove -y -n "$_name" >/dev/null 2>&1 || true
    _tmp="$(mktemp -d)"
    split_yml "$_yml" "$_tmp/conda.yml" "$_tmp/pip.txt"
    mm create -y -f "$_tmp/conda.yml"
    if [ -s "$_tmp/pip.txt" ]; then
        "$_prefix/bin/pip" install --no-input -r "$_tmp/pip.txt"
    fi
    rm -rf "$_tmp"
    ok "$_name created"
}

step "Creating score-env (Vina, OpenMM, RDKit, meeko — Python 3.11)"
create_env score-env envs/score-env.yml "$SCORE_PREFIX"

step "Installing HybriDock-Pep into score-env"
"$SCORE_PREFIX/bin/pip" install --no-input -e .
ok "hybridock-pep → $SCORE_PREFIX/bin/hybridock-pep"

# autogrid (AD4 grids) and openbabel (last-resort PDBQT conversion) are
# deliberately absent from score-env.yml because they make it unsolvable on ARM
# Linux. Colab is linux-64, where both exist — install them best-effort.
step "Installing optional score-env tooling (autogrid, openbabel)"
if [ "$LITE" -eq 1 ]; then
    # AD4 is off by default (the production ridge gives it w_ad4=0) and meeko
    # handles receptor prep, so neither of these is on the plain docking path.
    # obabel is only the fallback if meeko cannot run, and the liveness check
    # below reports that clearly if it happens.
    ok "--lite: skipping autogrid + openbabel ('--scoring ad4' unavailable)"
elif [ -x "$SCORE_PREFIX/bin/autogrid4" ] && [ -x "$SCORE_PREFIX/bin/obabel" ]; then
    ok "autogrid4 and obabel already present"
elif mm install -y -n score-env -c conda-forge 'autogrid>=4.2.9' 'openbabel>=3.1'; then
    ok "autogrid4 + obabel installed"
else
    warn "optional tooling failed to install — '--scoring ad4' unavailable, everything else unaffected"
fi

# Boost is a build-time dependency: score-env.yml pulls libboost-devel only so
# pip can compile the Vina extension. Once that is built, the 188 MB of C++
# headers under include/boost are dead weight — Vina links against the ~11 MB of
# shared libraries in lib/, which stay. This is the single largest thing --lite
# reclaims, and it is bigger than every data file in the repo combined.
# Re-running this script without --lite restores them if a rebuild ever needs them.
if [ "$LITE" -eq 1 ] && [ -d "$SCORE_PREFIX/include/boost" ]; then
    _boost_mb="$(du -sm "$SCORE_PREFIX/include/boost" 2>/dev/null | cut -f1)"
    if "$SCORE_PREFIX/bin/python3" -c "import vina" >/dev/null 2>&1; then
        rm -rf "$SCORE_PREFIX/include/boost"
        ok "--lite: dropped Boost headers after Vina built (${_boost_mb:-~188} MB reclaimed)"
    else
        warn "--lite: keeping Boost headers — Vina does not import, so it may still need rebuilding"
    fi
fi

# ---------------------------------------------------------------------------
# 5. rapidock env + PyTorch/PyG
# ---------------------------------------------------------------------------
if [ "$SKIP_RAPIDOCK" -eq 0 ]; then
    step "Creating rapidock (diffusion sampling stack — Python 3.10)"
    create_env rapidock envs/rapidock-env.yml "$RAPIDOCK_PREFIX"

    install_torch_stack() {
        _spec="$1"; _index="$2"; _pyg="$3"
        if [ -n "$_index" ]; then
            "$RAPIDOCK_PREFIX/bin/pip" install --no-input "$_spec" --index-url "$_index"
        else
            "$RAPIDOCK_PREFIX/bin/pip" install --no-input "$_spec"
        fi
        # torch-geometric is pure Python and wants its ordinary dependency
        # resolution. The four compiled extensions get --no-deps so that
        # resolving them can never pull a different torch in behind our back,
        # and --force-reinstall because a mismatched build already on disk
        # carries the same version number and would otherwise be kept.
        "$RAPIDOCK_PREFIX/bin/pip" install --no-input torch-geometric
        "$RAPIDOCK_PREFIX/bin/pip" install --no-input --no-deps --force-reinstall \
            torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f "$_pyg"
    }

    # Two independent things can be wrong here, and checking only one of them
    # is how a broken env reaches a dock:
    #
    #   1. The wheel has no kernels for this GPU. It imports fine and dies at
    #      the first kernel launch ("no kernel image is available for execution
    #      on the device"), so a real matmul is the only honest test.
    #   2. torch and the PyG extensions were built against different ABIs. torch
    #      itself is perfectly healthy — matmul included — and torch_scatter
    #      throws `undefined symbol` on import. Stage 1 then dies ~10 s in with
    #      a bare "subprocess exited with code 1".
    #
    # An earlier version of this script only did (1), passed, and shipped an env
    # that failed (2) at dock time. Both, or neither.
    stack_works() {
        "$RAPIDOCK_PREFIX/bin/python3" - <<'PY' >/dev/null 2>&1
import sys
import torch
if torch.cuda.is_available():
    x = torch.randn(64, 64, device="cuda")
    torch.mm(x, x).sum().item()
    torch.cuda.synchronize()
elif "+cpu" not in torch.__version__:
    sys.exit(1)          # a CUDA build that cannot see the GPU is a failure
import torch_scatter, torch_sparse, torch_cluster, torch_geometric  # noqa: F401
PY
    }

    step "Installing PyTorch + PyG into rapidock"
    install_torch_stack "$TORCH_SPEC" "$TORCH_INDEX" "$PYG_FIND"

    step "Verifying the sampling stack (CUDA kernels + PyG extensions)"
    if stack_works; then
        ok "torch and the PyG extensions agree${GPU_NAME:+ — kernels run on $GPU_NAME}"
    elif [ "$BACKEND" = "cuda" ]; then
        warn "$TORCH_SPEC does not work here — falling back to torch 2.6.0+cu124"
        install_torch_stack "torch==2.6.0+cu124" \
            "https://download.pytorch.org/whl/cu124" \
            "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
        if stack_works; then
            ok "fallback build works${GPU_NAME:+ — kernels run on $GPU_NAME}"
        else
            warn "the sampling stack is still broken. Stage 1 will fail. Diagnose with:" \
                 "$RAPIDOCK_PREFIX/bin/python3 -c 'import torch_scatter'"
        fi
    else
        warn "the sampling stack is broken on CPU — see the import error above"
    fi

    "$RAPIDOCK_PREFIX/bin/python3" - <<'PY' || warn "could not query the rapidock env's torch"
import torch
print(f"  torch {torch.__version__} | cuda={torch.version.cuda} "
      f"| available={torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY

    # ----------------------------------------------------------------------
    # Pre-warm RAPiDock's SO(3)/torus lookup tables.
    #
    # utils/so3.py and utils/torus.py build truncated-infinite-series tables in
    # MODULE-LEVEL code, so merely importing inference.py costs ~10 minutes of
    # single-threaded numpy (so3: 1000 eps x 2000 terms x 2000 omega; torus: a
    # 5001x5001 grid, ~200 MB per .npy). They cache to RELATIVE paths, so
    # without this the cost lands on the user's first dock -- silently, because
    # rapidock_runner.py routes non-progress stdout to logger.debug, leaving
    # "Generating poses..." on screen with no explanation for ten minutes.
    #
    # Done here, in the directory rapidock_runner.py pins as the sampling
    # subprocess's cwd, so the cache is actually found at dock time. Adds ~10
    # min to an install already advertised as 15-25 min, and takes the same
    # amount off the first dock.
    step "Pre-computing RAPiDock's SO(3)/torus tables (~10 min, once)"
    _rd="$REPO_ROOT/third_party/RAPiDock"
    if [ -f "$_rd/.p.npy" ] && [ -f "$_rd/.so3_cdf_vals2.npy" ]; then
        ok "lookup tables already cached"
    elif ( cd "$_rd" && "$RAPIDOCK_PREFIX/bin/python3" -c \
              "import sys; sys.path.insert(0, '.'); import utils.so3, utils.torus" ); then
        ok "lookup tables cached in third_party/RAPiDock"
    else
        warn "could not pre-compute the lookup tables -- the first dock will spend ~10 min on it"
    fi
fi

# ---------------------------------------------------------------------------
# 6. RAPiDock checkpoints (shipped in the repo — no download)
# ---------------------------------------------------------------------------
# These used to come from Zenodo, which made a first-time Colab install depend
# on a host outside GitHub and PyPI — and zenodo.org is blocked outright on some
# school and institutional networks, which turned this step into a hard stop for
# the exact audience the notebook is written for. Both files are 54 MB, so they
# are committed under weights/ and arrived with the clone in step 3.
#
# That also removes the reason to cache them on Drive: a re-clone is cheaper
# than a Drive round-trip, and the cache still holds the thing that actually
# matters between sessions, the 2.4 GB ESM-2 download.
step "Installing RAPiDock model weights"
if [ "$LITE" -eq 1 ]; then
    # rapidock_global.pt is read by exactly one code path: the pocket-search
    # pass that `dock --blind` runs when no --site is given
    # (sampling/pocket_search.py). Site-directed docking never touches it.
    bash scripts/install_weights.sh --lite || warn \
        "checkpoint install incomplete — see the message above"
else
    bash scripts/install_weights.sh || warn \
        "checkpoint install incomplete — see the message above"
fi

# longer_local.pt (peptides >=13 residues) has never been published; docking
# those peptides falls back to rapidock_local.pt with a warning.

# ---------------------------------------------------------------------------
# 7. Verify what the pipeline will actually reach for
# ---------------------------------------------------------------------------
step "Verifying the install"
"$SCORE_PREFIX/bin/python3" - <<'PY'
import importlib, sys
for mod, label in (("hybridock_pep", "hybridock_pep"), ("vina", "AutoDock Vina"),
                   ("openmm", "OpenMM"), ("rdkit", "RDKit"), ("meeko", "meeko")):
    try:
        importlib.import_module(mod)
        print(f"  \033[32m✓\033[0m {label}")
    except Exception as exc:
        print(f"  \033[31m✗\033[0m {label}: {exc}")
        sys.exit(1)
PY

# meeko's mk_prepare_receptor.py imports rdkit at module load: being on disk is
# not the same as being runnable, and a dead copy silently degrades receptor
# prep to the obabel fallback mid-dock instead of failing here.
if PREP="$SCORE_PREFIX/bin/mk_prepare_receptor.py"; [ -x "$PREP" ]; then
    if "$PREP" --help >/dev/null 2>&1; then
        ok "meeko receptor prep is runnable"
    else
        warn "mk_prepare_receptor.py is installed but not runnable — prep will fall back to obabel"
    fi
else
    warn "meeko's mk_prepare_receptor.py not found in score-env"
fi

if [ -x "$SCORE_PREFIX/bin/autogrid4" ]; then
    ok "autogrid4 present ('--scoring vina,ad4' available)"
fi

# Stage 1 loads this ~70 s in, after ESM-2 has been downloaded and the receptor
# graph built. Checking it here turns a late, expensive torch.load failure into
# an immediate one.
_CKPT_DIR="$REPO_ROOT/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
if [ "$SKIP_RAPIDOCK" -eq 0 ]; then
    if [ -s "$_CKPT_DIR/rapidock_local.pt" ]; then
        ok "rapidock_local.pt in place (docking ready)"
    else
        warn "rapidock_local.pt is missing from $_CKPT_DIR — Stage 1 will fail." \
             "Re-run this cell, or copy it there from weights/."
    fi
    if [ "$LITE" -eq 0 ] && [ ! -s "$_CKPT_DIR/rapidock_global.pt" ]; then
        warn "rapidock_global.pt is missing — 'dock --blind' will fail (ordinary docking is fine)"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Hand the notebook its paths
# ---------------------------------------------------------------------------
ENV_FILE="/content/hybridock_env.sh"
[ -d /content ] || ENV_FILE="$REPO_ROOT/hybridock_env.sh"
cat > "$ENV_FILE" <<EOF
# Written by scripts/colab_setup.sh — source this in any shell cell.
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
export HYBRIDOCK_PEP="$SCORE_PREFIX/bin/hybridock-pep"
export SCORE_PYTHON="$SCORE_PREFIX/bin/python3"
export RAPIDOCK_PYTHON="$RAPIDOCK_PREFIX/bin/python3"
export RAPIDOCK_DIR="$REPO_ROOT/third_party/RAPiDock"
# APPENDED, not prepended, and that matters. score-env/bin holds python3 and
# pip, so putting it first makes a plain \`python3\` in the Colab terminal mean
# score-env's 3.11 and \`pip install foo\` land in score-env instead of the
# notebook's own interpreter — silently, with no error to explain it. Appending
# keeps the system tools winning while score-env's binaries stay reachable by
# name. hybridock-pep resolves through the /usr/local/bin shim regardless, and
# the pipeline's own tool lookup does not depend on PATH order at all
# (toolpath.which searches sys.prefix/bin before \$PATH).
export PATH="\$PATH:$SCORE_PREFIX/bin"
EOF

# ---------------------------------------------------------------------------
# 9. Make the commands work in a plain shell
# ---------------------------------------------------------------------------
# The notebook can hold paths in Python variables; a terminal cannot. Without
# this, every shell — the Colab terminal, an xterm cell, a fresh `!bash` — needs
# the user to remember to source an env file first, and typing `hybridock-pep`
# gets "command not found" on a perfectly good install. Shim it onto PATH so the
# documented commands in README/INSTALL work verbatim, and set the env vars
# inside the shim so they hold no matter how the shell was started.
#
# Nothing here may be fatal. This is the last step of a 20-minute install, and
# everything the pipeline needs already works without it — the shims are a
# convenience. /usr/local/bin is writable on Colab (you are root) but this
# script only requires Linux, and the micromamba step above proves writability
# only when it actually installed rather than finding one already there. A
# permission error under `set -e` would throw away a good install over a
# nice-to-have.
step "Putting hybridock-pep on PATH"
SHIM_DIR="/usr/local/bin"
if ! { mkdir -p "$SHIM_DIR" 2>/dev/null && [ -w "$SHIM_DIR" ]; }; then
    warn "$SHIM_DIR is not writable — skipping the PATH shims." \
         "Run 'source $ENV_FILE' in each shell instead; that puts score-env's" \
         "bin on PATH and gives you the same commands."
    SHIM_DIR=""
fi

for _cmd in hybridock-pep hybridock-tui; do
    [ -n "$SHIM_DIR" ] || break
    if [ -x "$SCORE_PREFIX/bin/$_cmd" ]; then
        cat > "$SHIM_DIR/$_cmd" <<EOF
#!/usr/bin/env bash
# Generated by scripts/colab_setup.sh — do not edit; re-run the script instead.
export RAPIDOCK_PYTHON="\${RAPIDOCK_PYTHON:-$RAPIDOCK_PREFIX/bin/python3}"
export RAPIDOCK_DIR="\${RAPIDOCK_DIR:-$REPO_ROOT/third_party/RAPiDock}"
exec "$SCORE_PREFIX/bin/$_cmd" "\$@"
EOF
        chmod +x "$SHIM_DIR/$_cmd"
        ok "$_cmd → $SHIM_DIR/$_cmd"
    else
        warn "$_cmd not found in score-env — skipping its PATH shim"
    fi
done

# A Colab terminal starts an interactive bash that reads ~/.bashrc. Sourcing the
# env file there means SCORE_PYTHON and friends are set in every new terminal,
# not just the one that happened to run this script.
if ! grep -qs "hybridock_env.sh" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# HybriDock-Pep (scripts/colab_setup.sh)\n[ -f %s ] && . %s\n' \
        "$ENV_FILE" "$ENV_FILE" >> "$HOME/.bashrc"
    ok "~/.bashrc sources $ENV_FILE in new terminals"
else
    ok "~/.bashrc already wired"
fi

ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
step "Setup complete in ~${ELAPSED} min"
cat <<EOF
  hybridock-pep : on PATH (/usr/local/bin/hybridock-pep)
  hybridock-tui : on PATH — the guided terminal UI
  score python  : $SCORE_PREFIX/bin/python3
  rapidock py   : $RAPIDOCK_PREFIX/bin/python3
  env snippet   : $ENV_FILE

  From a terminal, in $REPO_ROOT:

    hybridock-pep crystal-score --receptor data/pdbs/1YCR_mdm2.pdb \\
        --peptide-pdb data/pdbs/1YCR_peptide.pdb --peptide ETFSDLWKLLPE

    hybridock-pep dock --peptide ETFSDLWKLLPE \\
        --receptor data/pdbs/1YCR_mdm2.pdb --site 25.20 -25.61 -7.97 --box 30 \\
        --n-samples 20 --output-dir runs/colab_demo

    hybridock-tui        # guided UI, no flags to memorize
EOF
