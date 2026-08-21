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
#   4. Downloads both RAPiDock checkpoints from Zenodo.
#   5. Optionally redirects the torch/ESM cache and the checkpoints at a
#      persistent --cache-dir (a Google Drive folder), so the ~2.5 GB ESM-2
#      download happens once per account rather than once per session.
#   6. Writes /content/hybridock_env.sh + prints the paths the notebook needs.
#
# Flags:
#   --cache-dir DIR     Persist ESM/torch weights + checkpoints here (e.g. a
#                       mounted Drive folder). Default: session-local only.
#   --backend cuda|cpu  Force a backend (default: auto-detect via nvidia-smi).
#   --skip-rapidock     score-env only (Stage 2 scoring / --input-poses runs).
#   --force             Recreate envs that already exist.
#   -h, --help          Show this help.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CACHE_DIR=""
FORCE=0
SKIP_RAPIDOCK=0
BACKEND=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cache-dir) CACHE_DIR="${2:?--cache-dir needs a path}"; shift 2 ;;
        --backend)   BACKEND="${2:?--backend needs cuda or cpu}"; shift 2 ;;
        --skip-rapidock) SKIP_RAPIDOCK=1; shift ;;
        --force)     FORCE=1; shift ;;
        -h|--help)   sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    mkdir -p "$CACHE_DIR/torch" "$CACHE_DIR/checkpoints"
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
if [ -x "$SCORE_PREFIX/bin/autogrid4" ] && [ -x "$SCORE_PREFIX/bin/obabel" ]; then
    ok "autogrid4 and obabel already present"
elif mm install -y -n score-env -c conda-forge 'autogrid>=4.2.9' 'openbabel>=3.1'; then
    ok "autogrid4 + obabel installed"
else
    warn "optional tooling failed to install — '--scoring ad4' unavailable, everything else unaffected"
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
fi

# ---------------------------------------------------------------------------
# 6. RAPiDock checkpoints
# ---------------------------------------------------------------------------
step "Downloading RAPiDock model weights"
WEIGHTS_DIR="third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
mkdir -p "$WEIGHTS_DIR"

_sha256() { sha256sum "$1" | awk '{print $1}'; }

# Both are required: rapidock_local.pt for ordinary site-directed docking,
# rapidock_global.pt for the --blind pocket search. Same Zenodo record and
# same checksums as install.sh.
fetch_ckpt() {
    _name="$1"; _want="$2"; _dest="$WEIGHTS_DIR/$_name"
    # With a persistent cache, the real file lives in the cache and the repo
    # gets a symlink — a fresh `git clone` per session then costs nothing.
    if [ -n "$CACHE_DIR" ]; then
        _cached="$CACHE_DIR/checkpoints/$_name"
        if [ -f "$_cached" ] && [ "$(_sha256 "$_cached")" = "$_want" ]; then
            ln -sfn "$_cached" "$_dest"
            ok "$_name restored from cache"
            return 0
        fi
        _dest="$_cached"
    fi
    if [ -f "$_dest" ] && [ "$(_sha256 "$_dest")" = "$_want" ]; then
        ok "$_name already present and verified"
    else
        curl -fsSL "https://zenodo.org/api/records/14193621/files/$_name/content" \
            -o "$_dest.part" || {
            warn "$_name download failed — re-run this cell to retry"
            rm -f "$_dest.part"
            return 0
        }
        mv "$_dest.part" "$_dest"
        if [ "$(_sha256 "$_dest")" = "$_want" ]; then
            ok "$_name downloaded and checksum-verified"
        else
            warn "$_name checksum mismatch — the file may be truncated; re-run this cell"
        fi
    fi
    if [ -n "$CACHE_DIR" ]; then ln -sfn "$_dest" "$WEIGHTS_DIR/$_name"; fi
    return 0
}

fetch_ckpt rapidock_local.pt \
    d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693
fetch_ckpt rapidock_global.pt \
    a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5

# longer_local.pt (peptides ≥13 residues) is not published on that Zenodo
# record; docking those peptides falls back to rapidock_local.pt with a warning.

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
export PATH="$SCORE_PREFIX/bin:\$PATH"
EOF

ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
step "Setup complete in ~${ELAPSED} min"
cat <<EOF
  hybridock-pep : $SCORE_PREFIX/bin/hybridock-pep
  score python  : $SCORE_PREFIX/bin/python3
  rapidock py   : $RAPIDOCK_PREFIX/bin/python3
  env snippet   : $ENV_FILE

  Try it:
    source $ENV_FILE
    \$HYBRIDOCK_PEP dock --peptide ETFSDLWKLLPE \\
        --receptor data/pdbs/1YCR_mdm2.pdb --site 25.20 -25.61 -7.97 --box 30 \\
        --n-samples 20 --output-dir runs/colab_demo
EOF
