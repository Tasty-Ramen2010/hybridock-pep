# HybriDock-Pep — Installation Guide

This document walks you through setting up both conda environments and the
non-redistributable third-party tools required to run HybriDock-Pep end-to-end.

> **Just want it installed?** Run `./install.sh` (Linux/WSL2/macOS) or
> `install.bat` (Windows) from the repo root — it automates every step below
> including receptor-prep tooling (see Step 4), and finishes by launching a
> guided terminal UI. The rest of this document is the manual walkthrough,
> useful if you want to understand or control each step individually, or if
> `install.sh` doesn't fit your setup.

> **Platform summary:**
> | Platform | Stage 1 (RAPiDock) | Stage 2 (scoring/MM-GBSA) |
> |---|---|---|
> | Linux x86-64 + CUDA | ✅ CUDA (full speed) | ✅ |
> | WSL2 + CUDA passthrough | ✅ CUDA | ✅ |
> | macOS Apple Silicon (M1–M4) | ✅ MPS (~10× slower than CUDA) | ✅ (native, no Rosetta) |
> | macOS Intel | ✅ CPU (slow, use `--n-samples 10`) | ✅ |
> | Native Windows (no WSL) | ❌ (needs the Linux/WSL2/macOS CUDA or MPS stack) | ✅ (`conda env create -f envs/score-env.yml` — CI-verified) |
>
> `--input-poses` lets you skip Stage 1 entirely and supply pre-generated poses
> from a CUDA machine to run only Stage 2 scoring on any platform, including
> native Windows.
>
> **Windows users:** `install.bat` sets up WSL2 and runs the full pipeline
> there (Stage 1 + Stage 2, CUDA passthrough) — this is the path to use if you
> want everything working with one script. **After it finishes, run everything
> else — `conda activate`, `hybridock-pep`, `./launch_ui.sh` — inside the
> WSL2/Ubuntu terminal** (Start menu → search "Ubuntu", or type `wsl` in any
> Windows terminal), not PowerShell/CMD; those commands aren't installed on
> the Windows side. `install.bat` is safe to run again any time (e.g. after a
> `git pull`) — it detects an existing WSL2 distro and skips conda
> environments / model weights that are already present rather than
> recreating them, and it does not need to be re-run for normal day-to-day
> use. If you only need Stage 2 scoring and would rather stay outside WSL2,
> `conda env create -f envs/score-env.yml` now works directly in a native
> Windows conda prompt/PowerShell (Vina is built from source there against a
> MinGW-w64 + Boost toolchain — see the `conda-platforms` Windows job in
> `.github/workflows/ci.yml` for exactly how CI does this, or run
> `scripts/patch_vina_windows_setup.py` yourself if installing outside conda).

---

## Prerequisites

- **conda:** [Miniforge](https://github.com/conda-forge/miniforge/releases)
  strongly preferred over full Anaconda. Any conda ≥ 23.x works.
- **Disk space:** ~20 GB for both environments (PyTorch + optional CUDA libs dominate).
- **For CUDA Stage 1:** NVIDIA driver ≥ 550, CUDA 12.8 runtime, compute capability ≥ 8.0
  (RTX 5070 is CC 12.0 — Blackwell-compatible via PyTorch 2.7+cu128).
- **For macOS Stage 1 (MPS):** macOS 12.3+ (Monterey) on Apple Silicon. MPS is automatic.

---

## Step 1 — Create the scoring environment (all platforms)

`score-env` contains Vina, OpenMM, scikit-learn, RDKit, meeko, pdbfixer, and the
HybriDock-Pep package itself.

```bash
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .
```

Verify: `hybridock-pep --help` prints the CLI usage.

---

## Step 2 — Create the GPU/inference sampling environment

`rapidock` contains the RAPiDock-Reloaded diffusion model stack (Python 3.10,
MDAnalysis, e3nn, RDKit, fair-esm). PyTorch + PyG are installed via pip after
the base env is created because the CUDA wheels are only on `download.pytorch.org`.

### Step 2a — Create the base conda env

**Linux / WSL2:**
```bash
conda env create -f envs/rapidock-env.yml
```

**macOS (Apple Silicon or Intel):**
```bash
conda env create -f envs/rapidock-env-macos.yml
```

### Step 2b — Install PyTorch + PyG (platform-specific)

**Linux / WSL2 with CUDA 12.8 (RTX 5070 / any CUDA 12.x GPU):**
```bash
conda run -n rapidock pip install torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n rapidock pip install \
    torch-geometric \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

**macOS Apple Silicon (MPS) or Intel (CPU):**
```bash
conda run -n rapidock pip install torch torchvision torchaudio

conda run -n rapidock pip install \
    torch-scatter torch-sparse torch-cluster torch_geometric
```

> **Why separate pip steps?** PyTorch CUDA wheels (`+cu128`) are only on
> `download.pytorch.org`, not conda-forge. Mixing them into the conda solve
> causes `PackagesNotFoundError`. The two-phase approach always works cleanly.

### Step 2c — Verify PyTorch sees your device

```bash
# Should print True + device name on CUDA machines
conda run -n rapidock python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Compute cap:', torch.cuda.get_device_capability(0))
# macOS Apple Silicon
if hasattr(torch.backends, 'mps'):
    print('MPS available:', torch.backends.mps.is_available())
"
```

Expected on RTX 5070: `CUDA: True, Device: NVIDIA GeForce RTX 5070, Compute cap: (12, 0)`
Expected on M-series Mac: `CUDA: False, MPS available: True`

---

## Step 3 — Initialise the RAPiDock-Reloaded submodule

RAPiDock-Reloaded is bundled at `third_party/RAPiDock/` as a git submodule.
If you cloned with `--recursive` it's already present; otherwise:

```bash
git submodule update --init --recursive
```

No `pip install` needed — the runner imports directly from that path.

### Step 3b — Download model weights (required)

The pre-trained checkpoint files (~55 MB each) are **not** in git.
Download from [Zenodo (RAPiDock checkpoints)](https://zenodo.org/records/14193621)
and place them at:

```
third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/
  rapidock_local.pt    ← required
  rapidock_global.pt   ← optional (only for --ckpt rapidock_global)
```

The runner raises `FileNotFoundError: rapidock_local.pt` if this is skipped.

> **Alternate weight path:** Set `RAPIDOCK_MODEL_DIR=/abs/path` to override.
> Set `RAPIDOCK_DIR=/abs/path` to override the submodule location entirely.

Verify the import chain works:

```bash
conda run -n rapidock python3 -c "
import sys; sys.path.insert(0, 'third_party/RAPiDock')
from utils.inference_parsing import get_parser
print('RAPiDock-Reloaded: OK')
"
```

---

## Step 4 — Receptor-prep tooling (no license, no manual download)

ADFRsuite is **not required**. Two pip/conda packages replace it, both with
native Apple Silicon builds:

| what it does | tool | install |
|---|---|---|
| receptor PDB → PDBQT | meeko's `mk_prepare_receptor.py` | already in `envs/score-env.yml` (`meeko>=0.7`) |
| AutoDock4 grid maps | `autogrid4` | `conda install -c conda-forge autogrid` |

`install.sh` sets both up for you. To do it by hand:

```bash
conda activate score-env
conda install -c conda-forge autogrid     # only needed for --scoring ad4
pip install 'meeko>=0.7'                  # usually already present
```

Verify:

```bash
which mk_prepare_receptor.py   # → score-env/bin/mk_prepare_receptor.py
which autogrid4                # → score-env/bin/autogrid4
```

> `AD4_parameters.dat` is **not** needed. A GPF's `parameter_file` directive is
> optional, and autogrid4 falls back to its built-in AD4.2 parameters. That file
> only ever shipped with ADFRsuite, and needing it was the sole reason ADFRsuite
> was a hard requirement.

### If you already have ADFRsuite

It is still used automatically when `prepare_receptor` is on PATH, which keeps
results identical to earlier installs. Nothing to change. Note that ADFRsuite
ships a Python 2.7 binary named `python`, so always call `python3` explicitly
when its `bin/` is on your PATH.

## Step 5 — Install PyRosetta (optional, rarely needed)

Used only for RAPiDock's optional post-relax step. **Default is OFF** in
HybriDock-Pep (CLAUDE.md §2.5 — C-terminal cysteine alignment bug). Skip this
unless you specifically need PyRosetta-based relaxation on non-cysteine peptides.

**Obtain license + wheel:** <https://www.pyrosetta.org/downloads>

```bash
conda activate rapidock
pip install pyrosetta-*.whl
```

---

## Step 6 — Verify the full installation

```bash
conda activate score-env
bash scripts/smoke_test.sh
```

**Linux/WSL2 expected output:**
```
[PASS] CUDA compute capability 12.0 >= 12.0
[PASS] prepare_receptor found on PATH
[PASS] AutoDock Vina Python API 1.2.x >= 1.2.5
Results: 3 passed, 0 warnings, 0 failed
```

**macOS expected output:**
```
[INFO] No CUDA GPU — MPS (Apple Silicon) or CPU will be used for Stage 1
[PASS] prepare_receptor found on PATH (Rosetta 2)
[PASS] AutoDock Vina Python API 1.2.x >= 1.2.5
Results: 2 passed, 1 info, 0 failed
```

Run the unit test suite:

```bash
pytest                         # fast unit tests (~5 s)
pytest -m slow                 # add integration tests (~2 min)
pytest --cov=hybridock_pep     # with coverage report
```

---

## macOS quick-start end-to-end

Once both environments are set up, a full dock run on macOS Apple Silicon:

```bash
# Stage 1 + 2: 20-pose run via MPS (~2 min on M3, ~25 min on Intel CPU)
hybridock-pep dock \
    --peptide LISAAALAAIFAAALAC \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 28.25 15.44 66.27 --box 30 \
    --n-samples 20 \
    --output-dir runs/pfldh_macos_test

# Stage 2 only — score poses generated on a CUDA machine
hybridock-pep dock \
    --input-poses /path/to/poses_from_cuda_machine/ \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 28.25 15.44 66.27 --box 30 \
    --output-dir runs/pfldh_scored
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `libpython2.7.so.1.0` error in `conda run` | a stray ADFRsuite `python` shadows conda | Always use `python3` explicitly |
| `autogrid4` segfaults immediately | relative `parameter_file` path | Already fixed in code (the line is omitted entirely unless ADFRsuite supplies an absolute path) |
| `No module named 'torch'` in rapidock env | PyTorch not installed (Step 2b skipped) | Re-run Step 2b for your platform |
| `CUDA capability sm_120 not compatible` | PyTorch < 2.6 | Re-run Step 2b with `torch==2.7.0` |
| `HIS residue has the wrong set of atoms` | pdbfixer edge case on RCSB PDB | Already handled gracefully in `receptor.py` |
| `babel: command not found` | OpenBabel missing | `conda install -c conda-forge openbabel` (only a last-resort fallback; meeko is preferred) |
| MPS fallback warnings in Stage 1 | Ops not yet on Metal | Normal — `PYTORCH_ENABLE_MPS_FALLBACK=1` already set by inference.py |
| `torch-scatter` ImportError on macOS | Wrong PyG install command used | Use the macOS pip command from Step 2b (no `+cu128`) |
| Stage 1 very slow on macOS Intel | No MPS, running on CPU | Expected; use `--n-samples 10` or use `--input-poses` bypass |
