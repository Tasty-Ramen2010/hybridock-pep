# HybriDock-Pep — Installation Guide

This document walks you through setting up both conda environments and the
non-redistributable third-party tools required to run HybriDock-Pep end-to-end.

> **Platform summary:**
> | Platform | Stage 1 (RAPiDock) | Stage 2 (scoring/MM-GBSA) |
> |---|---|---|
> | Linux x86-64 + CUDA | ✅ CUDA (full speed) | ✅ |
> | WSL2 + CUDA passthrough | ✅ CUDA | ✅ |
> | macOS Apple Silicon (M1–M4) | ✅ MPS (~10× slower than CUDA) | ✅ (ADFRsuite via Rosetta 2) |
> | macOS Intel | ✅ CPU (slow, use `--n-samples 10`) | ✅ |
>
> `--input-poses` lets you skip Stage 1 entirely and supply pre-generated poses
> from a CUDA machine to run only Stage 2 scoring on any platform.

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

## Step 4 — Install ADFRsuite (required for Stage 2, non-redistributable)

ADFRsuite provides `prepare_receptor`, `autogrid4`, and `babel` (OpenBabel).
Licensed by The Scripps Research Institute — **cannot be bundled** with HybriDock-Pep.

**Download:** <https://ccsb.scripps.edu/adfrsuite/downloads/>

### Linux x86-64

```bash
tar xzf ADFRsuite_x86_64Linux_1.0.tar.gz
cd ADFRsuite_x86_64Linux_1.0
bash install.sh   # follow prompts; default install path is fine

# Add to PATH — append to ~/.bashrc for persistence:
export PATH="/path/to/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
```

### macOS (Rosetta 2)

ADFRsuite ships an x86_64 macOS binary that runs under Rosetta 2 on Apple Silicon:

```bash
# 1. Enable Rosetta 2 (one-time, already enabled on most M-series Macs):
softwareupdate --install-rosetta --agree-to-license

# 2. Download the macOS installer from the link above (ADFRsuite_*macOS*.tar.gz)
tar xzf ADFRsuite_MacOS_1.0.tar.gz
cd ADFRsuite_MacOS_1.0
bash install.sh

# 3. Add to PATH:
export PATH="/path/to/ADFRsuite_MacOS_1.0/bin:$PATH"
```

**macOS Gatekeeper:** If macOS blocks the binary with "cannot be opened because
the developer cannot be verified", run:

```bash
xattr -dr com.apple.quarantine /path/to/ADFRsuite_MacOS_1.0/bin/
```

Verify:

```bash
which prepare_receptor   # → ADFRsuite/bin/prepare_receptor
which autogrid4          # → ADFRsuite/bin/autogrid4
which babel              # → ADFRsuite/bin/babel
babel --version          # → OpenBabel 3.x.x
```

> **PATH note:** ADFRsuite ships a Python 2.7 binary named `python`. Always call
> `python3` explicitly in scripts to avoid shadowing the conda Python.

---

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
pytest -m slow                 # add integration tests (~2 min, requires ADFRsuite)
pytest --cov=hybridock_pep     # with coverage report
```

---

## macOS quick-start end-to-end

Once both environments are set up, a full dock run on macOS Apple Silicon:

```bash
# Stage 1 + 2: 20-pose run via MPS (~2 min on M3, ~25 min on Intel CPU)
hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 31.9 17.5 9.5 --box 20 \
    --n-samples 20 \
    --output-dir runs/pfldh_macos_test

# Stage 2 only — score poses generated on a CUDA machine
hybridock-pep dock \
    --input-poses /path/to/poses_from_cuda_machine/ \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 31.9 17.5 9.5 --box 20 \
    --output-dir runs/pfldh_scored
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `libpython2.7.so.1.0` error in `conda run` | ADFRsuite `python` shadows conda | Always use `python3` explicitly |
| `autogrid4` segfaults immediately | `AD4_parameters.dat` relative path | Already fixed in code; check ADFRsuite is on PATH |
| `No module named 'torch'` in rapidock env | PyTorch not installed (Step 2b skipped) | Re-run Step 2b for your platform |
| `CUDA capability sm_120 not compatible` | PyTorch < 2.6 | Re-run Step 2b with `torch==2.7.0` |
| `HIS residue has the wrong set of atoms` | pdbfixer edge case on RCSB PDB | Already handled gracefully in `receptor.py` |
| `babel: command not found` | ADFRsuite not on PATH | Add ADFRsuite `bin/` to PATH (Step 4) |
| `"cannot be opened because the developer cannot be verified"` (macOS) | Gatekeeper blocks ADFRsuite | `xattr -dr com.apple.quarantine /path/to/ADFRsuite_MacOS_1.0/bin/` |
| MPS fallback warnings in Stage 1 | Ops not yet on Metal | Normal — `PYTORCH_ENABLE_MPS_FALLBACK=1` already set by inference.py |
| `torch-scatter` ImportError on macOS | Wrong PyG install command used | Use the macOS pip command from Step 2b (no `+cu128`) |
| Stage 1 very slow on macOS Intel | No MPS, running on CPU | Expected; use `--n-samples 10` or use `--input-poses` bypass |
