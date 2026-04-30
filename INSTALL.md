# HybriDock-Pep — Installation Guide

This document walks you through setting up both conda environments and the
non-redistributable third-party tools required to run HybriDock-Pep end-to-end.

> **macOS ARM note:** Stage 2 (scoring and analysis) runs natively on macOS ARM.
> Stage 1 (GPU sampling via RAPiDock) requires a CUDA-capable GPU and cannot run
> on Apple Silicon. Use `--input-poses` to skip Stage 1 on macOS and supply
> pre-generated poses from a Linux/CUDA machine.

---

## Prerequisites

- **GPU (for Stage 1):** NVIDIA RTX 5070 or any Blackwell-generation card with
  compute capability ≥ 12.0. Driver ≥ 550 recommended (CUDA 12.8 runtime).
  Stage 2 (scoring) runs on any modern CPU.
- **conda:** [Miniforge](https://github.com/conda-forge/miniforge/releases)
  strongly preferred. Any conda ≥ 23.x works.
- **Disk space:** ~20 GB for both environments (PyTorch + CUDA libs dominate).
- **OS:** Linux x86-64 (full pipeline). WSL2 with CUDA passthrough also works.
  macOS ARM: Stage 2 only.

---

## Step 1 — Create the scoring environment

`score-env` contains the physics-based scoring stack (Vina, OpenMM, scikit-learn,
RDKit, meeko) and the HybriDock-Pep package itself.

```bash
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .
```

Verify: `hybridock-pep --help` prints the CLI usage.

---

## Step 2 — Create the GPU sampling environment

`rapidock` (note: **no** `-env` suffix) contains the RAPiDock diffusion model
stack. PyTorch + PyG must be installed via pip after the base env is created,
because the CUDA-specific wheels are not on conda-forge.

```bash
# 2a. Create the base env (Python, NumPy, MDAnalysis, e3nn, RDKit, fair-esm)
conda env create -f envs/rapidock-env.yml

# 2b. Install PyTorch 2.7 with CUDA 12.8 (Blackwell-compatible)
conda run -n rapidock pip install torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2c. Install PyG extensions matching the torch+CUDA version
conda run -n rapidock pip install torch-scatter torch-sparse torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

> **Why separate pip steps?** PyTorch's CUDA-specific wheels (`+cu128` builds)
> are only available from `download.pytorch.org`, not conda-forge. Mixing the
> conda solve with these pip URLs causes solver conflicts; two-phase install
> avoids this cleanly.

---

## Step 3 — Install RAPiDock from source

RAPiDock is not on PyPI. Clone and install into the `rapidock` env:

```bash
# The pinned SHA below is the last commit validated against HybriDock-Pep.
# Update this SHA when upgrading RAPiDock.
RAPIDOCK_SHA="main"   # replace with pinned SHA before submission

git clone https://github.com/huifengzhao/RAPiDock.git ~/RAPiDock

# No pip install needed — we import directly from source.
# Set RAPIDOCK_DIR so the runner can auto-detect the install:
echo 'export RAPIDOCK_DIR="$HOME/RAPiDock"' >> ~/.bashrc
source ~/.bashrc
```

> **Auto-detection:** If `RAPIDOCK_DIR` is not set, the runner searches
> `~/RAPiDock` automatically. Setting the env var is only required if you
> install RAPiDock to a non-standard location.

Verify the install:

```bash
conda run -n rapidock python3 -c "
import sys; sys.path.insert(0, '$HOME/RAPiDock')
from utils.inference_parsing import get_parser
print('RAPiDock OK — parser loaded')
"
```

---

## Step 4 — Install ADFRsuite (required, non-redistributable)

ADFRsuite provides `prepare_receptor`, `autogrid4`, and `babel`. It is licensed
by The Scripps Research Institute and **cannot be bundled** with HybriDock-Pep.

**Download:** <https://ccsb.scripps.edu/adfrsuite/downloads/>

```bash
# Example for Linux x86-64 (filename varies by version):
tar xzf ADFRsuite_x86_64Linux_1.0.tar.gz
cd ADFRsuite_x86_64Linux_1.0
bash install.sh    # follow the prompts; default install path is fine

# Add bin/ to PATH — add this to ~/.bashrc for persistence:
export PATH="/path/to/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
```

Verify:

```bash
which prepare_receptor   # must resolve to ADFRsuite/bin/prepare_receptor
which autogrid4          # must resolve
which babel              # must resolve (OpenBabel bundled with ADFRsuite)
```

> **PATH ordering:** ADFRsuite's `bin/` must NOT shadow conda's `python3`. The
> default PATH ordering (ADFRsuite first, then conda) is intentional for
> `prepare_receptor` and `autogrid4`, but our driver always calls `python3`
> explicitly — never `python` — to avoid the bundled Python 2.7 wrapper.

---

## Step 5 — Install PyRosetta (optional)

PyRosetta is used by RAPiDock's optional post-relax step. **This step is
skipped by default** in HybriDock-Pep (see CLAUDE.md §2.5). Only needed for
non-cysteine peptides where higher-accuracy relaxation is desired.

**Obtain license + wheel:** <https://www.pyrosetta.org/downloads>

```bash
conda activate rapidock
pip install pyrosetta-*.whl
```

---

## Step 6 — Verify everything

```bash
conda activate score-env
bash scripts/smoke_test.sh
```

Expected output on a correctly configured machine:

```
[PASS] CUDA compute capability 12.0 >= 12.0 (Blackwell-compatible)
[PASS] prepare_receptor found on PATH
[PASS] AutoDock Vina Python API 1.2.7 >= 1.2.5 (score-env)

Results: 3 passed, 0 warnings, 0 failed
```

Then run the unit tests:

```bash
pytest                         # fast unit tests (~3 s)
pytest -m slow                 # include integration tests (~2 min)
pytest --cov=hybridock_pep     # with coverage report
```

All 171 fast tests should pass. The slow integration test requires the RTX 5070
and a downloaded receptor PDB.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `EnvironmentLocationNotFound: rapidock-env` | Old code referenced wrong env name | Env is named `rapidock`, not `rapidock-env` |
| `libpython2.7.so.1.0` error in conda run | ADFRsuite `python` in PATH shadows conda | Always use `python3`, not `python`, in scripts |
| `PackagesNotFoundError: pytorch-cuda=12.8` | Missing nvidia channel | `conda config --add channels nvidia` |
| `autogrid4` segfaults immediately | `AD4_parameters.dat` relative path | Already fixed in code; check ADFRsuite is on PATH |
| `No module named 'torch'` in rapidock env | PyTorch not installed via pip step | Re-run Step 2b above |
| `HIS residue has the wrong set of atoms` | pdbfixer edge case on some RCSB PDBs | Already handled gracefully in `receptor.py` |
| `babel: command not found` | ADFRsuite not on PATH | Add ADFRsuite `bin/` to PATH (Step 4) |
| `CUDA capability sm_120 not compatible` | PyTorch < 2.6 installed | Re-run Step 2b with `torch==2.7.0` |
