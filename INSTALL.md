# HybriDock-Pep — Installation Guide

This document walks through setting up both conda environments and the
non-redistributable third-party tools required to run HybriDock-Pep end-to-end.

> **macOS ARM note:** Stage 2 (scoring and analysis) runs natively on macOS ARM.
> Stage 1 (GPU sampling via RAPiDock) requires a CUDA-capable GPU and cannot run
> on Apple Silicon. Use `--input-poses` to skip Stage 1 on macOS and supply
> pre-generated poses from a Linux/CUDA machine.

---

## Prerequisites

- **GPU (for Stage 1):** NVIDIA RTX 5070 or any Blackwell-generation card with
  compute capability >= 12.0. Driver >= 550 recommended for CUDA 12.8 support.
  Stage 2 (scoring) runs on any modern CPU.
- **conda:** [Miniforge](https://github.com/conda-forge/miniforge/releases) is
  strongly preferred over Anaconda for conda-forge compatibility. Any conda >=
  23.x works.
- **Disk space:** Allow ~20 GB total for both environments (PyTorch + CUDA libs
  dominate).
- **Operating system:** Linux (full pipeline), macOS ARM (Stage 2 only), WSL2
  (full pipeline with CUDA passthrough).

---

## Step 1 — Create the scoring environment

The `score-env` contains the physics-based scoring stack (Vina, OpenMM,
scikit-learn, etc.) and the HybriDock-Pep package itself.

```bash
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .
```

After this step, `hybridock-pep --help` should print the CLI usage.

---

## Step 2 — Create the GPU sampling environment

The `rapidock-env` contains the RAPiDock diffusion model stack (PyTorch 2.7,
CUDA 12.8, PyG). The RAPiDock commit SHA is pinned in Phase 4 of development;
until then the placeholder in `envs/rapidock-env.yml` must be replaced manually.

```bash
conda env create -f envs/rapidock-env.yml
```

> **Note:** The first solve can take several minutes due to the PyTorch/CUDA
> channel priority resolution. If the solve stalls, try:
> `conda config --add channels pytorch && conda config --add channels nvidia`

The driver script (`hybridock_pep/driver.py`) invokes RAPiDock via
`conda run -n rapidock-env` — you do not need to activate `rapidock-env`
manually during normal use.

> **Activation order:** `score-env` is the active environment for all normal use
> (`hybridock-pep` commands, running tests, calibration). `rapidock-env` is invoked
> automatically by the driver via `conda run -n rapidock-env` — do not activate it
> manually during docking runs. Activating `rapidock-env` directly will hide the
> `hybridock-pep` CLI (it is installed in `score-env`, not `rapidock-env`).

---

## Step 3 — Install ADFRsuite (required, non-redistributable)

ADFRsuite provides `prepare_receptor4.py`, `prepare_ligand4.py`, and
`autogrid4`, which are required for Stage 2 receptor/ligand preparation and
AutoDock4 grid generation. ADFRsuite is licensed by The Scripps Research
Institute and **cannot be bundled** with HybriDock-Pep (see CLAUDE.md §2.6).

**Download:** <https://ccsb.scripps.edu/adfrsuite/downloads/>

Installation steps:

1. Download the appropriate installer for your OS from the link above.
2. Run the installer (e.g., `bash ADFRsuite_x86_64Linux_1.0.tar.gz.run`).
3. Accept the license agreement when prompted.
4. Add the ADFRsuite `bin/` directory to your `PATH`. For example:

   ```bash
   export PATH="/path/to/ADFRsuite/bin:$PATH"
   ```

   Add this line to your shell profile (`.bashrc`, `.zshrc`, etc.) for
   persistence.

5. Verify that the following commands resolve:

   ```bash
   which prepare_receptor4.py
   which prepare_ligand4.py
   which autogrid4
   ```

ADFRsuite must be on `PATH` whenever `score-env` is active.

---

## Step 3.5 — PULCHRA v3.04 (Side-Chain Reconstructor)

PULCHRA rebuilds all-atom side chains from Cα-only traces. Version 3.04 is required
exactly — v3.07 produces incomplete aromatic side-chain atoms (documented bug in CLAUDE.md §2.3).
Bioconda ships 3.06/3.07; build from source:

```bash
# Download PULCHRA 3.04 source
wget https://cssb.biology.gatech.edu/sites/default/files/pulchra_308.tgz
# (use the v3.04 source if the above ships a later version — verify with: pulchra --version)
tar xzf pulchra_308.tgz
cd pulchra_308/src
make

# Add to PATH (add to ~/.bashrc or conda activate script):
export PATH="$PWD:$PATH"

# Verify version:
pulchra --version   # must print: PULCHRA 3.04
```

> **Version check:** If `pulchra --version` reports anything other than 3.04, rebuild from
> the correct source archive. The aromatic side-chain bug in v3.07 produces incorrect
> PDBQT files and degrades docking accuracy silently.

---

## Step 4 — Install PyRosetta (optional, license-restricted)

PyRosetta is used by RAPiDock's optional post-relax step. This step is
**skipped by default** in HybriDock-Pep due to a known ref2015 alignment
failure on C-terminal cysteine residues (see CLAUDE.md §2.5 and PDF §16.1).
Phase 1 does not depend on PyRosetta.

If you need the relax step for a non-cysteine peptide, obtain a license and
install into `rapidock-env`:

**Download (academic license required):** <https://www.pyrosetta.org/downloads>

```bash
conda activate rapidock-env
pip install pyrosetta-*.whl
```

Replace `pyrosetta-*.whl` with the actual filename downloaded from the
PyRosetta site. PyRosetta is NOT available via conda-forge or PyPI and must
be installed from the licensed wheel file.

---

## Step 5 — Verify

Once Steps 1–3 are complete, activate the scoring environment and run the
dependency smoke test:

```bash
conda activate score-env
bash scripts/smoke_test.sh
```

The script checks:

1. **CUDA compute capability >= 12.0** — warns on macOS ARM (expected), fails
   if a non-Blackwell NVIDIA GPU is detected.
2. **`prepare_receptor4.py` on PATH** — fails with an ADFRsuite download link
   if missing.
3. **AutoDock Vina >= 1.2.5** — fails with the upgrade command if the version
   is below the required minimum.

A fully configured machine exits with code 0 and three `[PASS]` lines:

```
[PASS] score-env: hybridock-pep CLI installed
[PASS] rapidock-env: CUDA capability >= 12.0
[PASS] ADFRsuite: prepare_receptor4.py on PATH
```

If any line shows `[FAIL]`, follow the fix in the corresponding INSTALL.md step above.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `PackagesNotFoundError: pytorch-cuda=12.8` | Old conda or missing nvidia channel | `conda config --add channels nvidia && conda update conda` |
| `prepare_receptor4.py: command not found` | ADFRsuite not on PATH | See Step 3 |
| `vina: command not found` after score-env create | pip install may have failed | `conda activate score-env && pip install 'vina>=1.2.5'` |
| `[FAIL] CUDA compute capability X.Y < 12.0` | GPU is pre-Blackwell | Use an RTX 5070 or newer for Stage 1 |
| `SyntaxError` in rapidock-env at runtime | Wrong Python version activated | Ensure `rapidock-env` uses Python 3.9 (`python --version`) |
