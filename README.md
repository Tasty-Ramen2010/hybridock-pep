# HybriDock-Pep

HybriDock-Pep is a hybrid peptide docking tool for the iGEM 2026 Best Software Tool award.
It combines RAPiDock diffusion-model pose generation (100 stochastic passes) with physics-based
rescoring via AutoDock Vina, AutoDock4 electrostatics, and a backbone entropy correction —
producing ranked poses with a calibrated ΔG estimate more accurate than Vina alone.

**Target application:** Malaria rapid-diagnostic peptide LISDAELEAIFEADC targeting PfLDH
(PDB 1CZB) over hLDH (PDB 1I0Z) — providing binding selectivity evidence for iGEM 2026.

---

## Architecture

HybriDock-Pep runs in two stages:

1. **Stage 1 (GPU, rapidock-env):** RAPiDock generates N=100 all-atom peptide pose PDBs via
   stochastic diffusion inference on the receptor structure.
2. **Stage 2 (CPU, score-env):** Each pose is independently scored by AutoDock Vina
   (`--score_only`) and AutoDock4 (`--scoring ad4`), combined with a backbone entropy
   correction (ΔS = α × n_residues), and clustered by contact-zone Cα RMSD.

The driver script orchestrates both stages via `conda run -n rapidock-env` subprocess calls.
No Python objects cross the subprocess boundary — only file paths and integer flags.

See [docs/architecture.md](docs/architecture.md) for the full module map and data flow diagram.

---

## Prerequisites

- **NVIDIA GPU (Stage 1):** Blackwell-generation card (RTX 5070 or newer, compute capability
  >= 12.0). Driver >= 550 for CUDA 12.8. Stage 2 runs on any modern CPU.
- **conda:** [Miniforge](https://github.com/conda-forge/miniforge/releases) preferred.
  Any conda >= 23.x works.
- **ADFRsuite:** Download from <https://ccsb.scripps.edu/adfrsuite/downloads/>
  (provides `prepare_receptor4.py` and `autogrid4`). Add `bin/` to PATH.
- **PULCHRA v3.04:** Required for side-chain reconstruction. Build from source — see
  [INSTALL.md Step 3.5](INSTALL.md#step-35--pulchra-v304-side-chain-reconstructor).
  v3.07 (Bioconda) has an aromatic side-chain bug; use v3.04 exactly.
- **Disk space:** ~20 GB for both conda environments (PyTorch + CUDA dominate).

For complete setup instructions including ADFRsuite PATH configuration and smoke test
verification, see [INSTALL.md](INSTALL.md).

---

## Quick Install

```bash
# 1. Create the scoring environment (score-env)
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# 2. Create the GPU sampling environment (rapidock-env)
conda env create -f envs/rapidock-env.yml

# 3. Verify: smoke test should print three [PASS] lines
bash scripts/smoke_test.sh
```

> **macOS ARM:** Stage 2 (scoring) runs natively. Stage 1 (GPU sampling) requires
> a CUDA machine. Use `--input-poses` to supply pre-generated poses and skip Stage 1.

---

## CLI Reference

### `dock` — End-to-end docking run

```bash
hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor receptors/1czb.pdb \
    --site 22.5 14.1 38.7 \
    --box 20 \
    --n-samples 100 \
    --scoring vina,ad4 \
    --refine-topk 10 \
    --output-dir runs/pfldh_run1
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--peptide` | str (required) | — | Peptide AA sequence (single-letter codes) |
| `--receptor` | path (required) | — | Receptor PDB file |
| `--site X Y Z` | float×3 (required) | — | Grid box center in Angstroms |
| `--box` | float (required) | — | Grid box edge length in Angstroms |
| `--n-samples` | int | 100 | Number of RAPiDock passes; mutually exclusive with `--input-poses` |
| `--scoring` | str | `vina,ad4` | Comma-separated scoring backends (`vina`, `ad4`) |
| `--refine-topk` | int | None | Top-K for MM-GBSA refinement (v2 scope; accepted but not dispatched) |
| `--output-dir` | path (required) | — | Output directory (created if absent) |
| `--seed` | int | None | Random seed for deterministic sampling |
| `--input-poses` | path | None | Pre-generated poses directory; skips Stage 1 |
| `--calibration` | path | `data/calibration.json` | Path to entropy calibration file |

### `calibrate` — Fit entropy correction parameters

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --pdbs-dir data/pdbs/ \
    --output calibration.json
```

### `benchmark` — Run accuracy benchmark suite

```bash
hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --baselines vina,adcp,rapidock \
    --report benchmark_report.md
```

### `prep` — Prepare receptor PDBQT

```bash
hybridock-pep prep \
    --receptor receptors/1czb.pdb \
    --output-dir receptors/
```

---

## Expected Output Files

After a successful `hybridock-pep dock` run, the `--output-dir` contains:

| File | Description |
|------|-------------|
| `ranked_poses.csv` | Top-10 poses with hybrid score, Vina score, AD4 score, entropy correction, cluster ID |
| `best_pose.pdb` | Centroid of the top-ranked cluster (not necessarily the top individual scorer) |
| `cluster_summary.csv` | Per-cluster mean, std, and 95% CI of hybrid score |
| `convergence_plot.png` | Running mean ± σ of hybrid score vs. number of poses N |
| `silhouette_plot.png` | Cluster quality scores across cluster counts k |
| `run_metadata.json` | Full provenance: git SHA, RAPiDock SHA, all CLI args, seeds, software versions, receptor SHA256 |
| `poses/pose_*.pdb` | All N raw pose PDB files from Stage 1 |
| `pdbqt/pose_*.pdbqt` | PDBQT versions of all poses (intermediate; used by Vina/AD4) |

---

## Running Tests

```bash
# Fast unit tests (no GPU, no ADFRsuite required)
pytest

# Include slow integration test (MDM2/p53, requires score-env stack)
pytest -m slow

# With coverage report
pytest --cov=hybridock_pep

# Specific module tests
pytest tests/test_scoring.py -x -v
```

The integration test (`pytest -m slow`) runs the full pipeline on the MDM2/p53 complex
(PDB 2OY2, peptide ETFSDLWKLLPE) using fixture poses in `tests/fixtures/mdm2_p53/`.
Expected: corrected ΔG < −3 kcal/mol. If not, something in the rescoring pipeline is broken.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'pdbfixer'` | Running tests in base Python env, not score-env | `conda activate score-env` then re-run pytest |
| `RuntimeError: CUDA device capability 12.0 required` | Wrong PyTorch/CUDA build for Blackwell GPU | Use PyTorch 2.7 + CUDA 12.8; see INSTALL.md Step 2 |
| `FileNotFoundError: prepare_receptor4.py` | ADFRsuite not on PATH | Download ADFRsuite, add `ADFRsuite_x86_64Linux_1.0/bin` to PATH |
| `pulchra: command not found` or wrong version | PULCHRA not built from source or wrong version | Build PULCHRA v3.04 from source; v3.07 has aromatic side-chain bug (CLAUDE.md §2.3) |
| `ImportError: cannot import name 'Vina'` | Using score-env commands inside rapidock-env | Always run `hybridock-pep` commands in score-env; rapidock-env is only for Stage 1 subprocess |
| Stage 1 fails on macOS | No CUDA on Apple Silicon | Use `--input-poses` to skip Stage 1; generate poses on a CUDA machine |

---

## License

HybriDock-Pep source code is released under the [MIT License](LICENSE).

Third-party dependencies retain their own licenses. Notable:
- Meeko (LGPL-2.1) — used via dynamic import; LGPL library exception applies
- AutoDock Vina (Apache-2.0) — see [pypi.org/project/vina](https://pypi.org/project/vina/)
- ADFRsuite — non-redistributable; download from <https://ccsb.scripps.edu/adfrsuite/downloads/>

See [docs/licenses.txt](docs/licenses.txt) for the full dependency audit.

## Citation

If you use HybriDock-Pep in your work, please cite:

> HybriDock-Pep: Hybrid ML + Physics Peptide Docking.
> Denmark High School iGEM Team 2026.
> https://github.com/[repo-url]

RAPiDock (Stage 1 generative model):
> Zhao et al. *Nature Machine Intelligence* 7:1308 (2025). DOI: 10.1038/s42256-025-01234-5
