# HybriDock-Pep

**Truly hybrid peptide docking:** AI diffusion model pose generation + AutoDock Vina + AutoDock4 electrostatics + backbone entropy correction + MM-GBSA free energy refinement — five orthogonal signals fused into a single calibrated ΔG estimate.

Built for the **iGEM 2026 Best Software Tool** award by the Denmark High School Dry Lab team.

**Target application:** Malaria rapid-diagnostic peptide LISDAELEAIFEADC targeting PfLDH (PDB 1T2D) over hLDH (PDB 1I0Z) — providing computational binding selectivity evidence for the iGEM 2026 project.

---

## What makes it hybrid

Most docking tools use a single scoring function. HybriDock-Pep combines five independent sources of binding signal:

| Signal | What it captures | Implementation |
|--------|-----------------|----------------|
| **AI diffusion (RAPiDock)** | Learned structural priors from protein–peptide co-crystal database | Stage 1: 100 stochastic inference passes on RTX GPU |
| **AutoDock Vina** | Empirical shape complementarity + hydrophobics | `vina --score_only` per pose |
| **AutoDock4 electrostatics** | Gasteiger partial charges, H-bond geometry | `vina --scoring ad4` per pose |
| **Backbone entropy correction** | Conformational entropy penalty α × n_contact_residues | Calibrated on 6 PepSet crystal complexes (Pearson r = 0.860) |
| **MM-GBSA (optional)** | Molecular mechanics + implicit solvent ΔG decomposition | OpenMM AMBER ff14SB + GBn2, CUDA GPU, `--refine-topk K` |

The hybrid score is:

```
hybrid = vina + z_score(ad4) × w_ad4 + α × n_effective_residues
```

With optional MM-GBSA re-ranking of the top-K cluster representatives after clustering.

---

## Pipeline Overview

```
  Peptide sequence + Receptor PDB
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 1 — AI Diffusion (rapidock-env, GPU)         │
  │  RAPiDock × N=100 stochastic passes                 │
  │  → 100 all-atom peptide pose PDBs                   │
  └────────┬────────────────────────────────────────────┘
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 1.5 — OpenMM Clash Relief (optional)         │
  │  AMBER ff14SB + GBn2, harmonic restraints           │
  │  → minimized poses (reverted if >0.5Å displacement) │
  └────────┬────────────────────────────────────────────┘
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 2 — Physics Rescoring (score-env, CPU)       │
  │  • AutoDock Vina (--score_only)                     │
  │  • AutoDock4 (--scoring ad4, Gasteiger charges)     │
  │  • Contact-zone entropy correction (calibrated α)   │
  │  → hybrid_score per pose                            │
  └────────┬────────────────────────────────────────────┘
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 3 — Clustering                               │
  │  Kabsch-aligned contact-zone Cα RMSD                │
  │  Agglomerative + silhouette k-selection             │
  │  → cluster_id per pose, k_optimal, silhouette score │
  └────────┬────────────────────────────────────────────┘
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 3.5 — MM-GBSA Refinement (optional, GPU)     │
  │  AMBER ff14SB + GBn2, single-trajectory ΔG          │
  │  One representative per cluster, top-K by cluster   │
  │  CUDA → OpenCL → CPU fallback                       │
  │  → mmgbsa_dg per top-K pose                         │
  └────────┬────────────────────────────────────────────┘
           │
  ┌────────▼────────────────────────────────────────────┐
  │  Stage 4 — Output                                   │
  │  ranked_poses.csv  best_pose.pdb                    │
  │  cluster_summary.csv  convergence_plot.png          │
  │  silhouette_plot.png  run_metadata.json             │
  └─────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **NVIDIA GPU (Stage 1 + MM-GBSA):** Blackwell-generation card (RTX 5070 or newer, CC ≥ 12.0). Driver ≥ 550 for CUDA 12.8. Stages 2–4 run on any modern CPU.
- **conda:** [Miniforge](https://github.com/conda-forge/miniforge/releases) preferred. Any conda ≥ 23.x works.
- **ADFRsuite:** Download from <https://ccsb.scripps.edu/adfrsuite/downloads/> (provides `prepare_receptor` and `autogrid4`). Add `bin/` to PATH.
- **PULCHRA v3.04:** Required for side-chain reconstruction. Build from source — see [INSTALL.md](INSTALL.md). v3.07 (Bioconda) has an aromatic side-chain bug; use v3.04 exactly.
- **Disk space:** ~20 GB for both conda environments (PyTorch + CUDA dominate).

---

## Quick Install

```bash
# 1. Create the scoring environment (score-env)
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# 2. Create the GPU sampling environment (rapidock)
conda env create -f envs/rapidock-env.yml
# Then install PyTorch + PyG separately — see INSTALL.md Step 2

# 3. Verify: smoke test should print three [PASS] lines
bash scripts/smoke_test.sh
```

> **macOS ARM:** Stage 2 (scoring) runs natively. Stage 1 (GPU sampling) requires a CUDA machine. Use `--input-poses` to supply pre-generated poses and skip Stage 1.

---

## CLI Reference

### `dock` — End-to-end docking run

```bash
hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 31.9 17.5 9.5 \
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
| `--refine-topk` | int | None | Run MM-GBSA on top-K cluster representatives (AMBER ff14SB + GBn2, CUDA GPU) |
| `--mmgbsa-cpu-only` | flag | False | Force MM-GBSA to use CPU instead of CUDA (slower but no GPU dependency) |
| `--output-dir` | path (required) | — | Output directory (created if absent) |
| `--seed` | int | None | Random seed for deterministic sampling |
| `--input-poses` | path | None | Pre-generated poses directory; skips Stage 1 |
| `--calibration` | path | `data/calibration.json` | Path to entropy calibration file |
| `--no-minimize` | flag | False | Skip OpenMM clash-relief minimization of RAPiDock poses |

### `calibrate` — Fit entropy correction parameters

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

### `benchmark` — Run accuracy benchmark suite

```bash
hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --output-dir runs/benchmark
```

### `prep` — Prepare receptor PDBQT

```bash
hybridock-pep prep \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --output-dir data/pdbs/
```

---

## Expected Output Files

After a successful `hybridock-pep dock` run, `--output-dir` contains:

| File | Description |
|------|-------------|
| `ranked_poses.csv` | Top-10 poses sorted by hybrid score. Columns: `hybrid_score`, `vina_score`, `ad4_score`, `entropy_correction`, `mmgbsa_dg` (when `--refine-topk` used), `cluster_id`, `pose_filename` |
| `best_pose.pdb` | Best pose by MM-GBSA ΔG (if refined) or best cluster centroid by hybrid score |
| `cluster_summary.csv` | Per-cluster mean, std, 95% CI, and best pose index |
| `convergence_plot.png` | Running mean ± σ of hybrid score vs. number of top-N poses |
| `silhouette_plot.png` | Silhouette score vs. cluster count k; k_optimal annotated |
| `run_metadata.json` | Full provenance: git SHA, RAPiDock SHA, CLI args, seeds, software versions, receptor SHA256 |
| `poses/pose_*.pdb` | All N raw pose PDBs from Stage 1 |
| `poses_minimized/` | OpenMM clash-relieved poses (Stage 1.5, when minimization enabled) |
| `pdbqt/pose_*.pdbqt` | PDBQT versions of all poses (used by Vina/AD4) |

---

## Scoring Methodology

### Hybrid score formula

```
hybrid_score = vina_score
             + ensemble_z_score(ad4) × w_ad4       # when beta=0 (calibration mode)
             + alpha × (n_contact + gamma × n_non_contact)
```

- **`vina_score`**: AutoDock Vina `--score_only` output (kcal/mol). Captures shape, hydrophobics, H-bonds without partial charges.
- **`ad4` term**: AutoDock4 scoring (`--scoring ad4`) adds Gasteiger electrostatics — the explicit charge signal Vina lacks. Integrated via within-run z-score normalization to avoid absolute-scale calibration artifacts.
- **Entropy correction**: α × n_effective_residues penalizes conformational entropy lost upon binding. `n_effective = n_contact + γ × n_non_contact` where contacts are residues with ≥1 heavy atom within 5Å of the receptor. α and γ calibrated by L-BFGS-B on PepSet crystal poses.

### MM-GBSA (optional, `--refine-topk K`)

Selects one representative per cluster (best hybrid_score pose), takes the top-K clusters by mean score, and computes:

```
ΔG_bind = E(complex) − E(receptor) − E(peptide)
```

All energies evaluated with AMBER ff14SB + GBn2 implicit solvent in OpenMM. Single-trajectory approximation: minimize the complex once, extract component energies from the same minimized geometry. Runs on CUDA GPU (mixed precision) by default; automatic fallback to CPU.

### Calibration

Current calibration (`data/calibration.json`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 0.1 kcal/mol/residue | Entropy coefficient |
| `beta` | 0.0 | Direct AD4 blend (0 = ensemble z-score mode) |
| `gamma` | 0.2 | Non-contact residue weight |
| `ensemble_ad4_weight` | 0.3 | AD4 z-score blend weight |
| `pearson_r` | 0.860 | Training set correlation (6 PepSet complexes, crystal poses) |

Calibration was performed on crystal-quality poses (upper bound on scoring accuracy). Full pipeline (RAPiDock → scoring) benchmark r on the 10-complex held-out test set is pending the first full GPU run.

---

## Running Tests

```bash
# Fast unit tests — no GPU, no ADFRsuite required (~6 s)
pytest

# Full integration suite — requires score-env, Vina, ADFRsuite (~38 min)
pytest -m slow

# With coverage report
pytest --cov=hybridock_pep

# Specific module
pytest tests/test_scoring.py -x -v
```

**Integration test baseline (MDM2/p53):** PDB 1YCR, peptide ETFSDLWKLLPE, K_d ≈ 0.6 µM. Expected hybrid score < −3 kcal/mol. If it fails, the rescoring pipeline is broken.

**PepSet crystal-pose suite:** 11 protein families (PDZ, SH2, bromodomain, calmodulin, BCL-2, MDM2, kinase, amphipathic helix, ARM repeat, SH3, WW). All Vina and AD4 scores must be negative on crystal-quality poses. Last run: **45/45 passed**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'pdbfixer'` | Running in base Python env | `conda activate score-env` then re-run |
| `RuntimeError: CUDA device capability 12.0 required` | Wrong PyTorch/CUDA build | Use PyTorch 2.7 + CUDA 12.8; see INSTALL.md |
| `FileNotFoundError: prepare_receptor` | ADFRsuite not on PATH | Add `ADFRsuite_x86_64Linux_1.0/bin` to PATH |
| `pulchra: command not found` or wrong version | PULCHRA not v3.04 | Build v3.04 from source; v3.07 has aromatic side-chain bug |
| `ImportError: cannot import name 'Vina'` | Inside rapidock env | Always run `hybridock-pep` commands in score-env |
| Stage 1 fails on macOS | No CUDA on Apple Silicon | Use `--input-poses` to skip Stage 1 |
| MM-GBSA crashes with CUDA error | OpenMM/Blackwell incompatibility | Add `--mmgbsa-cpu-only` flag |
| autogrid4 very slow on large peptides | Grid computation is O(N_atoms × N_grid) | Use `--box 25` instead of 40; or skip AD4 with `--scoring vina` |

---

## License

HybriDock-Pep source code is released under the [MIT License](LICENSE).

Third-party dependencies retain their own licenses:
- Meeko (LGPL-2.1) — used via dynamic import; LGPL library exception applies
- AutoDock Vina (Apache-2.0)
- ADFRsuite — non-redistributable; download from <https://ccsb.scripps.edu/adfrsuite/downloads/>
- OpenMM (MIT)
- RAPiDock — see upstream repository license

See [docs/licenses.txt](docs/licenses.txt) for the full dependency audit.

---

## Citation

If you use HybriDock-Pep in your work, please cite:

> HybriDock-Pep: Hybrid AI + Physics Peptide Docking.
> Denmark High School iGEM Team 2026.
> https://github.com/[repo-url]

RAPiDock (Stage 1 generative model):
> Zhao et al. *Nature Machine Intelligence* 7:1308 (2025). DOI: 10.1038/s42256-025-01234-5
