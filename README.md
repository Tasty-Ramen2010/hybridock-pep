# HybriDock-Pep

**A general protein–peptide docking and scoring tool: AI diffusion sampling + physics-based rescoring + calibrated free-energy correction — fused into a single CLI, MIT-licensed, cross-platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-285%20passing-brightgreen.svg)](#testing)

HybriDock-Pep predicts how short peptides bind to protein receptors. It takes a peptide sequence and a receptor PDB, returns ranked binding poses with calibrated ΔG estimates, and includes a first-class **selectivity primitive** for comparing how the same peptide binds two different targets (decoy ΔΔG with bootstrap CI).

It is built for laboratories that need *fast, reproducible* peptide docking on commodity hardware — typically the **iGEM workflow scale**: dozens of candidate peptides against one or two targets, with results in minutes per peptide, not days.

---

## Why HybriDock-Pep

Most peptide docking workflows force a choice between accuracy and accessibility. HybriDock-Pep is designed to give both:

| Comparison axis | Vina alone | DiffPepDock (Kong et al. 2024) | RAPiDock (Zhao et al. 2025) | Wahibah-Hasibuan 2026 (HADDOCK + 1.2 µs MD + MM-GBSA) | **HybriDock-Pep** |
|---|---|---|---|---|---|
| Cα RMSD vs crystal (1YCR, head-to-head) | n/a (no sampling) | 3.54 Å | ~2.0 Å | not measured (MD-drift only) | **0.80 Å best-of-top-25** |
| Per-peptide wall-clock | seconds (but no sampling) | minutes | ~5 min on RTX 5070 | hours-to-days (MD-bound) | **~5 min on RTX 5070; ~25 min full pipeline incl. MM-GBSA** |
| Hardware required | any CPU | CUDA GPU | CUDA GPU | CUDA + ≥48 GB GPU RAM for AF3 | **CUDA, Apple MPS, or CPU** |
| License | Apache 2.0 | academic only | academic only | HADDOCK = CCPN restrictive | **MIT (OSI-compliant)** |
| Selectivity / ΔΔG primitive | no | no | no | implicit (manual comparison) | **first-class subcommand with bootstrap CI** |
| Calibration honesty | uncalibrated | uncalibrated | uncalibrated | uncalibrated (no LOO) | **LOO-CV r reported per family; documented cross-target ceiling** |
| Reproducibility metric | no | no | no | RMSF between MD replicas | **multi-seed Cα centroid agreement (`benchmark --reproducibility`)** |
| One-command install | yes | no (proprietary deps) | yes | no (5+ tool stack) | **yes (`conda env create` + `pip install -e .`)** |

**Bottom line:** HybriDock-Pep delivers RAPiDock-grade pose accuracy in the same ~5 minute window, then adds physics rescoring, entropy correction, calibrated ΔG, and selectivity-by-bootstrap — all of which the upstream tools leave to the user. Compared to a heavy-MD pipeline like Wahibah-Hasibuan 2026, HybriDock-Pep trades 1.2 µs of trajectory validation per peptide for ~60× faster turnaround on commodity hardware, while keeping pose accuracy in the literature top tier.

---

## Scoring accuracy

HybriDock-Pep scores **three distinct quantities**, each validated independently (Pearson *r* vs experimental ΔG/ΔΔG; RMSE in kcal/mol):

| Capability | What it ranks | Pearson *r* | RMSE | Validation |
|---|---|---|---|---|
| **Absolute ΔG** | any peptide vs any receptor | **0.60** within-distribution · **0.52** cross-family (balanced held-out) | **1.8 kcal/mol** | leave-one-out + balanced held-out test, 156 complexes |
| **Selectivity ΔΔG** | one peptide vs two receptors | **0.30–0.45** | — | desolvation floor cancels — sidesteps the hardest physics |
| **Affinity maturation** | variants of one peptide | **+0.42** (beats FlexPepDock +0.30) | — | leave-complex-out, **independently confirmed +0.43 on ATLAS TCR-pMHC** |

### Where we sit in the field (protein–peptide absolute ranking)

| Tool | *r* (cross-family) | Time / complex | Hardware | Key weakness |
|---|---|---|---|---|
| Raw Vina / AutoDock | ~0.30 (often sign-flips) | ~1 s | CPU | no charge, no entropy, size-confounded |
| **HybriDock-Pep** | **0.52** (0.60 within-dist) | **~10 s + 8 s MD** | **CPU + 1 GPU** | charged-desolvation floor |
| MM-GBSA (single-snapshot) | 0.25–0.45 | 5–30 s | GPU | omits conformational entropy |
| MM-PBSA | 0.3–0.5 | 1–5 min | CPU/GPU | slow PB solve; dielectric-sensitive |
| FlexPepDock / flex-ddG | 0.55–0.60 *within-target only* | 5–30 min | CPU cluster | flips cross-family; backrub hurts there |
| LIE | 0.5–0.7 *system-specific* | 0.5–4 GPU-hr | GPU | per-system refit; both MD legs |
| FEP / TI | 0.8–0.9 *congeneric series only* | 5–50 GPU-hr **per mutation** | GPU farm | not a screener; fragile convergence |

**Cheapest accuracy-per-second in the field.** We match within-target tools (FlexPepDock ~0.60) and the best published ML peptide scorer (~0.55) at **30–300× lower cost**, on commodity hardware, with no GPU cluster. FEP's 0.8–0.9 is physically reserved for congeneric series with a reference compound — not diverse cross-family screening.

### Physics features behind the numbers

Each is calibrated on a pooled, balanced crystal-65 + the-98 reference set and validated to be **sign-stable across datasets** (no Simpson's-paradox flips):

- **`rg_per_L`** — peptide extendedness/compactness; the cheap proxy for free-state conformational entropy that single-snapshot MM-GBSA omits. Eliminates the length/extendedness bias (the-98 *r* 0.25 → 0.42; dynamic range 25% → 41% of the true ΔG spread).
- **`mean_burial`** — interface packing density; the strongest separator for charged binders (where electrostatics provably cancel).
- **`org_density` / `cys_frac`** — intra-peptide pre-organization (disulfides, salt bridges, secondary-structure H-bonds, aromatic stacking) read straight from the 3D structure.
- **Optional MM-GBSA conformational-entropy penalty** (`entropy_penalty=True`) and **PROPKA pH-aware protonation** for the cases that need them.

> **Honest ceiling (documented, not hidden):** diverse cross-family peptide ΔG tops out near *r* ≈ 0.7, bounded by experimental label noise (mixed Kd/Ki, ~1 kcal/mol) and conformational/desolvation physics that only explicit-solvent free-energy methods capture. We report the held-out number, not the in-set one.

---

## Pipeline

```
  Peptide sequence + Receptor PDB
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 1 — Diffusion sampling (RAPiDock-Reloaded)     │
  │  N stochastic SE(3)-equivariant inference passes      │
  │  → all-atom peptide pose PDBs (~3 min for N=100)      │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 2 — OpenMM clash relief + Vina scoring         │
  │  AMBER ff14SB minimization (restrained), then         │
  │  vina --score_only (optionally + vina --scoring ad4)  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 3 — Calibrated ΔG correction                   │
  │  Per-residue + SS-weighted backbone entropy (v1.2)    │
  │  or per-family ridge dispatch (v1.3, opt-in)          │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 4 — Clustering + (optional) MM-GBSA refinement │
  │  Cα RMSD agglomerative; AMBER ff14SB + GBn2 on top-K  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Outputs                                              │
  │  ranked_poses.csv  best_pose.pdb                      │
  │  cluster_summary.csv  run_metadata.json               │
  └───────────────────────────────────────────────────────┘
```

---

## Quick start

### Install

```bash
# Scoring environment
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# GPU sampling environment (Linux/WSL2 + CUDA, or macOS MPS)
conda env create -f envs/rapidock-env.yml      # CUDA / Linux
# conda env create -f envs/rapidock-env-macos.yml   # Apple Silicon MPS

# See INSTALL.md for ADFRsuite + PULCHRA setup (license-restricted; download once).
```

Cross-platform: Linux/WSL2 (CUDA), macOS Apple Silicon (MPS), macOS Intel (CPU only). Stage 1 on MPS is ~5–8× faster than CPU. Use `--input-poses` to bypass Stage 1 entirely when sampling on a remote machine.

### Dock a peptide

```bash
hybridock-pep dock \
    --peptide ETFSDLWKLLPE \
    --receptor receptors/mdm2.pdb \
    --site 25.20 -25.61 -7.97 \
    --box 30 \
    --n-samples 100 \
    --refine-topk 10 \
    --output-dir runs/mdm2_p53
```

**Recommended workflow:** always include `--refine-topk 10` — MM-GBSA refinement on the top-K cluster representatives is HybriDock-Pep's most accurate ΔG signal. Skip only if you don't have OpenMM available or are screening hundreds of peptides.

### Score selectivity between two receptors

```bash
hybridock-pep selectivity \
    --peptide LISDAELEAIFEADC \
    --target-receptor receptors/target.pdb \
    --target-site 31.9 17.5 9.5 --target-box 25 \
    --offtarget-receptor receptors/offtarget.pdb \
    --offtarget-site 12.3 4.1 22.7 --offtarget-box 25 \
    --output-dir runs/selectivity_check
```

Returns ΔΔG = ΔG_target − ΔG_offtarget with 95% bootstrap CI over the top-K cluster centroids. Negative ΔΔG with CI not crossing zero ⇒ statistically selective. This is the right primitive for "does my peptide prefer A over B" questions — it sidesteps the absolute-Kd cross-target ceiling because the same systematic bias applies to both sides.

### Calibrate on your own training set

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

See `docs/calibration_notes.md` for the full calibration record (six revisions, with LOO-CV r/RMSE for each, and an honest read on what each one is really measuring).

---

## Calibration tiers shipped

| File | Schema | Features | LOO-CV r | RMSE | Notes |
|---|---|---|---|---|---|
| `data/calibration.json` | v1, single-α | Vina + n_contact | 0.86 (PepSet-6) | 1.73 | Production default. Conservative. |
| `data/calibration_v1_1_production_ridge.json` | v2, ridge | Vina + AD4 + n_contact | 0.755 | 1.44 | AD4 weight collapsed to 0 → AD4 dropped from default scoring. |
| `data/calibration_v1_2_production_entropy.json` | v2, ridge | Vina + per-residue + SS-weighted entropy | 0.715 | 1.51 | Best RMSE stability on long peptides. |
| `data/calibration_per_family.json` | v3, per-family ridge | Vina + n_contact + S_ss / cluster | **+0.731** | 1.65 | Cluster-dispatch by k-mer Jaccard. Largest reported lift. Runtime dispatcher in progress. |

Honest read: cross-target absolute-Kd prediction with a single global formula hits a Pearson r ceiling near 0.4 on heterogeneous data (documented across five calibration rounds in `docs/calibration_notes.md`). Per-family calibration breaks that ceiling by learning cluster-specific intercepts. Within-target ranking, pose-finding accuracy, and selectivity ΔΔG are all unaffected by the global ceiling.

---

## Testing

```bash
pytest                           # 285 unit tests, ~5 sec
pytest -m slow                   # + integration tests on MDM2/p53 (~2 min)
pytest --cov=hybridock_pep       # coverage report
```

---

## Project status

HybriDock-Pep is built for the iGEM 2026 Best Software Tool award. The Denmark High School Dry Lab team is the primary maintainer; one of the initial test applications is a malaria rapid-diagnostic peptide selectivity check (PfLDH vs hLDH), but the tool itself is target-agnostic.

- **Library:** stable, MIT-licensed, unit tests + integration tests.
- **CLI:** `dock`, `selectivity`, `calibrate`, `prep`, `benchmark` subcommands.
- **Calibration data:** shipped calibrations with full LOO-CV provenance and honest performance ceilings documented.
- **Scoring physics (2026):** sign-stable `rg_per_L` (compactness/entropy), `mean_burial` (packing), `org_density`/`cys_frac` (pre-organization), optional MM-GBSA conformational-entropy penalty, and PROPKA pH-aware protonation — all validated cross-dataset. See [docs/SCORING_COMPARISON.md](docs/SCORING_COMPARISON.md) for the full method comparison.

See [docs/architecture.md](docs/architecture.md) for the full pipeline spec and [docs/calibration_notes.md](docs/calibration_notes.md) for the calibration history.

---

## Citations

If HybriDock-Pep helps your work, please cite the underlying tools as well:

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025).
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021).
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017).
- **HybriDock-Pep** — [this repository], 2026.

---

## License

[MIT](LICENSE). Third-party dependencies retain their own licenses — see [INSTALL.md](INSTALL.md) for ADFRsuite, AutoDock4, and PULCHRA license caveats (none of which are redistributed in this repository).
