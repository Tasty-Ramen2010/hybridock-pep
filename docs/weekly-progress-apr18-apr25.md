# HybriDock-Pep — Weekly Progress Report
**Period:** April 18–25, 2026  
**Prepared by:** Ram (Dry Lab Member, Denmark High School iGEM Team)  
**For:** iGEM Captains

---

## Overview

This week we went from a blank repository to a working five-phase software pipeline for HybriDock-Pep — a hybrid peptide docking tool being built for the iGEM 2026 Best Software Tool award. The tool combines RAPiDock (a diffusion generative model) with physics-based rescoring via AutoDock Vina and AutoDock4 to rank peptide binding poses more accurately than any single method alone.

In seven days: **86 commits, ~2,950 lines of production Python, 7 test files, 136 passing tests, and 5 of 8 planned pipeline phases fully implemented and verified.**

---

## What HybriDock-Pep Does

The tool is a two-stage docking pipeline targeting malaria diagnostics — specifically, optimizing the peptide `LISDAELEAIFEADC` that we designed to bind PfLDH (the malaria enzyme) over the human version hLDH.

1. **Stage 1 — Pose Generation (GPU):** RAPiDock runs 100 stochastic inference passes on an RTX 5070 to generate 100 all-atom peptide pose PDBs.
2. **Stage 2 — Rescoring (CPU):** Each pose is scored with AutoDock Vina, AutoDock4 (which adds a charge signal Vina ignores), and an entropy correction. The top candidates can optionally be refined with MM-GBSA via OpenMM for highest-accuracy ΔG estimates.
3. **Output:** Ranked CSV, best-pose PDB, convergence plot, cluster dendrogram, and a full run metadata JSON.

---

## Phase-by-Phase Progress

### Phase 1 — Foundation (Apr 18–19)
*Goal: Repository skeleton, conda environments, core data models.*

- Created the two-environment architecture: `rapidock-env` (Python 3.9, PyTorch 2.7, CUDA 12.8 for the RTX 5070's Blackwell GPU) and `score-env` (Python 3.11 for all scoring and analysis). This separation is required because the two stacks are incompatible.
- Implemented core data models (`DockConfig`, `PoseRecord`, `ScoredPose`, `PoseFailure`) in `models.py` with Pydantic validation and full type hints.
- Added `smoke_test.sh` to verify that CUDA, ADFRsuite, and Vina are correctly installed on a fresh environment.
- Added `INSTALL.md` with links to ADFRsuite and AutoDock4 (not bundled — they have non-redistributable licenses).
- Set up `pyproject.toml` with the `hybridock-pep` entry point and package structure.

**Key decisions made:**
- PyTorch 2.7 + CUDA 12.8 (not the original 2.3/12.4) — this is the first release with native sm_120 (Blackwell) support, which is required for our GPU.
- `DockConfig` set to `frozen=True` (immutable) because it crosses the subprocess boundary between the two conda environments.

---

### Phase 2 — Receptor/Ligand Preparation (Apr 19–21)
*Goal: Wrappers for pdbfixer, prepare_receptor4.py, Meeko, and autogrid4.*

- **`prep/receptor.py`:** Runs pdbfixer (adds missing atoms/residues, removes non-water HETATM), then calls `prepare_receptor4.py` to produce the PDBQT file Vina and AD4 need. A custom `_filter_pdb_lines()` pre-filter was added to strip alternate conformations before pdbfixer sees them — without this, autogrid4 throws an "Unknown Receptor Type" error.
- **`prep/ligand.py`:** Batch PDBQT converter using Meeko. Runs in a `ProcessPoolExecutor` so all 100 poses are converted in parallel. Per-pose failures are collected as `PoseFailure` records rather than crashing the whole batch.
- **`prep/grids.py`:** Programmatically generates the Grid Parameter File (GPF) for autogrid4 and runs autogrid4 to produce the AD4 map files. Includes a hard abort if the `receptor.HD.map` file is missing — this prevents a silent failure mode in the AD4 scorer downstream.
- Added full test coverage: `TestReceptorPrep`, `TestLigandBatch`, `TestGrids`.
- Code review pass identified and fixed 5 warnings and 4 informational findings.
- Phase 2 verification: 14/14 requirements passed.

---

### Phase 3 — Scoring Core (Apr 21–23)
*Goal: Vina scorer, AD4 scorer, entropy correction, calibration script.*

This was the algorithmic core of the project.

- **`scoring/vina.py`:** Wraps the Vina Python API to score all 100 poses in batch (one grid computation, 100 pose evaluations). Includes a grid boundary check that warns when a pose centroid is outside the search box. Lazy import pattern so the module loads in environments where Vina isn't installed.
- **`scoring/ad4.py`:** Wraps `vina --scoring ad4` using the precomputed AD4 map files. Uses `load_maps()` (not `set_receptor()`) — a non-obvious API detail: the C++ binding raises a `RuntimeError` on `set_receptor` when `sf_name='ad4'`. Flags poses where the AD4 score is positive (repulsive/unphysical) as anomalies.
- **`scoring/entropy.py`:** Implements the backbone entropy correction: `ΔG_corrected = ΔG_vina + α × n_residues × RT`. The coefficient α is calibrated on a training set. RT is fixed at 0.592 kcal/mol (298K) in v1.
- **`scripts/calibrate_alpha.py`:** Reads pre-computed scores and known binding affinities from a training CSV, fits α and β via L-BFGS-B (minimizing RMSE vs. experimental pKd), and writes `data/calibration.json`.
- **`data/training_complexes.csv`:** Training dataset with PDB IDs, sequences, and pKd values.
- **`data/calibration.json`:** Fitted calibration parameters.
- Code review pass: 2 critical issues found and fixed (pdbqt_path None guard before grid boundary check; missing convergence check on L-BFGS-B).
- 25 scoring tests passing across all three scorers.

**Key decision:** The hybrid score formula: `ΔG_hybrid = w_vina × ΔG_vina + w_ad4 × ΔG_ad4 + entropy_correction`. AD4 is run in parallel with Vina (not as a fallback) because it provides the electrostatic charge signal that Vina explicitly ignores by design.

---

### Phase 4 — Sampling Integration (Apr 23–24)
*Goal: RAPiDock subprocess wrapper, PDB parsing, pose I/O, run metadata.*

- **`sampling/run_rapidock.py`:** The Python 3.9-compatible shim that runs inside `rapidock-env`. Critically: no Python 3.10+ syntax (`match`/`case`, `X | Y` unions) anywhere in this file. Hardcodes `fastrelax=False` to avoid the known PyRosetta/ref2015 alignment failure on C-terminal cysteine.
- **`sampling/rapidock_runner.py`:** The orchestrator that calls `conda run -n rapidock-env python run_rapidock.py` via `subprocess.Popen` with streaming output. All file paths passed across the conda boundary are converted to absolute paths — relative paths break unpredictably at subprocess boundaries.
- **`sampling/pose_io.py`:** PDB parser that extracts peptide sequences (SEQRES-first with ATOM-line fallback), validates poses, and handles the case where Biopython's parser disagrees with the raw PDB format.
- **`output/metadata.py`:** Writes `run_metadata.json` with git SHA, RAPiDock commit SHA, all CLI arguments, random seeds, software versions, receptor SHA256, and wallclock time — full reproducibility record.
- 18 test stubs → all GREEN; full suite at 120 passing tests + 1 skipped.

---

### Phase 5 — CLI & Driver (Apr 24)
*Goal: argparse CLI with all dock flags, driver orchestrating both pipeline stages.*

- **`cli.py`:** Full argparse CLI with the `dock` subcommand and 11 flags (`--peptide`, `--receptor`, `--site`, `--box`, `--n-samples`, `--scoring`, `--refine-topk`, `--output-dir`, `--seed`, `--input-poses`, `--verbose`). Input validation fires before any subprocess is spawned — you find out about a malformed FASTA in under a second, not 30 minutes into a GPU run. The `--input-poses` flag lets macOS users (who lack CUDA) run Stage 2 on pre-generated poses.
- **`driver.py`:** The top-level `run_dock()` orchestrator that sequences: receptor prep → ligand prep → Vina scoring → AD4 scoring → entropy correction → clustering → statistics → output writing. Handles the mutual-exclusion between `--n-samples` and `--input-poses`.
- TDD approach: RED gate test scaffolds written first (Phase 5 Plan 01), then CLI (Plan 02), then driver (Plan 03).
- Full suite: **136 tests passing**.

**Key decision:** Driver import is deferred until after `DockConfig` validation. Eager import caused `ImportError` in validation error-path tests because `driver.py` was a Wave 2 artifact.

---

## Where We Are Now

| Phase | Description | Status |
|-------|-------------|--------|
| 1 — Foundation | Environments, scaffold, data models | Complete |
| 2 — Preparation | Receptor, ligand, grid prep | Complete |
| 3 — Scoring Core | Vina, AD4, entropy, calibration | Complete |
| 4 — Sampling Integration | RAPiDock wrapper, PDB I/O, metadata | Complete |
| 5 — CLI & Driver | argparse CLI, pipeline orchestrator | Complete |
| **6 — Analysis & Plots** | RMSD clustering, statistics, figures | **In planning** |
| 7 — Integration & E2E | End-to-end tests, MDM2/p53 baseline | Not started |
| 8 — Documentation & Polish | README, tutorial notebook, iGEM wiki | Not started |

**Phase 6 is planned and ready to execute.** Architecture decisions are locked: AgglomerativeClustering with precomputed pairwise Cα RMSD, contact-zone residues only (terminal residues excluded — they dominate full-peptide RMSD and corrupt cluster quality), matplotlib for convergence curves and dendrogram.

---

## Technical Highlights

**What makes this non-trivial:**

1. **Two incompatible Python environments** coordinated via subprocess — not just a software convenience, but a hard requirement because the GPU inference stack (PyTorch 2.7, CUDA 12.8, PyG) is incompatible with the scoring stack (OpenMM, Meeko, ADFRsuite).

2. **GPU compatibility:** RAPiDock's original requirements pinned CUDA 11.5 / PyTorch 1.11. Our RTX 5070 (Blackwell, compute capability 12.0) requires CUDA 12.8 / PyTorch 2.7. We identified and resolved this incompatibility before writing any inference code.

3. **Scoring formula correctness:** AutoDock Vina ignores partial charges by design. Rather than attempting to patch Vina (which was considered and rejected), we run AutoDock4 in parallel — it uses Gasteiger charges explicitly and catches electrostatics-dominated binding that Vina misses.

4. **Known failure modes preempted:** The PyRosetta ref2015 / C-terminal cysteine alignment bug (§16.1 of the spec) is hardcoded off by default. The autogrid4 HD map missing-file failure mode is caught at prep time with a clear error.

---

## Test Coverage Summary

| Test File | What It Covers |
|-----------|---------------|
| `test_models.py` | DockConfig validation, PoseRecord/ScoredPose inheritance |
| `test_prep.py` | Receptor prep, ligand batch, grid generation |
| `test_scoring.py` | Vina batch scorer, AD4 scorer, entropy correction, calibration |
| `test_sampling.py` | RAPiDock runner, PDB parsing, pose I/O, metadata writing |
| `test_cli.py` | All 11 CLI flags, mutual exclusion, input validation |
| `test_driver.py` | run_dock() orchestration, error propagation |
| `test_output.py` | Metadata JSON structure, run record completeness |

**Total: 136 tests passing, 0 failing.**

---

## What's Next

**Phase 6 (Analysis & Plots)** is next — this implements the RMSD clustering, ensemble statistics, and matplotlib figures (convergence curve, cluster dendrogram). Once Phase 6 is done, the pipeline produces all the outputs described in our technical spec: `ranked_poses.csv`, `cluster_summary.csv`, `best_pose.pdb`, `convergence.png`, `dendrogram.png`, `run_metadata.json`.

After that, Phase 7 closes with the MDM2/p53 integration test (PDB 2OY2, peptide `ETFSDLWKLLPE`, known Kd ≈ 0.6 µM) and Phase 8 covers documentation and the iGEM wiki page.

We are on track for the repository freeze ≥ 2 weeks before the November Jamboree.

---

*Report generated April 25, 2026. Repository: `iGEMDryLab/unknown_software`. Branch: `master`.*
