# Roadmap: HybriDock-Pep — Milestone v1.0

## Overview

Build the complete hybrid peptide docking pipeline from the ground up: two conda environments, a physics-based scoring stack, a RAPiDock sampling integration, a full CLI, RMSD clustering with convergence diagnostics, and a benchmarked, documented release ready for iGEM 2026. Phases follow the natural build order — environment and contracts first, scoring before sampling, sampling before orchestration, analysis after data exists, documentation after the pipeline works.

## Phases

- [x] **Phase 1: Foundation** - Both conda environments set up, core dataclasses defined, package scaffold in place, smoke test passing
- [ ] **Phase 2: Preparation Pipeline** - Receptor and ligand PDBQT preparation wrappers working, autogrid4 AD4 map generation validated
- [ ] **Phase 3: Scoring Core** - Vina and AD4 per-pose scoring implemented, entropy correction with α calibration working
- [ ] **Phase 4: Sampling Integration** - RAPiDock subprocess wrapper running 100 poses, pose I/O parsing, provenance metadata written
- [ ] **Phase 5: CLI & Driver** - Single entry point with all subcommands, pre-run validation, seed propagation, full two-stage orchestration
- [ ] **Phase 6: Analysis & Plots** - Contact-zone Cα RMSD clustering, ensemble statistics, convergence and silhouette plots
- [ ] **Phase 7: Output & Integration** - Ranked CSV, best-pose PDB, ΔG reporting, MDM2/p53 integration test passing
- [ ] **Phase 8: Benchmark & Documentation** - Benchmark suite at target accuracy, install docs, architecture docs, license audit, tutorial notebook

## Phase Details

### Phase 1: Foundation
**Goal**: Both conda environments are installable and verified, the Python package structure exists in score-env, and core dataclasses define the interfaces every later module plugs into
**Depends on**: Nothing (first phase)
**Requirements**: TEST-01
**Success Criteria** (what must be TRUE):
  1. `conda env create -f envs/rapidock-env.yml` completes without error; `conda run -n rapidock-env python -c "import torch; print(torch.cuda.get_device_capability())"` reports (12, 0)
  2. `conda env create -f envs/score-env.yml && pip install -e .` completes without error and `hybridock-pep --help` prints usage
  3. `bash scripts/smoke_test.sh` exits 0 on a correctly configured machine and exits non-zero with a diagnostic message for any missing dependency (CUDA capability, ADFRsuite on PATH, Vina version)
  4. `DockConfig`, `PoseRecord`, `ScoredPose`, and `PoseFailure` dataclasses are importable from `hybridock_pep` with full type annotations
**Plans**: 2 plans
  - [x] 01-01-PLAN.md — Conda envs + smoke test + INSTALL.md (TEST-01)
  - [x] 01-02-PLAN.md — Python package scaffold + core models + CLI stub + tests

### Phase 2: Preparation Pipeline
**Goal**: Receptor PDB and per-pose ligand PDBQT preparation are fully automated and validated, and AD4 affinity maps are generated with a hard abort if the HD map is missing
**Depends on**: Phase 1
**Requirements**: PREP-01, PREP-02, PREP-03
**Success Criteria** (what must be TRUE):
  1. Given a raw receptor PDB, `hybridock-pep prep` produces a valid PDBQT via pdbfixer + ADFRsuite `prepare_receptor4.py` with no manual steps
  2. Given 100 pose PDB files, Meeko converts all 100 to PDBQT in a single stateless parallelized batch with no file left behind
  3. After running `autogrid4`, the pipeline verifies `receptor.HD.map` exists and aborts with a clear error message if it is missing
  4. Unit tests in `test_prep.py` cover receptor prep, ligand batch prep, and the HD-map guard; all pass
**Plans**: 4 plans
  - [x] 02-01-PLAN.md — PrepError + test fixtures + prep/receptor.py (PREP-01)
  - [ ] 02-02-PLAN.md — prep/ligand.py Meeko batch converter (PREP-02)
  - [ ] 02-03-PLAN.md — prep/grids.py GPF builder + autogrid4 + HD map guard (PREP-03)
  - [ ] 02-04-PLAN.md — tests/test_prep.py covering all three modules (PREP-01, PREP-02, PREP-03)

### Phase 3: Scoring Core
**Goal**: Every pose can be independently scored by Vina and AD4 in parallel, and the backbone entropy correction produces a calibrated hybrid score with α validated against the allowed range
**Depends on**: Phase 2
**Requirements**: SCORE-01, SCORE-02, SCORE-03
**Success Criteria** (what must be TRUE):
  1. Given a pose PDBQT and receptor PDBQT, `vina --score_only` via the Vina Python API returns a score; any pose with atoms outside grid boundaries is logged to `run_metadata.json` rather than silently dropped
  2. Given a pose PDBQT and AD4 affinity maps, `vina --scoring ad4` returns an AD4 score in parallel with the Vina score; any positive AD4 score is flagged as an anomaly in output
  3. Given a `calibration.json` with α in range 0.2–1.2 kcal/mol/residue, `entropy.py` computes the backbone entropy correction; α outside this range causes an immediate abort with a diagnostic message
  4. `scripts/calibrate_alpha.py` runs on the training set and writes a valid `calibration.json`; unit tests in `test_scoring.py` cover all three scoring modules
**Plans**: TBD

### Phase 4: Sampling Integration
**Goal**: RAPiDock runs 100 stochastic inference passes from score-env via a subprocess wrapper, all poses are parsed into PoseRecord objects, and every run writes complete provenance metadata
**Depends on**: Phase 3
**Requirements**: SAMP-01, SAMP-02
**Success Criteria** (what must be TRUE):
  1. Running the sampling stage via `conda run --no-capture-output -n rapidock-env` produces exactly 100 `poses/pose_*.pdb` files; GPU OOM errors surface in real time rather than being swallowed
  2. Passing `--seed N` to the sampling wrapper propagates the seed into RAPiDock and is recorded in `run_metadata.json`
  3. After any sampling run, `run_metadata.json` contains: git SHA, RAPiDock commit SHA, all CLI args, random seed, Vina version, OpenMM version, CUDA version, receptor SHA256, peptide sequence hash, and timestamp
  4. `pose_io.py` parses all 100 PDB files into a `list[PoseRecord]` without error on RAPiDock-format output
**Plans**: TBD

### Phase 5: CLI & Driver
**Goal**: A single `hybridock-pep` entry point exposes all four subcommands, validates all inputs before any subprocess is spawned, and the driver orchestrates both pipeline stages end-to-end
**Depends on**: Phase 4
**Requirements**: CLI-01, CLI-02, CLI-03
**Success Criteria** (what must be TRUE):
  1. `hybridock-pep dock`, `hybridock-pep calibrate`, `hybridock-pep benchmark`, and `hybridock-pep prep` all exist with help strings on every flag (with units)
  2. Providing an invalid peptide sequence, a missing receptor PDB, or out-of-range site coordinates causes an error with a clear message in under 1 second — before any subprocess is spawned
  3. `hybridock-pep dock --seed 42` produces the same output on two successive runs (modulo CUDA nondeterminism); the metadata JSON notes the seed
  4. `driver.py` orchestrates Stage 1 (rapidock-env subprocess) and Stage 2 (score-env in-process) in sequence and passes a `DockConfig` through both; if `poses/pose_*.pdb` already exist and `--skip-sampling` is not yet a v1 flag, the driver still completes without error
**Plans**: TBD

### Phase 6: Analysis & Plots
**Goal**: Completed poses are clustered by binding mode using contact-zone Cα RMSD, cluster quality is quantified, ensemble statistics are computed, and all diagnostic plots are generated
**Depends on**: Phase 5
**Requirements**: ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05
**Success Criteria** (what must be TRUE):
  1. Clustering computes pairwise Cα RMSD over contact-zone residues only (not full peptide) using agglomerative clustering with average linkage and precomputed metric; silhouette score is reported in the run output
  2. `cluster_summary.csv` is written with per-cluster mean, std, and 95% CI of the hybrid score
  3. `convergence_plot.png` exists after a run and shows running mean ± σ of hybrid score vs. number of poses N
  4. `silhouette_plot.png` exists after a run and shows cluster quality validation scores across cluster counts
  5. Unit tests in `test_clustering.py` cover the contact-zone RMSD computation, clustering, and silhouette score calculation
**Plans**: TBD

### Phase 7: Output & Integration
**Goal**: The pipeline writes all required output files, reports ΔG to the user, and the MDM2/p53 integration test passes the corrected ΔG threshold
**Depends on**: Phase 6
**Requirements**: OUT-01, OUT-02, OUT-03, TEST-02
**Success Criteria** (what must be TRUE):
  1. `ranked_poses.csv` contains the top-10 poses with columns: hybrid score, Vina score, AD4 score, entropy correction, cluster ID, pose filename
  2. `best_pose.pdb` is written as the centroid of the top-ranked cluster — not the top individual scorer
  3. ΔG estimate in kcal/mol is printed to stdout at run completion and appears in `ranked_poses.csv`
  4. The MDM2/p53 integration test (`pytest -m slow`, PDB 2OY2, peptide `ETFSDLWKLLPE`) passes with corrected ΔG < −3 kcal/mol
**Plans**: TBD

### Phase 8: Benchmark & Documentation
**Goal**: The pipeline meets accuracy targets on the 10-complex benchmark suite, install and usage documentation is complete, the license is clean, and the tutorial notebook runs end-to-end
**Depends on**: Phase 7
**Requirements**: TEST-03, DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. `hybridock-pep benchmark` on the 10-complex test set achieves Pearson r ≥ 0.55 on the held-out set and ≥ 0.10 improvement over Vina-alone
  2. `README.md` and `INSTALL.md` provide a one-command install path (`conda env create` + `pip install -e .`) with no undocumented manual steps; ADFRsuite download link is present
  3. `docs/architecture.md` documents the module map, data flow, and subprocess orchestration pattern
  4. `pip-licenses` output is committed and confirms no GPL/LGPL/AGPL dependency in either conda environment
  5. `docs/tutorial.ipynb` runs top-to-bottom without errors on a fresh install, demonstrating the full MDM2/p53 docking walkthrough
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-04-20 |
| 2. Preparation Pipeline | 1/4 | In progress | - |
| 3. Scoring Core | 0/? | Not started | - |
| 4. Sampling Integration | 0/? | Not started | - |
| 5. CLI & Driver | 0/? | Not started | - |
| 6. Analysis & Plots | 0/? | Not started | - |
| 7. Output & Integration | 0/? | Not started | - |
| 8. Benchmark & Documentation | 0/? | Not started | - |
