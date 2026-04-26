# Roadmap: HybriDock-Pep — Milestone v1.0

## Overview

Build the complete hybrid peptide docking pipeline from the ground up: two conda environments, a physics-based scoring stack, a RAPiDock sampling integration, a full CLI, RMSD clustering with convergence diagnostics, and a benchmarked, documented release ready for iGEM 2026. Phases follow the natural build order — environment and contracts first, scoring before sampling, sampling before orchestration, analysis after data exists, documentation after the pipeline works.

## Phases

- [x] **Phase 1: Foundation** - Both conda environments set up, core dataclasses defined, package scaffold in place, smoke test passing
- [x] **Phase 2: Preparation Pipeline** - Receptor and ligand PDBQT preparation wrappers working, autogrid4 AD4 map generation validated *(complete 2026-04-20)*
- [x] **Phase 3: Scoring Core** - Vina and AD4 per-pose scoring implemented, entropy correction with α calibration working *(complete 2026-04-21)*
- [x] **Phase 4: Sampling Integration** - RAPiDock subprocess wrapper running 100 poses, pose I/O parsing, provenance metadata written *(complete 2026-04-23)*
- [x] **Phase 5: CLI & Driver** - Single entry point with all subcommands, pre-run validation, seed propagation, full two-stage orchestration *(complete 2026-04-25)*
- [x] **Phase 6: Analysis & Plots** - Contact-zone Cα RMSD clustering, ensemble statistics, convergence and silhouette plots *(complete 2026-04-25)*
- [x] **Phase 7: Output & Integration** - Ranked CSV, best-pose PDB, ΔG reporting, MDM2/p53 integration test passing *(complete 2026-04-25)*
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
  - [x] 02-02-PLAN.md — prep/ligand.py Meeko batch converter (PREP-02)
  - [x] 02-03-PLAN.md — prep/grids.py GPF builder + autogrid4 + HD map guard (PREP-03)
  - [x] 02-04-PLAN.md — tests/test_prep.py covering all three modules (PREP-01, PREP-02, PREP-03)

### Phase 3: Scoring Core
**Goal**: Every pose can be independently scored by Vina and AD4 in parallel, and the backbone entropy correction produces a calibrated hybrid score with α validated against the allowed range
**Depends on**: Phase 2
**Requirements**: SCORE-01, SCORE-02, SCORE-03
**Success Criteria** (what must be TRUE):
  1. Given a pose PDBQT and receptor PDBQT, `vina --score_only` via the Vina Python API returns a score; any pose with atoms outside grid boundaries is logged to `run_metadata.json` rather than silently dropped
  2. Given a pose PDBQT and AD4 affinity maps, `vina --scoring ad4` returns an AD4 score in parallel with the Vina score; any positive AD4 score is flagged as an anomaly in output
  3. Given a `calibration.json` with α in range 0.2–1.2 kcal/mol/residue, `entropy.py` computes the backbone entropy correction; α outside this range causes an immediate abort with a diagnostic message
  4. `scripts/calibrate_alpha.py` runs on the training set and writes a valid `calibration.json`; unit tests in `test_scoring.py` cover all three scoring modules
**Plans**: 4 plans
  - [x] 03-01-PLAN.md — scoring/vina.py: Vina Python API batch scorer + grid boundary check (SCORE-01)
  - [x] 03-02-PLAN.md — scoring/ad4.py: AD4 Vina(sf_name='ad4') + load_maps + anomaly flag (SCORE-02)
  - [x] 03-03-PLAN.md — scoring/entropy.py + data/calibration.json: hybrid formula, α/β validation, fit_calibration() (SCORE-03)
  - [x] 03-04-PLAN.md — scripts/calibrate_alpha.py + complete tests/test_scoring.py + data/training_complexes.csv (SCORE-01, SCORE-02, SCORE-03)

### Phase 4: Sampling Integration
**Goal**: RAPiDock runs 100 stochastic inference passes from score-env via a subprocess wrapper, all poses are parsed into PoseRecord objects, and every run writes complete provenance metadata
**Depends on**: Phase 3
**Requirements**: SAMP-01, SAMP-02
**Success Criteria** (what must be TRUE):
  1. Running the sampling stage via `conda run --no-capture-output -n rapidock-env` produces exactly 100 `poses/pose_*.pdb` files; GPU OOM errors surface in real time rather than being swallowed
  2. Passing `--seed N` to the sampling wrapper propagates the seed into RAPiDock and is recorded in `run_metadata.json`
  3. After any sampling run, `run_metadata.json` contains: git SHA, RAPiDock commit SHA, all CLI args, random seed, Vina version, OpenMM version, CUDA version, receptor SHA256, peptide sequence hash, and timestamp
  4. `pose_io.py` parses all 100 PDB files into a `list[PoseRecord]` without error on RAPiDock-format output
**Plans**: 4 plans
  - [x] 04-01-PLAN.md — Test scaffolds: tests/test_sampling.py + tests/test_output.py (SAMP-01, SAMP-02)
  - [x] 04-02-PLAN.md — sampling/rapidock_runner.py + sampling/run_rapidock.py (SAMP-01)
  - [x] 04-03-PLAN.md — sampling/pose_io.py Biopython batch PDB parser (SAMP-01)
  - [x] 04-04-PLAN.md — output/metadata.py two-write provenance JSON (SAMP-02)

### Phase 5: CLI & Driver
**Goal**: A single `hybridock-pep` entry point exposes all four subcommands, validates all inputs before any subprocess is spawned, and the driver orchestrates both pipeline stages end-to-end
**Depends on**: Phase 4
**Requirements**: CLI-01, CLI-02, CLI-03
**Success Criteria** (what must be TRUE):
  1. `hybridock-pep dock`, `hybridock-pep calibrate`, `hybridock-pep benchmark`, and `hybridock-pep prep` all exist with help strings on every flag (with units)
  2. Providing an invalid peptide sequence, a missing receptor PDB, or out-of-range site coordinates causes an error with a clear message in under 1 second — before any subprocess is spawned
  3. `hybridock-pep dock --seed 42` produces the same output on two successive runs (modulo CUDA nondeterminism); the metadata JSON notes the seed
  4. `driver.py` orchestrates Stage 1 (rapidock-env subprocess) and Stage 2 (score-env in-process) in sequence and passes a `DockConfig` through both; if `poses/pose_*.pdb` already exist and `--skip-sampling` is not yet a v1 flag, the driver still completes without error
**Plans**: 3 plans
  - [x] 05-01-PLAN.md — Wave 0 TDD RED gate: test_cli.py + test_driver.py scaffolds (CLI-01, CLI-02, CLI-03)
  - [x] 05-02-PLAN.md — cli.py expansion: all 4 subcommands with real arg defs + dispatch (CLI-01, CLI-02, CLI-03)
  - [x] 05-03-PLAN.md — driver.py: run_dock() pipeline orchestrator, Stage 1+2 wiring (CLI-01, CLI-02, CLI-03)

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
**Plans**: 5 plans
  - [x] 06-01-PLAN.md — Wave 0 TDD RED gate: test_clustering.py scaffold (ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05)
  - [x] 06-02-PLAN.md — analysis/clustering.py: ClusterResult, RMSD matrix, silhouette k-search, cluster_poses() (ANAL-01)
  - [x] 06-03-PLAN.md — analysis/statistics.py: compute_cluster_stats(), 95% CI, write_cluster_summary_csv() (ANAL-02)
  - [x] 06-04-PLAN.md — analysis/plotting.py: plot_convergence(), plot_silhouette(), Agg backend (ANAL-03, OUT-04, OUT-05)
  - [x] 06-05-PLAN.md — Integration: analysis/__init__.py export, driver.py Stage 3 stub replacement, full suite green (ANAL-01..03, OUT-04, OUT-05)

### Phase 7: Output & Integration
**Goal**: The pipeline writes all required output files, reports ΔG to the user, and the MDM2/p53 integration test passes the corrected ΔG threshold
**Depends on**: Phase 6
**Requirements**: OUT-01, OUT-02, OUT-03, TEST-02
**Success Criteria** (what must be TRUE):
  1. `ranked_poses.csv` contains the top-10 poses with columns: hybrid score, Vina score, AD4 score, entropy correction, cluster ID, pose filename
  2. `best_pose.pdb` is written as the centroid of the top-ranked cluster — not the top individual scorer
  3. ΔG estimate in kcal/mol is printed to stdout at run completion and appears in `ranked_poses.csv`
  4. The MDM2/p53 integration test (`pytest -m slow`, PDB 2OY2, peptide `ETFSDLWKLLPE`) passes with corrected ΔG < −3 kcal/mol
**Plans**: 3 plans
  - [x] 07-01-PLAN.md — csv_writer.py + output/__init__.py + pyproject.toml pytest marker + test_csv_writer.py (OUT-01, OUT-02, OUT-03)
  - [x] 07-02-PLAN.md — driver.py Stage 4 + return type + cli.py/test_driver.py caller updates (OUT-01, OUT-02, OUT-03)
  - [x] 07-03-PLAN.md — MDM2/p53 fixtures (25 PDBs + calibration.json) + test_e2e.py integration test (TEST-02)

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
| 2. Preparation Pipeline | 4/4 | Complete | 2026-04-20 |
| 3. Scoring Core | 4/4 | Complete | 2026-04-21 |
| 4. Sampling Integration | 4/4 | Complete | 2026-04-23 |
| 5. CLI & Driver | 3/3 | Complete | 2026-04-25 |
| 6. Analysis & Plots | 5/5 | Complete | 2026-04-25 |
| 7. Output & Integration | 3/3 | Complete | 2026-04-25 |
| 8. Benchmark & Documentation | 0/TBD | Not started | - |
