# Requirements: HybriDock-Pep

**Defined:** 2026-04-19
**Core Value:** Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.

## v1 Requirements

### CLI

- [ ] **CLI-01**: User can run `dock`, `calibrate`, `benchmark`, `prep` subcommands from a single `hybridock-pep` entry point
- [ ] **CLI-02**: All inputs (peptide sequence, receptor PDB, site coordinates, box size) are validated and errors reported before any subprocess is spawned
- [ ] **CLI-03**: User can pass `--seed N` to make a run fully deterministic (modulo CUDA nondeterminism, flagged in metadata)

### Preparation

- [x] **PREP-01**: User can prepare a receptor PDB as PDBQT (pdbfixer cleaning → ADFRsuite `prepare_receptor4.py` wrapper)
- [x] **PREP-02**: Pipeline prepares all 100 diffusion poses as PDBQT in batch (Meeko, stateless per-pose, parallelized)
- [x] **PREP-03**: Pipeline generates AutoDock4 affinity maps via `autogrid4`; aborts with clear error if `receptor.HD.map` is missing after generation

### Sampling

- [x] **SAMP-01
**: Pipeline runs RAPiDock N=100 stochastic inference passes in `rapidock-env` via `conda run` subprocess; seed propagated for reproducibility
- [x] **SAMP-02
**: Every run writes `run_metadata.json` containing: git SHA, RAPiDock commit SHA, all CLI args, random seed, tool versions (Vina, OpenMM, CUDA), receptor SHA256, peptide sequence hash, timestamp

### Scoring

- [x] **SCORE-01
**: Pipeline scores each pose with `vina --score_only` via the Vina Python API; validates all atom coordinates against grid boundaries before scoring and logs any clipped poses to `run_metadata.json` (never silently drops)
- [x] **SCORE-02
**: Pipeline scores each pose with `vina --scoring ad4` in parallel with Vina; flags any positive AD4 scores as anomalies in output
- [x] **SCORE-03
**: Pipeline applies backbone entropy correction using calibrated α loaded from `calibration.json`; validates α is in range 0.2–1.2 kcal/mol/residue and aborts if outside

### Analysis

- [x] **ANAL-01
**: Pipeline clusters poses by pairwise Cα RMSD computed over contact-zone residues only (not full peptide), using agglomerative clustering with average linkage and precomputed metric; reports silhouette score per run
- [x] **ANAL-02
**: Pipeline computes per-cluster ensemble statistics (mean, std, 95% CI of hybrid score) and writes `cluster_summary.csv`
- [x] **ANAL-03
**: Pipeline generates `convergence_plot.png` showing running mean ± σ of hybrid score vs. number of poses N

### Output

- [ ] **OUT-01**: Pipeline writes `ranked_poses.csv` with top-10 poses including: hybrid score, Vina score, AD4 score, entropy correction, cluster ID, pose filename
- [ ] **OUT-02**: Pipeline writes `best_pose.pdb` — the centroid of the top-ranked cluster, not just the top individual scorer
- [ ] **OUT-03**: ΔG estimate in kcal/mol is reported in `ranked_poses.csv` and printed to stdout at run completion
- [x] **OUT-04
**: Pipeline generates `convergence_plot.png` (running mean ± σ vs N) confirming ensemble convergence
- [x] **OUT-05
**: Pipeline generates `silhouette_plot.png` showing cluster quality validation scores

### Testing

- [x] **TEST-01**: `scripts/smoke_test.sh` checks: RTX 5070 CUDA compute capability, ADFRsuite on PATH, Vina version ≥ 1.2.5; exits non-zero with diagnostic message on any failure
- [ ] **TEST-02**: Integration test on MDM2/p53 (PDB 2OY2, peptide `ETFSDLWKLLPE`) passes when corrected ΔG < −3 kcal/mol; tagged `pytest -m slow`
- [ ] **TEST-03**: Benchmark suite runs on 10 reference complexes and achieves Pearson r ≥ 0.55 on held-out test set and ≥ 0.10 improvement over Vina-alone

### Documentation

- [ ] **DOCS-01**: `README.md` and `INSTALL.md` provide one-command install (`conda env create` + `pip install -e .`) with ADFRsuite download link; no manual steps beyond what is documented
- [ ] **DOCS-02**: `docs/architecture.md` documents module map, data flow between components, and subprocess orchestration pattern
- [ ] **DOCS-03**: License audit confirms no GPL/LGPL/AGPL dependencies in either conda environment (`pip-licenses` output committed)
- [ ] **DOCS-04**: `docs/tutorial.ipynb` demonstrates full MDM2/p53 docking walkthrough from fresh install to final output, runs top-to-bottom without errors

## v2 Requirements

### Optional Refinement

- **OPT-01**: `--refine-topk N` flag runs MM-GBSA via OpenMM (GBn2 implicit solvent + mandatory pre-minimization) on top-K cluster centroids for publication-quality ΔG
- **OPT-02**: `--skip-sampling` flag reuses existing `poses/` directory to iterate on scoring without re-running GPU inference

### Analysis Extras

- **VIZ-01**: Cluster dendrogram plot (`dendrogram.png`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| `--off-target` dual-receptor selectivity workflow | User runs `dock` twice manually and subtracts; two CLI invocations is sufficient |
| GUI or web interface | Dry lab CLI users only |
| Multi-GPU RAPiDock parallelism | One GPU, sequential inference; fork if parallel needed, never thread |
| Recompiling Vina with Coulomb term | Explicitly rejected in spec §5.6–5.7 |
| PyRosetta relax step | ref2015 cysteine alignment failure on LISDAELEAIFEADC (§16.1) |
| Bundling ADFRsuite/AutoDock4 binaries | Non-redistributable licenses; INSTALL.md links to official download |
| PULCHRA backbone reconstruction | RAPiDock outputs full-atom poses; PULCHRA only needed for ADCP-style backbone-only output |
| Per-atom charge extraction from Vina scores | No-op — Vina ignores the `q` column entirely |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLI-01 | Phase 5 | Pending |
| CLI-02 | Phase 5 | Pending |
| CLI-03 | Phase 5 | Pending |
| PREP-01 | Phase 2 | Complete |
| PREP-02 | Phase 2 | Complete |
| PREP-03 | Phase 2 | Complete |
| SAMP-01 | Phase 4 | Pending |
| SAMP-02 | Phase 4 | Pending |
| SCORE-01 | Phase 3 | Pending |
| SCORE-02 | Phase 3 | Pending |
| SCORE-03 | Phase 3 | Pending |
| ANAL-01 | Phase 6 | Pending |
| ANAL-02 | Phase 6 | Pending |
| ANAL-03 | Phase 6 | Pending |
| OUT-01 | Phase 7 | Pending |
| OUT-02 | Phase 7 | Pending |
| OUT-03 | Phase 7 | Pending |
| OUT-04 | Phase 6 | Pending |
| OUT-05 | Phase 6 | Pending |
| TEST-01 | Phase 1 | Pending |
| TEST-02 | Phase 7 | Pending |
| TEST-03 | Phase 8 | Pending |
| DOCS-01 | Phase 8 | Pending |
| DOCS-02 | Phase 8 | Pending |
| DOCS-03 | Phase 8 | Pending |
| DOCS-04 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26 (all mapped)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-19*
*Last updated: 2026-04-19 after roadmap creation (8 phases)*
