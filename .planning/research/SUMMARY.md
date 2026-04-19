# Research Summary: HybriDock-Pep

## TL;DR

- **The two-environment split is non-negotiable.** `rapidock-env` (Python 3.9, PyTorch 2.7, CUDA 12.8) and `score-env` (Python 3.11, Vina 1.2.7, OpenMM 8.4) cannot be merged. Every architectural decision flows from this constraint.
- **Upgrade the CUDA stack on day one.** The published RAPiDock requirements (CUDA 11.5, PyTorch 1.11) will not run on the RTX 5070 (Blackwell, CC 12.0). Target PyTorch 2.7 + CUDA 12.8 — first release with native sm_120. PyTorch 2.3 + CUDA 12.4 runs in emulation only and should not be used.
- **Four things will silently corrupt results with no visible error:** PULCHRA v3.07 (pin 3.04 exactly), grid-box-clipping poses dropped without warning (add pre-Vina boundary check), terminal-residue RMSD dominating clustering (cluster over contact-zone Cα only), and entropy α calibrated on wrong distribution.
- **The unique scientific value is the hybrid pipeline itself.** The RSC Chemical Communications 2026 review explicitly identifies lack of physics-calibrated ΔG as the main limitation of generative docking tools. No public tool bridges this gap.
- **MM-GBSA and selectivity workflow are Phase 5+ features.** The core pipeline (Phases 1–4) can ship without them and will already beat Vina-alone.

---

## Stack Decisions

### rapidock-env (Python 3.9, GPU inference)

| Package | Pin | Critical Note |
|---------|-----|---------------|
| Python | 3.9.x | Hard requirement from RAPiDock typing syntax |
| PyTorch | 2.7.0 | First native Blackwell sm_120; do NOT use 2.3 |
| pytorch-cuda | 12.8 | Required for CC 12.0 — 12.4 is emulation only |
| torch-geometric | 2.6.x | cu128 wheels community-maintained; build from source if 404 |
| e3nn | 0.5.1 | Keep pinned; 0.5.1 tested against PT 2.x |
| fair-esm | 2.0.0 | **Highest fragility point.** Package is abandoned upstream. Test import before any other work. May require monkey-patching. |
| biopython | 1.84 | Pin — do not upgrade in this env |
| rdkit | 2024.03.x | Use `rdkit` from conda-forge, NOT the deprecated `rdkit-pypi` |
| PULCHRA | **3.04 exactly** | Build from source. Bioconda ships 3.06 (aromatic side-chain bug). Smoke-test version on every install. |

### score-env (Python 3.11, all in-repo code)

| Package | Pin | Critical Note |
|---------|-----|---------------|
| vina | 1.2.7 | Use Python API (not subprocess) for per-pose scoring — avoids fork+exec × 100 |
| meeko | 0.7.1 | Ligand/peptide prep only; use ADFRsuite `prepare_receptor4.py` for receptor |
| openmm | 8.4.0 | GBn2 implicit solvent for MM-GBSA + pre-scoring minimization |
| openmmforcefields | 0.15.1 | Ships AmberTools 24 params; required for AMBER ff14SB |
| scikit-learn | 1.5.x | Use `average` linkage with precomputed metric — Ward requires Euclidean and RMSD is not Euclidean |
| numpy | 2.1.x | Avoid 2.0.x (breaking dtype changes) |
| pdbfixer | 1.9+ | Receptor PDB cleaning before ADFRsuite prep |
| ADFRsuite | manual download | Non-redistributable. Never commit binaries. Link in INSTALL.md only. |

**Key deviation from spec:** Spec says "PyTorch 2.3+ / CUDA 12.4." Research confirms native sm_120 support landed in PyTorch 2.7 / CUDA 12.8. Update `envs/rapidock-env.yml` on day one.

---

## Feature Priorities

### Must have (table stakes)

- Receptor PDB + peptide sequence string inputs with pre-run validation (bad inputs fail in < 1 s)
- 100 stochastic poses from RAPiDock
- Per-pose Vina + AD4 parallel scoring (both scores in `ranked_poses.csv`)
- Backbone entropy correction with calibrated α (loaded from `calibration.json`)
- Agglomerative RMSD clustering → `cluster_summary.csv`
- Best-pose PDB output (cluster centroid, not top-scorer)
- Convergence plot + cluster dendrogram
- `--seed N` deterministic mode
- `run_metadata.json` full provenance
- Receptor PDBQT preparation wrapper (ADFRsuite)
- Per-pose ligand PDBQT preparation (Meeko, batched, stateless)

### Key differentiators

1. **Hybrid ML + physics pipeline** — RAPiDock sampling + Vina/AD4/entropy. No public tool does this combination.
2. **Dual-scoring with charge decomposition** — Vina (no charges) and AD4 (Gasteiger charges) in parallel; score discrepancy flags electrostatics-dominated binding.
3. **Calibrated entropy correction** — α fit on training complexes; JChem Inf Model 2020 shows this boosts R² from 0.36 to 0.69.
4. **Convergence diagnostics** — Running mean of hybrid score vs. N. Absent from every docking tool in the field.
5. **Reproducibility-first metadata** — Machine-readable provenance. Field gap confirmed in literature.

### Phase 5+ features (do not block critical path)

- `--refine-topk N` MM-GBSA rescoring via OpenMM + GBn2
- `--off-target` dual-receptor selectivity workflow and ΔΔG output
- Benchmark suite (10 complexes), tutorial notebook, iGEM wiki

### Explicit anti-features

- Vina recompile with Coulomb term (rejected in spec §5.6–5.7)
- PyRosetta post-relax step (ref2015 cysteine bug)
- Per-atom charge extraction from Vina scores (no-op — Vina ignores `q`)
- Multi-GPU RAPiDock parallelism
- Bundling ADFRsuite/AutoDock4 binaries (license violation)

---

## Architecture Guidance

### Component boundaries

```
cli.py          Argparse + validation only → DockConfig dataclass → driver.py
driver.py       Single orchestrator; only module that spawns subprocesses
prep/           receptor.py (once), ligand.py (per-pose, stateless), grids.py (once)
sampling/       rapidock_runner.py (runs IN rapidock-env, never imported by score-env)
                pose_io.py (in score-env, parses PDB files → PoseRecord list)
scoring/        vina.py, ad4.py, entropy.py, mmgbsa.py — all pure functions
analysis/       clustering.py (contact-zone Cα RMSD), statistics.py, plotting.py
output/         csv_writer.py, metadata.py — pure serialization
```

### Critical interface decisions

- Four core dataclasses (`DockConfig`, `PoseRecord`, `ScoredPose`, `PoseFailure`) are the interfaces every module plugs into. Define in Phase 1.
- Stage 1/Stage 2 boundary is **file-based** (100 PDB files in `poses/`). Only cross-environment IPC.
- Everything within Stage 2 is in-memory after `pose_io.py` parses PDB files.
- Scoring workers must be **module-level functions** — ProcessPoolExecutor uses pickle; closures do not.
- Pass `alpha` as a plain float into workers — do not read `calibration.json` inside each worker.
- Use `concurrent.futures.ProcessPoolExecutor` for per-pose scoring fan-out.
- Use `conda run --no-capture-output -n rapidock-env` for Stage 1 — `--no-capture-output` required to see GPU OOM errors in real time.
- `--skip-sampling` flag: if `poses/pose_*.pdb` already exist, skip Stage 1. Critical developer-iteration affordance.

### Build order

```
Layer 0:  prep/ modules — establish file format contracts first
Layer 1:  scoring/entropy.py (pure, no binaries), scoring/vina.py, scoring/ad4.py
Layer 2:  sampling/pose_io.py
Layer 3:  scoring/mmgbsa.py (build last — most complex, optional)
Layer 4:  analysis/clustering.py, statistics.py, plotting.py
Layer 5:  output/csv_writer.py, output/metadata.py
Layer 6:  driver.py + cli.py
```

---

## Top Pitfalls to Avoid

Ranked by potential for silent wrong results:

**1. PULCHRA v3.07 aromatic side-chain bug — SILENT wrong results**
Silently produces incomplete Phe/Tyr/Trp/His atoms. LISDAELEAIFEADC contains Phe — triggers on every run.
Fix: Build 3.04 from source. Add `pulchra --version` to smoke_test.sh. Abort if not exactly 3.04.

**2. Grid-box-clipping poses silently dropped — BIASED ensemble statistics**
Vina 1.2.5+ raises fatal error for poses outside grid box. Without pre-scoring validation, poses lost silently.
Fix: Validate atom coordinates against grid boundaries before Vina. Log every skipped pose to `run_metadata.json`.

**3. Terminal-residue RMSD dominating clustering — WRONG cluster centroids**
Full-peptide Cα RMSD groups by terminal position, not binding mode.
Fix: Compute RMSD over 5–8 contact-zone residues only. Validate with silhouette scores.

**4. autogrid4 missing HD map — SILENT AD4 failure**
If `receptor.HD.map` absent, `vina --scoring ad4` aborts. May silently fall back to Vina-only.
Fix: After every autogrid4 run, verify `receptor.HD.map` exists before launching AD4 scoring.

**5. Entropy α on wrong distribution — PLAUSIBLE-LOOKING WRONG ΔG**
α calibrated at wrong temperature or with train/test leakage produces wrong ΔG with no error signal.
Fix: Calibrate at T=310 K. Enforce < 30% receptor sequence identity between training and test sets.

**6. MM-GBSA without pre-minimization — UP TO 7 kJ/mol BIAS**
GBn2 energies of vacuo-generated poses differ significantly from properly solvated poses.
Fix: Mandatory short minimization in GBn2 before single-point energy evaluation.

---

## Roadmap Implications

### Recommended 6-phase structure

| Phase | Name | Goal | Key Deliverable |
|-------|------|------|-----------------|
| 1 | Foundation | Both envs set up, contracts defined | `prep/` wrappers, dataclasses, smoke tests |
| 2 | Scoring Core | Physics scoring validated without GPU | `scoring/vina.py`, `ad4.py`, `entropy.py`, calibration script |
| 3 | Sampling Integration | End-to-end on MDM2/p53 | `driver.py`, RAPiDock subprocess, wall-clock benchmark |
| 4 | Analysis & Output | Full v1 pipeline ships | Clustering, plots, CSV, PDB, CLI, integration tests |
| 5 | Optional Features | Publication-quality ΔG | MM-GBSA, selectivity workflow, benchmark suite |
| 6 | iGEM Polish | Award-ready | Tutorial notebook, architecture docs, license audit |

### Research flags
- **Phase 3:** Validate `fair-esm 2.0.0` import against PyTorch 2.7 on day one. Confirm PyG cu128 wheel availability.
- **Phase 5:** Validate MM-GBSA GBn2 accuracy on at least one complex with known ΔG before integrating into benchmark.

---

## Open Questions

1. Does `fair-esm 2.0.0` import cleanly against PyTorch 2.7? Answer on day one of Phase 3. (LOW confidence)
2. Are PyG cu128 prebuilt wheels available for PyTorch 2.7.0? Have source build fallback ready. (MEDIUM)
3. What is the exact contact-zone residue selection criterion for RMSD clustering? Decide in Phase 4.
4. What is α for LISDAELEAIFEADC on PfLDH? Unknown until Phase 2 calibration. Expected: 0.2–1.2 kcal/mol/residue.
5. Will RAPiDock inference stay under 5 minutes for 100 poses on the RTX 5070 with PyTorch 2.7? Benchmark in Phase 3.
6. Does the `--off-target` workflow require separate grid prep for hLDH (1I0Z)? Yes — confirm binding groove coordinates before Phase 5.

---

*Synthesized from STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md — 2026-04-19*
