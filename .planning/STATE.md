---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 8 Plan 03 complete — README.md (9-section user guide) + INSTALL.md three additions written
last_updated: "2026-04-26T17:48:33.000Z"
last_activity: 2026-04-26
progress:
  total_phases: 8
  completed_phases: 7
  total_plans: 31
  completed_plans: 28
  percent: 96
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-19)

**Core value:** Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.
**Current focus:** Phase 08 — benchmark & documentation

## Current Position

Phase: 08 (benchmark-documentation) — IN PROGRESS
Plan: 3 of 6 complete
Status: Plan 08-03 complete — README.md (9-section user guide, D-11) + INSTALL.md additions (D-12) delivered
Last activity: 2026-04-26

Progress: [█████████░] 96%

## Performance Metrics

**Velocity:**

- Total plans completed: 25
- Average duration: 4.0 min
- Total execution time: ~1.7 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 7 min | 3.5 min |
| 02-preparation | 4 | 19 min | 4.8 min |
| 03-scoring-core | 4 | 8 min | 2.0 min |
| 04-sampling-integration | 4 | 16 min | 4.0 min |
| 05-cli-driver | 3 | 12 min | 4.0 min |
| 06-analysis-plots | 5 | 20 min | 4.0 min |
| 07-output-integration | 3 | 12 min | 4.0 min |
| 08-benchmark-documentation | 3 | 11 min | 3.7 min |

**Recent Trend:** On track

## Accumulated Context

### Decisions

- PyTorch 2.7 + CUDA 12.8 (not 2.3/12.4) — first native sm_120; emulation only on older stack
- Vina Python API for scoring — avoids 100 fork+exec cycles per run
- Contact-zone Ca RMSD for clustering — terminal residues dominate full-peptide RMSD and corrupt cluster quality
- Skip PyRosetta relax by default — ref2015 alignment failure on C-terminal cysteine (§16.1)
- AD4 scoring in parallel with Vina — provides charge signal Vina ignores; discrepancy flags electrostatics-dominated binding
- Two separate conda envs — rapidock-env (Python 3.9) and score-env (Python 3.11); incompatible stacks
- DockConfig frozen=True (immutable) — config crosses subprocess boundary and must not mutate
- run_id auto-generated via @model_validator(mode='before') — must resolve before field validators fire
- ScoredPose extends PoseRecord via @dataclass inheritance — parent fields required, child fields defaulted
- PrepError(RuntimeError) defined in prep/errors.py — raised by prepare_receptor4.py non-zero exit and (future) autogrid4 HD map missing
- _filter_pdb_lines() pre-filters raw PDB text before pdbfixer — drops non-water HETATM and altLoc B/C/... to prevent "Unknown Receptor Type" in autogrid4
- prepare_receptor() always regenerates PDBQT (no caching guard) — pdbfixer 3-step then prepare_receptor4.py subprocess with hard abort on non-zero exit
- Meeko import inside try block in _prepare_single_ligand — catches rdkit/import errors as PoseFailure instead of propagating from ProcessPoolExecutor worker
- prepare_ligand_batch() collect-all-failures: len(successes)+len(failures)==len(inputs) always; batch never raises on per-pose errors
- _build_gpf() generates GPF programmatically from DockConfig — no template file; ligand_types includes HD for receptor.HD.map generation
- generate_ad4_maps() hard-aborts with PrepError (verbatim D-05 message) if receptor.HD.map missing — prevents silent vina --scoring ad4 failure downstream
- All hybridock_pep imports kept lazy in test_prep.py — pytest-cov triggers numpy double-import error in Python 3.13 base env; coverage measured via coverage run
- --n-samples default=None (not 100) in argparse to enable mutual-exclusion check with --input-poses; 100 applied in _run_dock after guard
- driver import deferred inside _run_dock after DockConfig validation block — driver.py is Wave 2; eager import caused ImportError in validation error-path tests
- Vina import lazy in scoring/vina.py (try/except ImportError, Vina=None fallback) — allows check_grid_boundary to run in base env; mock.patch replaces Vina in tests; real Vina loaded in score-env
- compute_vina_maps called once before pose loop (not per-pose) — batch scoring pattern; per-pose recomputation would be wasteful
- _append_clipped_pose helper in vina.py (not metadata.py) — metadata.py does not exist in Phase 3; avoids premature coupling to Phase 4 output module
- float(v.score()[0]) used throughout scoring — prevents raw numpy array type comparisons leaking into downstream arithmetic
- load_maps(str(maps_dir / 'receptor')) not set_receptor(): AD4 C++ binding raises RuntimeError on set_receptor with sf_name='ad4'; map prefix without extension resolves to HD/C/etc map files by Vina internals
- is_ad4_anomaly = ad4_score > 0 (strict positive, not zero): per D-06, zero score is not anomalous; positive score indicates repulsive/unphysical binding; pose still in scored list (informational flag)
- apply_hybrid_score() does NOT validate alpha/beta — validation is load_calibration()'s sole responsibility (separation of concerns, T-03-09)
- RT = 0.592 kcal/mol hardcoded at 298K in fit_calibration(); not a CLI parameter in v1 (D-09)
- scipy installed in base Python test env to unblock TestEntropy; production target is score-env (score-env.yml)
- calibrate_alpha.py aborts with ValueError if --scores-json not provided — live scoring wired in Phase 5; Phase 3 requires pre-computed scores
- RuntimeError in run_dock() only fires when records > 0 and all ligand prep fails — empty bypass run (zero records) is valid, not an error
- n_residues derived from len(peptide_sequence) in training CSV, not scores JSON — CSV is authoritative source of sequence info
- Post-write self-check: load_calibration() called after write_calibration() to validate alpha/beta convergence before caller proceeds
- black --target-version py311 required in base Python 3.13 env — without it, AST safety check fails on py314-targeted output
- 05-01 RED gate: test_cli.py + test_driver.py collected clean; 9 pass (DockConfig seed, existing subcommands/validation), 7 fail as expected (driver.py absent, _build_parser not exported)
- Lazy hybridock_pep imports in all Phase 4 test files — prevents ModuleNotFoundError in base Python env; established Phase 3 pattern extended to sampling and output tests
- SEQRES-first sequence extraction split into two complementary tests (test_parse_seqres_preferred + test_parse_atom_fallback) to independently verify both code paths for D-14
- Env var helpers in rapidock_runner.py return placeholder paths (not raise) when RAPIDOCK_DIR/MODEL_DIR/CKPT unset — testable without RAPiDock installed; WARNING logged for production misconfiguration
- fastrelax=False hardcoded in run_rapidock.py per CLAUDE.md §2.5 — ref2015 alignment failure on C-terminal cysteine in LISDAELEAIFEADC
- test_ci95 split into two test methods (n=2 and n=1) per plan verbatim code — plan acceptance criteria said 10 tests but template had 11; code template is authoritative
- statistics.py and plotting.py created as functional stubs — cluster_poses() calls them at runtime so they must work; plotting uses placeholder files when matplotlib absent
- Chain assignments in test_complexes_meta.csv are conventional defaults (A=receptor, B=peptide); must be verified from ATOM records before first benchmark run on RTX machine
- benchmark.py lives in scripts/ not src/ — imported in tests via _SCRIPTS_DIR path injection (sys.path.insert); not a package import
- VALID_STATUSES constant required in benchmark.py — tested by TestOutputSchema::test_status_values_are_defined; values: ok, skipped_download, skipped_prep, skipped_scoring
- validate_pdb_id(pdb_id) function required in benchmark.py — guards URL construction against injection (T-08-01); regex ^[0-9][A-Z0-9]{3}$
- scipy.stats.t.interval used directly (scale=SEM, df=n-1) in _ci95 — not t.ppf; module-level import with ImportError guard
- write_best_pose_pdb(cluster_result, config) — 2-arg signature; scored_poses not needed, best_pose_idx from ClusterResult.per_cluster_stats
- Stage 4 guard uses if cluster_result is not None: (not len check) — cluster_result sentinel initialized before Stage 3
- patch target for Stage 4 mocks is hybridock_pep.output.csv_writer.* (lazy import inside function body, not module-level on driver)
- conftest.py auto-skips @pytest.mark.slow tests when -m slow not passed — prevents e2e test from failing in environments missing score-env stack

### Pending Todos

None.

### Blockers/Concerns

- fair-esm 2.0.0 import against PyTorch 2.7 is unverified — validate on day one of Phase 4
- PyG cu128 prebuilt wheels for PyTorch 2.7.0 may not exist — have source build fallback ready
- PULCHRA must be built from source at exactly v3.04 — Bioconda ships 3.06 (aromatic side-chain bug)
- pytest --cov flag fails in Python 3.13 base env (numpy double-import conflict with pytest-cov hooks) — use score-env (Python 3.11) or coverage run for coverage measurement
- pdbfixer not installed in base miniconda3 env — 18 tests in test_prep.py and test_driver.py fail with ModuleNotFoundError; these pass in score-env

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | MM-GBSA --refine-topk (OPT-01) | v2 scope | Roadmap creation |
| v2 | --skip-sampling reuse flag (OPT-02) | v2 scope | Roadmap creation |
| v2 | Cluster dendrogram plot (VIZ-01) | v2 scope | Roadmap creation |

## Session Continuity

Last session: 2026-04-25T21:25:00.000Z
Stopped at: Phase 07 complete — all 3 plans shipped (csv_writer + output API, driver Stage 4 + tuple return, MDM2/p53 fixtures + e2e test)
Resume file: None
