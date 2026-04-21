---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03-04-PLAN.md — calibrate_alpha.py thin wrapper, training_complexes.csv D-08 schema, 30 scoring tests passing, 96% coverage
last_updated: "2026-04-21T13:09:04.192Z"
last_activity: 2026-04-21
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 10
  completed_plans: 10
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-19)

**Core value:** Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.
**Current focus:** Phase 3 — Scoring Core

## Current Position

Phase: 3 of 8 (Scoring Core — IN PROGRESS)
Plan: 4 of 4 in phase 3 (03-01 complete)
Status: Ready to execute
Last activity: 2026-04-21

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 6
- Average duration: 4.0 min
- Total execution time: 0.40 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 7 min | 3.5 min |
| 02-preparation | 4 | 19 min | 4.8 min |
| 03-scoring-core | 1 | 2 min | 2.0 min |

**Recent Trend:** On track

*Updated after each plan completion*
| Phase 03-scoring-core P02 | 1158 | 2 tasks | 3 files |
| Phase 03-scoring-core P03 | 5 | 2 tasks | 4 files |
| Phase 03-scoring-core P04 | 191 | 2 tasks | 6 files |

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
- n_residues derived from len(peptide_sequence) in training CSV, not scores JSON — CSV is authoritative source of sequence info
- Post-write self-check: load_calibration() called after write_calibration() to validate alpha/beta convergence before caller proceeds
- black --target-version py311 required in base Python 3.13 env — without it, AST safety check fails on py314-targeted output

### Pending Todos

None yet.

### Blockers/Concerns

- fair-esm 2.0.0 import against PyTorch 2.7 is unverified — validate on day one of Phase 4
- PyG cu128 prebuilt wheels for PyTorch 2.7.0 may not exist — have source build fallback ready
- PULCHRA must be built from source at exactly v3.04 — Bioconda ships 3.06 (aromatic side-chain bug)
- pytest --cov flag fails in Python 3.13 base env (numpy double-import conflict with pytest-cov hooks) — use score-env (Python 3.11) or coverage run for coverage measurement

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | MM-GBSA --refine-topk (OPT-01) | v2 scope | Roadmap creation |
| v2 | --skip-sampling reuse flag (OPT-02) | v2 scope | Roadmap creation |
| v2 | Cluster dendrogram plot (VIZ-01) | v2 scope | Roadmap creation |

## Session Continuity

Last session: 2026-04-21T13:09:04.181Z
Stopped at: Completed 03-04-PLAN.md — calibrate_alpha.py thin wrapper, training_complexes.csv D-08 schema, 30 scoring tests passing, 96% coverage
Resume file: None
