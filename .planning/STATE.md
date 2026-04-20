# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-19)

**Core value:** Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.
**Current focus:** Phase 2 — Preparation Pipeline

## Current Position

Phase: 2 of 8 (Preparation Pipeline)
Plan: 1 of 4 in current phase
Status: In progress — 02-01 complete
Last activity: 2026-04-20 — 02-01 complete; PrepError, receptor.py, fixtures

Progress: [███░░░░░░░] 19% (3 of 16 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 3.5 min
- Total execution time: 0.12 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 7 min | 3.5 min |
| 02-preparation | 1 | 3 min | 3 min |

**Recent Trend:** On track

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

- fair-esm 2.0.0 import against PyTorch 2.7 is unverified — validate on day one of Phase 4
- PyG cu128 prebuilt wheels for PyTorch 2.7.0 may not exist — have source build fallback ready
- PULCHRA must be built from source at exactly v3.04 — Bioconda ships 3.06 (aromatic side-chain bug)

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | MM-GBSA --refine-topk (OPT-01) | v2 scope | Roadmap creation |
| v2 | --skip-sampling reuse flag (OPT-02) | v2 scope | Roadmap creation |
| v2 | Cluster dendrogram plot (VIZ-01) | v2 scope | Roadmap creation |

## Session Continuity

Last session: 2026-04-20
Stopped at: Completed 02-01-PLAN.md — PrepError, receptor.py, test fixtures
Resume file: None
