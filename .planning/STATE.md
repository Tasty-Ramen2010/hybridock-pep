# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-19)

**Core value:** Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.
**Current focus:** Phase 1 — Foundation

## Current Position

Phase: 1 of 8 (Foundation)
Plan: 2 of 2 in current phase
Status: Phase complete — advancing to Phase 2
Last activity: 2026-04-20 — Plan 01-02 complete (package scaffold + core models + CLI stub + tests)

Progress: [██░░░░░░░░] 13% (2 of 16 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 3.5 min
- Total execution time: 0.12 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 7 min | 3.5 min |

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
Stopped at: Completed 01-02-PLAN.md — package scaffold, core models, CLI stub, tests
Resume file: .planning/phases/02-preparation/ (Phase 2, first plan)
