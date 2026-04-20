---
phase: 02-preparation
plan: "02"
subsystem: prep
tags: [prep, ligand, meeko, pdbqt, ProcessPoolExecutor, parallelism, PoseFailure, tdd]

dependency_graph:
  requires:
    - phase: 02-01
      provides: PoseFailure dataclass, PrepError, test fixtures (pose_tiny.pdb)
  provides:
    - hybridock_pep.prep.ligand.prepare_ligand_batch
    - hybridock_pep.prep.ligand._prepare_single_ligand
  affects:
    - prep/grids.py (next in phase 02)
    - driver.py (calls prepare_ligand_batch per-pose batch)
    - tests/test_prep.py (ligand test classes added)

tech-stack:
  added:
    - meeko>=0.7.1 (MoleculePreparation.from_pdb, write_pdbqt_string — already in score-env)
    - concurrent.futures.ProcessPoolExecutor (CPU-bound parallel PDBQT conversion)
  patterns:
    - Module-level worker function for ProcessPoolExecutor pickling compatibility
    - Local import inside try block to catch dependency import errors as PoseFailure
    - Collect-all-failures pattern: batch never raises on per-pose errors
    - TDD red-green with pytest

key-files:
  created:
    - src/hybridock_pep/prep/ligand.py
  modified:
    - tests/test_prep.py

key-decisions:
  - "Meeko import moved inside try block in worker — catches rdkit/import errors as PoseFailure instead of propagating"
  - "ProcessPoolExecutor chosen over ThreadPoolExecutor — PDBQT conversion is CPU-bound"
  - "_prepare_single_ligand at module level (not closure) — required for ProcessPoolExecutor pickling on macOS spawn"

patterns-established:
  - "Batch function collect-all-failures: len(successes) + len(failures) == len(inputs) always"
  - "Worker catches all exceptions including import errors inside try block"

requirements-completed:
  - PREP-02

duration: ~2min
completed: "2026-04-20"
---

# Phase 2 Plan 2: Ligand Batch PDBQT Converter Summary

**ProcessPoolExecutor-parallelized Meeko batch converter producing Gasteiger-charged PDBQT files from pose PDBs, with all per-pose failures collected as PoseFailure records (no exception propagation)**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-20T20:27:44Z
- **Completed:** 2026-04-20T20:29:44Z
- **Tasks:** 1 (TDD: RED commit + GREEN commit)
- **Files modified:** 2

## Accomplishments

- `prepare_ligand_batch(pdb_paths, output_dir)` returns `tuple[list[Path], list[PoseFailure]]` — batch never raises on per-pose failures
- `_prepare_single_ligand` module-level worker (picklable for ProcessPoolExecutor on macOS spawn start method)
- Meeko `MoleculePreparation.from_pdb` + `write_pdbqt_string` — Gasteiger charges auto-assigned, required by AD4 scoring (§2.1)
- 16 new passing tests in `tests/test_prep.py` covering structural requirements, behavioral invariants, and worker unit behavior
- Full suite: 42 tests pass (0 failures)

## Task Commits

Each task was committed atomically:

1. **RED — Failing tests for ligand batch PDBQT prep** - `738d2cb` (test)
2. **GREEN — prep/ligand.py implementation** - `981f517` (feat)

_TDD task: test commit (RED) followed by implementation commit (GREEN)_

## Files Created/Modified

- `src/hybridock_pep/prep/ligand.py` — `prepare_ligand_batch()` + `_prepare_single_ligand` worker
- `tests/test_prep.py` — 3 new test classes: `TestLigandBatchImports`, `TestLigandBatchBehavior`, `TestPrepareSingleLigandWorker`

## Decisions Made

- Meeko import placed inside the `try` block in `_prepare_single_ligand` to catch `ModuleNotFoundError` (missing rdkit) as a `PoseFailure` rather than propagating from the worker process
- `ProcessPoolExecutor` used (not `ThreadPoolExecutor`) — PDBQT conversion is CPU-bound (Meeko charge assignment and atom typing)
- `max_workers=None` defaults to `os.cpu_count()` — caller can override for constrained environments

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Moved meeko import inside try block to catch import errors as PoseFailure**
- **Found during:** Task 1 GREEN phase (running tests)
- **Issue:** Plan code template placed `from meeko import MoleculePreparation` before the `try` block. In the test environment (and any environment missing rdkit), this causes `ModuleNotFoundError` to propagate out of the worker process, breaking the collect-all-failures guarantee.
- **Fix:** Moved `from meeko import MoleculePreparation` to the first line inside the `try` block so import failures are caught as `PoseFailure(stage="prep", error_msg="ModuleNotFoundError: ...")`
- **Files modified:** `src/hybridock_pep/prep/ligand.py`
- **Verification:** All 16 ligand tests pass; `test_no_exception_propagates_from_batch` confirms no propagation even when meeko/rdkit unavailable
- **Committed in:** `981f517` (GREEN implementation commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in plan code template)
**Impact on plan:** Essential for correctness — the collect-all-failures contract would be silently violated in any environment missing rdkit. No scope creep.

## Issues Encountered

None beyond the Rule 1 fix above.

## Known Stubs

None — `prepare_ligand_batch` is fully functional. Behavior in environments with Meeko+rdkit available: writes real PDBQT files. Behavior without: collects `PoseFailure` records. Both paths verified by tests.

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns beyond the plan's threat model:
- T-02-05 (pose PDB tampering): accepted — local RAPiDock output only
- T-02-06 (ProcessPoolExecutor DoS): accepted — max_workers bounded by os.cpu_count()
- T-02-07 (PoseFailure error_msg disclosure): accepted — local user only

## Next Phase Readiness

- PREP-02 complete. `prepare_ligand_batch` ready for use in `driver.py` Stage 2.
- Phase 02-03 (`prep/grids.py`) can proceed: depends on `PrepError` (PREP-01, done) and `DockConfig` (Phase 01, done).
- No blockers.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| src/hybridock_pep/prep/ligand.py | FOUND |
| tests/test_prep.py (ligand tests) | FOUND |
| 738d2cb (test RED commit) | FOUND |
| 981f517 (feat GREEN commit) | FOUND |
| 42 tests pass | VERIFIED |

---
*Phase: 02-preparation*
*Completed: 2026-04-20*
