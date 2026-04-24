---
phase: "05"
plan: "02"
subsystem: cli
tags: [cli, argparse, dispatch, validation]
dependency_graph:
  requires: [05-01]
  provides: [cli-args, dock-dispatch, prep-dispatch, calibrate-dispatch]
  affects: [driver.py (Wave 2), tests/test_cli.py]
tech_stack:
  added: []
  patterns: [argparse subparsers, deferred import, DockConfig validation gate, dispatch table]
key_files:
  created: []
  modified:
    - src/hybridock_pep/cli.py
decisions:
  - "--n-samples default=None (not 100) to enable mutual-exclusion check with --input-poses; 100 applied in _run_dock after guard"
  - "driver import deferred to inside _run_dock after validation block — driver.py is Wave 2; import at function top caused ImportError in validation tests"
  - "DockConfig used as single validation gate in _run_dock and _run_prep; parser.error() called on ValidationError so exit code is always 2"
metrics:
  duration: "3 minutes"
  completed: "2026-04-24T18:23:54Z"
  tasks_completed: 2
  files_modified: 1
---

# Phase 05 Plan 02: Expand CLI — Real Arg Definitions and Dispatch Functions

Replaced four bare `sub.add_parser()` stubs with fully-specified subparsers and
wired all four dispatch functions; all 10 CLI tests now pass.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Expand `_build_parser()` with real arg definitions | 5e10eb3 | src/hybridock_pep/cli.py |
| 2 | Add dispatch functions and update `main()` | 5e10eb3 | src/hybridock_pep/cli.py |

## Test Results

- `tests/test_cli.py`: 10/10 passed
- Full suite: 130 passed, 6 pre-existing RED-gate failures in `test_driver.py` (driver.py is Wave 2 scope), 1 skipped

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking Import] Deferred `driver` import to after validation block**

- **Found during:** Task 2 verification — `test_invalid_peptide_exits_2` failed with `ImportError: cannot import name 'driver'`
- **Issue:** `from hybridock_pep import driver` at the top of `_run_dock` executed before the DockConfig validation gate. `driver.py` does not exist yet (Wave 2). Tests that hit the validation path (invalid peptide, missing receptor) crashed with ImportError instead of exiting with code 2.
- **Fix:** Moved `from hybridock_pep import driver` to immediately before the `driver.run_dock(...)` call, after all validation. Validation tests never reach the import; only a successful `dock` run (which requires driver.py) reaches it.
- **Files modified:** src/hybridock_pep/cli.py
- **Commit:** 5e10eb3

## Known Stubs

- `_run_benchmark`: raises `NotImplementedError("benchmark: Phase 8 scope")` — intentional; benchmark infrastructure is out of scope until Phase 8.
- `_run_dock` driver call: deferred import of `driver.run_dock` will fail at runtime until driver.py is created in Wave 2 (05-03). All CLI validation is exercisable without driver.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

- src/hybridock_pep/cli.py: exists
- Commit 5e10eb3: confirmed in git log
- 10 CLI tests: all passing
- 130 pre-existing tests: no regressions
