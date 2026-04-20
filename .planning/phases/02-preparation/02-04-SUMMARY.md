---
phase: 02-preparation
plan: "04"
subsystem: prep
tags: [prep, testing, receptor, ligand, grids, TestReceptorPrep, TestLigandBatch, TestGrids, monkeypatch]

dependency_graph:
  requires:
    - hybridock_pep.prep.receptor.prepare_receptor (02-01)
    - hybridock_pep.prep.ligand.prepare_ligand_batch (02-02)
    - hybridock_pep.prep.grids.generate_ad4_maps, _build_gpf (02-03)
    - hybridock_pep.models.DockConfig, PoseFailure (01-02)
    - hybridock_pep.prep.PrepError (02-01)
    - tests/fixtures/receptor_tiny.pdb, pose_tiny.pdb (01-02)
  provides:
    - tests/test_prep.py with class TestReceptorPrep, TestLigandBatch, TestGrids
    - FIXTURES_DIR module-level constant
    - meeko_available session-scoped skip fixture
  affects:
    - Phase 02 completion — final wave test file reconciliation

tech-stack:
  added: []
  patterns:
    - pytest monkeypatch.setattr targeting module-qualified paths (e.g. hybridock_pep.prep.receptor.subprocess.run)
    - subprocess.CompletedProcess for fake subprocess results (no MagicMock returncode duck-typing)
    - SpyFixer class pattern for verifying call ordering before subprocess
    - meeko_available session fixture for optional-dependency skip guard
    - Lazy imports inside test methods to avoid pytest-cov numpy/Python 3.13 double-import conflict

key-files:
  created: []
  modified:
    - tests/test_prep.py

key-decisions:
  - "All hybridock_pep imports kept lazy (inside test methods) — pytest-cov triggers numpy double-import in Python 3.13 base env; coverage measured via coverage run instead"
  - "Three required classes (TestReceptorPrep, TestLigandBatch, TestGrids) added without deleting pre-existing tests — 51 tests preserved, 11 new tests added"
  - "TestLigandBatch.test_batch_single_pose_success gated by meeko_available fixture — skips in base env, runs in score-env"

requirements-completed:
  - PREP-01
  - PREP-02
  - PREP-03

duration: ~6min
completed: "2026-04-20"
---

# Phase 2 Plan 4: Test Reconciliation — TestReceptorPrep, TestLigandBatch, TestGrids Summary

**Three required test classes (TestReceptorPrep, TestLigandBatch, TestGrids) added to test_prep.py using pytest monkeypatch style, with FIXTURES_DIR constant and meeko_available skip fixture, bringing prep/ coverage to 94% and completing Phase 2.**

## Performance

- **Duration:** ~6 min
- **Completed:** 2026-04-20
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added `FIXTURES_DIR = Path(__file__).parent / "fixtures"` at module level (plan requirement)
- Added `meeko_available` session-scoped skip fixture for optional Meeko dependency
- `TestReceptorPrep` (4 tests): `test_prepare_receptor_calls_prepare_receptor4`, `test_prepare_receptor_nonzero_exit_raises_prep_error`, `test_prepare_receptor_always_regenerates`, `test_pdbfixer_called_before_subprocess` — all using `monkeypatch.setattr("hybridock_pep.prep.receptor.subprocess.run", ...)`
- `TestLigandBatch` (3 tests): `test_batch_single_pose_success` (meeko-gated), `test_batch_missing_pdb_collected_as_failure`, `test_batch_successes_plus_failures_equals_input`
- `TestGrids` (5 tests): `test_build_gpf_contains_hd_type`, `test_build_gpf_npts_from_box_size`, `test_build_gpf_gridcenter_from_site_coords`, `test_generate_ad4_maps_hd_map_missing_raises` (match="receptor.HD.map not found after autogrid4"), `test_generate_ad4_maps_success_returns_maps_dir`
- All 62 tests pass (1 skipped: `test_batch_single_pose_success` — meeko not in base env)
- prep/ coverage: 94% (126 stmts, 7 missed in ligand.py meeko success path)

## Task Commits

1. `ef92fd1` — feat(02-04): add TestReceptorPrep, TestLigandBatch, TestGrids to test_prep.py

## Files Created/Modified

- `tests/test_prep.py` — 3 new test classes, FIXTURES_DIR constant, meeko_available fixture (345 lines inserted)

## Decisions Made

- Kept all hybridock_pep imports lazy (inside test methods, not at module level) to avoid pytest-cov's import hook triggering a numpy double-import error in the Python 3.13 base environment. Coverage measured via `python -m coverage run -m pytest` instead of `--cov` flag.
- Pre-existing 51 tests preserved; 11 new tests added to reach 62 total.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed module-level hybridock_pep imports to fix pytest-cov double-import conflict**
- **Found during:** Task 1 — coverage verification
- **Issue:** The plan's preamble imports (`from hybridock_pep.models import DockConfig, PoseFailure` etc. at module level) caused pytest-cov's import hooks to trigger a numpy double-import error (`ImportError: cannot load module more than once per process`) in the Python 3.13 system environment. This broke all tests when run with `--cov`.
- **Fix:** Moved all `hybridock_pep.*` imports inside test methods and fixtures (lazy import pattern, matching the pre-existing tests in this file). `FIXTURES_DIR` and `subprocess`/`pytest`/`MagicMock` kept at module level as they are stdlib and don't trigger the conflict.
- **Files modified:** `tests/test_prep.py`
- **Commit:** `ef92fd1`
- **Root cause:** Pre-existing environment issue — base conda env uses Python 3.13 with a numpy build that conflicts with pytest-cov's `sys.meta_path` hooks. score-env (Python 3.11) would not have this problem.

## Known Stubs

None. All three test classes exercise real production code paths with mocked external binaries.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes. Test file only — no production surface added.

- T-02-12 (fixture tampering): accepted — fixtures are version-controlled deterministic files
- T-02-13 (ProcessPoolExecutor DoS in TestLigandBatch): mitigated — all ligand batch tests use `max_workers=1`

## Self-Check: PASSED

| Item | Status |
|------|--------|
| tests/test_prep.py (modified) | FOUND |
| ef92fd1 (feat commit) | FOUND |
| class TestReceptorPrep | FOUND |
| class TestLigandBatch | FOUND |
| class TestGrids | FOUND |
| FIXTURES_DIR at module level | FOUND |
| monkeypatch.setattr("hybridock_pep.prep.receptor.subprocess.run" | FOUND |
| monkeypatch.setattr("hybridock_pep.prep.grids.subprocess.run" | FOUND |
| pytest.raises(PrepError, match="receptor.HD.map not found after autogrid4") | FOUND |
| 62 tests pass, 1 skipped | VERIFIED |
| prep/ coverage 94% | VERIFIED (via coverage run) |

---
*Phase: 02-preparation*
*Completed: 2026-04-20*
