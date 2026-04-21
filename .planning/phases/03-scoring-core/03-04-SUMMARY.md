---
phase: 03-scoring-core
plan: "04"
subsystem: scoring
tags: [calibration, csv, cli, coverage-gate, tdd, ruff, black]

# Dependency graph
requires:
  - phase: 03-01
    provides: score_vina_batch(), check_grid_boundary()
  - phase: 03-02
    provides: score_ad4_batch()
  - phase: 03-03
    provides: fit_calibration(), write_calibration(), load_calibration()

provides:
  - scripts/calibrate_alpha.py: thin CLI wrapper (parse_args + main); D-08 CSV + scores-JSON → fit_calibration → write_calibration → load_calibration self-check
  - data/training_complexes.csv: 3-column D-08 schema (pdb_id, peptide_sequence, experimental_pkd); 3 literature rows
  - TestCalibration: 5 integration tests covering CSV schema, script import, JSON output, end-to-end D-11 keys, and write_calibration schema
  - Phase 3 gate: 30 scoring tests passing; 96% line coverage on src/hybridock_pep/scoring/*

affects:
  - 05 (hybridock-pep calibrate subcommand wires to main() in Phase 5)
  - Phase 4 driver imports fit_calibration and apply_hybrid_score from entropy.py

# Tech tracking
tech-stack:
  added: []
  patterns:
    - thin-wrapper CLI: calibrate_alpha.py contains zero optimization logic; all math in entropy.fit_calibration()
    - post-write self-check: load_calibration() called on just-written file to catch boundary convergence issues at write-time
    - importlib.util.spec_from_file_location: used in tests to load scripts/ without sys.path mutation
    - lazy imports in test functions: all hybridock_pep imports inside test bodies per STATE.md decision

key-files:
  created:
    - scripts/calibrate_alpha.py
    - data/training_complexes.csv
  modified:
    - tests/test_scoring.py
    - src/hybridock_pep/scoring/vina.py
    - src/hybridock_pep/scoring/ad4.py
    - src/hybridock_pep/scoring/entropy.py

key-decisions:
  - "calibrate_alpha.py aborts with ValueError if --scores-json not provided — live scoring wired in Phase 5; Phase 3 requires pre-computed scores"
  - "n_residues derived from len(peptide_sequence) in training CSV, not from scores JSON — CSV is the authoritative source of sequence info"
  - "Post-write self-check: load_calibration() called after write_calibration() to validate alpha/beta convergence before caller proceeds"
  - "black --target-version py311 required — base Python is 3.13 which cannot parse py314-targeted output; explicit target prevents AST parse failures"

# Metrics
duration: 3min
completed: 2026-04-21
---

# Phase 3 Plan 4: Calibration Script + Coverage Gate Summary

**calibrate_alpha.py thin CLI wrapper calling entropy.fit_calibration() with D-08 3-column training CSV + --scores-json; 30 scoring tests passing at 96% coverage on scoring/*.py**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-04-21T13:04:40Z
- **Completed:** 2026-04-21T13:07:51Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 6

## Accomplishments

- Created `data/training_complexes.csv` with D-08 3-column schema (pdb_id, peptide_sequence, experimental_pkd); 3 literature-derived rows (2OY2, 1YCR, 3LNJ)
- Replaced `@pytest.mark.skip` TestCalibration stub with 5 real integration tests
- Implemented `scripts/calibrate_alpha.py` (218 lines incl. docstrings): parse_args(), main(); reads D-08 CSV + --scores-json; calls fit_calibration() and write_calibration(); post-write self-check via load_calibration()
- All 30 scoring tests pass (8 Vina + 7 AD4 + 10 Entropy + 5 Calibration)
- Coverage gate passed: 96% on src/hybridock_pep/scoring/*.py (gate: ≥70%)
- ruff check: clean; black --target-version py311 --check: clean

## Task Commits

1. **Task 1: Replace TestCalibration stub + data/training_complexes.csv (RED)** - `6c3f9a5` (test)
2. **Task 2: Implement scripts/calibrate_alpha.py + ruff/black scoring cleanup (GREEN)** - `39932e2` (feat)

## Files Created/Modified

- `scripts/calibrate_alpha.py` - Thin CLI wrapper; parse_args() + main(); aborts with ValueError if --scores-json absent
- `data/training_complexes.csv` - D-08 3-column training data (3 literature rows)
- `tests/test_scoring.py` - Replaced TestCalibration skip stub with 5 integration tests
- `src/hybridock_pep/scoring/vina.py` - Removed unused TYPE_CHECKING/Vina import block; black reformatted
- `src/hybridock_pep/scoring/ad4.py` - Removed unused TYPE_CHECKING/Vina import block; black reformatted
- `src/hybridock_pep/scoring/entropy.py` - black reformatted (no logic change)

## Decisions Made

- `calibrate_alpha.py` aborts with a clear `ValueError` when `--scores-json` is not provided. Phase 3 calibration requires pre-computed scores; live Vina/AD4 scoring is wired in Phase 5 via the `hybridock-pep calibrate` subcommand.
- `n_residues` is derived from `len(row["peptide_sequence"])` from the training CSV — not from the scores JSON. The CSV is the authoritative source for sequence information.
- Post-write self-check: `load_calibration()` is called on the just-written calibration.json. This is belt-and-suspenders — L-BFGS-B bounds prevent convergence outside [0.2,1.2] / [0.0,0.5], but the self-check catches any edge case before the caller uses a corrupt file.
- `black --target-version py311` is required when running black under Python 3.13. Without the explicit target, black targets py314 and the AST safety check fails to parse the output. This is an infrastructure constraint, not a code issue.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Unused TYPE_CHECKING import blocks in vina.py and ad4.py flagged by ruff F401**
- **Found during:** Task 2 (first ruff check run)
- **Issue:** Both `vina.py` and `ad4.py` had `if TYPE_CHECKING: from vina import Vina as _VinaType` blocks that ruff flagged as unused (F401). These were pre-existing from Phase 3 plans 01 and 02 but the ruff gate was first enforced in this plan.
- **Fix:** Removed the `TYPE_CHECKING` guard and `_VinaType` alias from both files. The runtime lazy import (`try: from vina import Vina`) is sufficient for both test isolation and type checking.
- **Files modified:** `src/hybridock_pep/scoring/vina.py`, `src/hybridock_pep/scoring/ad4.py`
- **Commit:** `39932e2`

**2. [Rule 1 - Bug] `sys` imported but unused in calibrate_alpha.py**
- **Found during:** Task 2 (first ruff check run on new script)
- **Issue:** `import sys` was present in the initial draft of calibrate_alpha.py but not used (no `sys.argv` or `sys.exit` call in the script body).
- **Fix:** Removed the `sys` import.
- **Files modified:** `scripts/calibrate_alpha.py`
- **Commit:** `39932e2`

**3. [Rule 3 - Blocking] black AST safety check failure under Python 3.13**
- **Found during:** Task 2 (first black --check run)
- **Issue:** `black --check` without `--target-version` targets py314 by default when invoked under Python 3.13. The AST safety check then fails to parse the output because Python 3.13 cannot parse py314-targeted AST. This caused `black --check` to exit 1 on all three existing scoring modules.
- **Fix:** Added `--target-version py311` to all black invocations. This matches CLAUDE.md §4 (Python 3.11 for score-env code).
- **Files modified:** None (invocation flag only)
- **Commit:** N/A (infrastructure fix)

---

**Total deviations:** 3 auto-fixed (1 bug, 1 bug, 1 blocking)
**Impact on plan:** All fixes were inline; no scope changes.

## Known Stubs

None — calibrate_alpha.py is fully implemented. The `--scores-json` requirement is intentional (not a stub): live scoring is Phase 5 scope.

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns beyond what was planned. `calibrate_alpha.py` reads from filesystem (CSV + JSON) — covered by T-03-13 and T-03-14 (float() conversion raises ValueError on corrupt data). No unplanned threat surface introduced.

## Self-Check: PASSED

- `scripts/calibrate_alpha.py` exists: FOUND
- `data/training_complexes.csv` exists: FOUND
- commit `6c3f9a5` (RED test): FOUND
- commit `39932e2` (GREEN feat): FOUND
- 30 scoring tests pass, 0 failures: CONFIRMED
- Coverage 96% ≥ 70% gate: CONFIRMED
- ruff clean: CONFIRMED
- black clean: CONFIRMED

---
*Phase: 03-scoring-core*
*Completed: 2026-04-21*
