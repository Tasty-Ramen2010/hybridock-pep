---
phase: 03-scoring-core
plan: 03
subsystem: scoring
tags: [scipy, entropy, calibration, hybrid-score, l-bfgs-b]

# Dependency graph
requires:
  - phase: 03-01
    provides: score_vina_batch(), ScoredPose.vina_score
  - phase: 03-02
    provides: score_ad4_batch(), ScoredPose.ad4_score

provides:
  - apply_hybrid_score(): D-01 formula — hybrid = vina + beta*(ad4-vina) + alpha*n_residues
  - load_calibration(): validates alpha in [0.2,1.2] and beta in [0.0,0.5] with diagnostic ValueError
  - write_calibration(): D-11 schema JSON writer with UTC ISO 8601 timestamp
  - fit_calibration(): scipy L-BFGS-B optimizer with hardcoded bounds and RT=0.592 conversion
  - data/calibration.json: shipped default (alpha=0.65, beta=0.22) with all 7 D-11 fields
  - scoring/__init__.py: exports all 7 public scoring functions

affects:
  - 03-04 (calibration CLI subcommand consumes fit_calibration and write_calibration)
  - 04 (driver.py orchestrates hybrid scoring per pose)
  - 05 (analysis/clustering.py ranks poses by hybrid_score)

# Tech tracking
tech-stack:
  added: [scipy (L-BFGS-B optimizer, pearsonr), numpy (array ops)]
  patterns:
    - load-validate-return: load_calibration reads JSON then range-validates before returning
    - TDD RED/GREEN: failing import tests committed first, then implementation
    - lazy-import pattern: all hybridock_pep imports inside test functions (numpy double-import guard)

key-files:
  created:
    - src/hybridock_pep/scoring/entropy.py
    - data/calibration.json
  modified:
    - tests/test_scoring.py
    - src/hybridock_pep/scoring/__init__.py

key-decisions:
  - "apply_hybrid_score() does NOT validate alpha/beta — validation is load_calibration()'s sole responsibility (separation of concerns, T-03-09)"
  - "RT = 0.592 kcal/mol hardcoded at 298K; not a CLI parameter in v1 (D-09)"
  - "scipy installed in test env (base Python) to unblock TestEntropy — score-env is the production target but tests must run locally"
  - "fit_calibration returns nan for pearson_r when n=1 (pearsonr requires >=2 points); documented in code"

patterns-established:
  - "Entropy module pattern: load → validate → apply → fit (four orthogonal functions, no shared state)"
  - "Calibration JSON D-11 schema: alpha, beta, n_complexes, pearson_r, rmse_kcal_mol, calibrated_at, training_csv"

requirements-completed: [SCORE-03]

# Metrics
duration: 5min
completed: 2026-04-21
---

# Phase 3 Plan 3: Entropy Correction Summary

**Backbone entropy correction with D-01 hybrid score formula, scipy L-BFGS-B calibration (alpha=0.65, beta=0.22), and load_calibration() ValueError guard on out-of-range coefficients**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-21T12:58:05Z
- **Completed:** 2026-04-21T13:02:30Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments

- Implemented `apply_hybrid_score()` with D-01 formula: `hybrid = vina + beta*(ad4-vina) + alpha*n_residues`; sets both `entropy_correction` and `hybrid_score` on ScoredPose in-place
- Implemented `load_calibration()` with dual range validation — raises ValueError quoting the bad value and valid range for both alpha (out of [0.2,1.2]) and beta (out of [0.0,0.5])
- Implemented `fit_calibration()` with scipy L-BFGS-B, x0=[0.65,0.22], pKd→ΔG via RT=0.592 kcal/mol (D-09)
- Shipped `data/calibration.json` with all 7 D-11 fields (alpha=0.65, beta=0.22)
- All 25 scoring tests pass (8 Vina + 7 AD4 + 10 Entropy); 0 regressions

## Task Commits

1. **Task 1: Replace TestEntropy stub with 10 failing tests (RED)** - `4c11bfa` (test)
2. **Task 2: Implement scoring/entropy.py + data/calibration.json + __init__.py (GREEN)** - `5dda0ec` (feat)

## Files Created/Modified

- `src/hybridock_pep/scoring/entropy.py` - Four public functions: load_calibration, write_calibration, apply_hybrid_score, fit_calibration
- `data/calibration.json` - Shipped default calibration with D-11 schema (alpha=0.65, beta=0.22)
- `tests/test_scoring.py` - Replaced TestEntropy skip stub with 10 real tests
- `src/hybridock_pep/scoring/__init__.py` - Added entropy exports; now exports all 7 public functions

## Decisions Made

- `apply_hybrid_score()` does not validate alpha/beta ranges — validation is exclusively `load_calibration()`'s responsibility. This separates concerns: the function that reads untrusted input (filesystem JSON) is the one that validates; the arithmetic function that receives already-validated params does not re-check.
- RT = 0.592 kcal/mol hardcoded (D-09). Not a CLI flag in v1. Noted in code with comment.
- `fit_calibration()` returns `nan` for `pearson_r` when called with a single data point (scipy pearsonr requires >= 2 points). The test `test_pkd_to_delta_g_conversion` uses n=1 and only checks that keys exist — this is intentional; the production calibration workflow requires at minimum 10 complexes (D-13).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] scipy not installed in base Python test environment**
- **Found during:** Task 2 (first test run after creating entropy.py)
- **Issue:** `ModuleNotFoundError: No module named 'scipy'` — scipy is a score-env dep, not in base Python env used by pytest
- **Fix:** `python3 -m pip install scipy` to unblock local test runs; production runs use score-env where scipy is declared in envs/score-env.yml
- **Files modified:** None (dependency install only)
- **Verification:** All 10 TestEntropy tests pass after install
- **Committed in:** 5dda0ec (part of feat commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** scipy install necessary to complete GREEN phase locally. No scope creep.

## Issues Encountered

None — plan executed smoothly after resolving scipy availability.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- scoring/entropy.py complete: apply_hybrid_score(), load_calibration(), fit_calibration(), write_calibration() all implemented and tested
- scoring/__init__.py exports all 7 functions (check_grid_boundary, score_vina_batch, score_ad4_batch, load_calibration, write_calibration, apply_hybrid_score, fit_calibration)
- data/calibration.json shipped with valid D-11 schema; passes load_calibration() validation
- Plan 03-04 (calibrate CLI subcommand) can consume fit_calibration() and write_calibration() directly
- Phase 4 driver can call apply_hybrid_score() per pose after Vina and AD4 scoring complete

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns beyond what was planned. load_calibration() reads from filesystem — covered by T-03-09 (validated on every read). No unplanned threat surface introduced.

## Known Stubs

None — all exported functions are fully implemented with no hardcoded placeholders or TODO stubs.

## Self-Check: PASSED

- `src/hybridock_pep/scoring/entropy.py` exists: FOUND
- `data/calibration.json` exists: FOUND
- commit `4c11bfa` (RED): FOUND
- commit `5dda0ec` (GREEN): FOUND
- 25 scoring tests pass, 0 failures: CONFIRMED

---
*Phase: 03-scoring-core*
*Completed: 2026-04-21*
