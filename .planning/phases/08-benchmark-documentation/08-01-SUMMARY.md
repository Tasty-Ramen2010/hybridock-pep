---
phase: 08-benchmark-documentation
plan: "01"
subsystem: testing
tags: [benchmark, csv, pytest, pdb-validation, test-scaffold]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: data/training_complexes.csv schema (pdb_id, peptide_sequence, experimental_pkd)
  - phase: 05-cli-driver
    provides: cli.py benchmark subcommand stub (_build_parser, main)
provides:
  - data/test_complexes.csv — 10 held-out benchmark complexes (pdb_id, peptide_sequence, experimental_pkd)
  - data/test_complexes_meta.csv — receptor/peptide chain ID mapping for site coordinate computation
  - tests/test_benchmark.py — structural test scaffold for benchmark.py (RED gate, 5 test classes)
affects:
  - 08-02 (benchmark.py implementation — tests/test_benchmark.py defines the interface it must satisfy)
  - 08-03 onward (uses data/test_complexes.csv as the benchmark dataset)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy benchmark module import via _SCRIPTS_DIR path injection (sys.path.insert) — same pattern as test_cli.py"
    - "RED gate tests: benchmark-importing tests fail with ModuleNotFoundError until Plan 08-02 ships benchmark.py"
    - "TestOutputSchema::test_results_csv_columns uses no external imports — validates CSV header schema independently"

key-files:
  created:
    - data/test_complexes.csv
    - data/test_complexes_meta.csv
    - tests/test_benchmark.py
  modified: []

key-decisions:
  - "Chain assignments in test_complexes_meta.csv are conventional defaults (A=receptor, B=peptide); must be verified from ATOM records before first benchmark run on RTX machine"
  - "RESEARCH.md Assumptions A1, A2, A7 flag 1PQ1, 4QVF, 3DAB peptide sequences as needing verification against literature"
  - "TestOutputSchema::test_results_csv_columns passes without benchmark.py — validates D-03 column schema independently using csv.DictWriter"

patterns-established:
  - "benchmark.py lives in scripts/ not src/ — imported via _SCRIPTS_DIR path injection in tests, not as a package"
  - "VALID_STATUSES constant in benchmark.py required — tested by TestOutputSchema::test_status_values_are_defined"
  - "validate_pdb_id(pdb_id) function required in benchmark.py — guards URL construction against injection (T-08-01)"

requirements-completed:
  - TEST-03

# Metrics
duration: 4min
completed: 2026-04-26
---

# Phase 8 Plan 01: Benchmark Dataset CSV Files and Test Scaffold Summary

**10-complex held-out benchmark dataset (test_complexes.csv + test_complexes_meta.csv) and RED-gate structural test scaffold for benchmark.py covering PDB ID injection validation, arg-parsing defaults, and D-03 output schema**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-26T17:41:39Z
- **Completed:** 2026-04-26T17:45:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Written `data/test_complexes.csv` with 10 held-out benchmark complexes, no overlap with training set (2OY2, 1YCR, 3LNJ), all PDB IDs matching `^[0-9][A-Z0-9]{3}$`
- Written `data/test_complexes_meta.csv` with receptor/peptide chain ID mapping (conventional A=receptor, B=peptide defaults)
- Written `tests/test_benchmark.py` with 5 test classes (16 tests total): 3 pass now against existing cli.py, 13 fail with ModuleNotFoundError (correct RED gate before Plan 08-02)
- T-08-01 threat mitigated by design: `TestPdbIdValidation` enforces `validate_pdb_id()` must exist in benchmark.py and reject malformed/injection-attempt IDs

## Task Commits

Each task was committed atomically:

1. **Task 1: Write data/test_complexes.csv and data/test_complexes_meta.csv** - `377d936` (feat)
2. **Task 2: Write tests/test_benchmark.py structural test scaffold** - `e990931` (test)

## Files Created/Modified

- `data/test_complexes.csv` — 10 held-out benchmark complexes; header: pdb_id, peptide_sequence, experimental_pkd
- `data/test_complexes_meta.csv` — chain ID mapping; header: pdb_id, receptor_chain, peptide_chain
- `tests/test_benchmark.py` — 5-class structural test scaffold; lazy imports throughout; RED gate for benchmark.py

## Decisions Made

- Chain assignments in `test_complexes_meta.csv` are conventional defaults (A=receptor, B=peptide); must be verified from ATOM records before the first benchmark run on the RTX machine. RESEARCH.md Assumptions A1, A2, A7 flag 1PQ1, 4QVF, 3DAB peptide sequences as needing verification.
- `TestOutputSchema::test_results_csv_columns` validates the D-03 CSV schema without importing `benchmark.py` — this test passes in the RED state and will continue to pass in the GREEN state.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `data/test_complexes.csv` and `data/test_complexes_meta.csv` ready for Plan 08-02 (benchmark.py implementation)
- `tests/test_benchmark.py` defines the full interface benchmark.py must satisfy (parse_args, validate_pdb_id, get_peptide_center, VALID_STATUSES)
- 13 RED tests will turn GREEN once Plan 08-02 ships `scripts/benchmark.py`

---
*Phase: 08-benchmark-documentation*
*Completed: 2026-04-26*
