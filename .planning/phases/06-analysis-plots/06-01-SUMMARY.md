---
phase: 06-analysis-plots
plan: "01"
subsystem: tests
tags: [tdd, red-gate, clustering, statistics, plotting, analysis]
dependency_graph:
  requires: []
  provides:
    - tests/test_clustering.py (RED-gate stubs for ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05)
  affects:
    - src/hybridock_pep/analysis/clustering.py (Wave 1 must satisfy these tests)
    - src/hybridock_pep/analysis/statistics.py (Wave 1 must satisfy these tests)
    - src/hybridock_pep/analysis/plotting.py (Wave 1 must satisfy these tests)
tech_stack:
  added: []
  patterns:
    - lazy imports (all hybridock_pep imports inside test function bodies)
    - synthetic pose fixtures as module-level constants (not pytest fixtures)
    - class-based test organization (TestClustering, TestStatistics, TestPlotting)
key_files:
  created:
    - tests/test_clustering.py
  modified: []
decisions:
  - "test_ci95 split into two methods (test_ci95_two_values, test_ci95_single_value) to independently verify n=2 CI and n=1 degenerate cases — plan's verbatim code included both"
  - "11 test methods written (plan verbatim template has 11); plan acceptance criteria said 10 but the provided code had 11 — code template takes precedence"
metrics:
  duration: "2 min"
  completed: "2026-04-25"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 06 Plan 01: RED-gate Test Scaffold Summary

**One-liner:** RED-gate test file for analysis modules — 11 stubs across TestClustering/TestStatistics/TestPlotting, all failing with ModuleNotFoundError before Wave 1 implementation.

## What Was Built

`tests/test_clustering.py` (221 lines) — the Wave 0 TDD scaffold for Phase 6.

Three test classes covering all five requirement IDs:

| Class | Tests | Requirements |
|-------|-------|--------------|
| TestClustering | 5 | ANAL-01 |
| TestStatistics | 4 | ANAL-02 |
| TestPlotting | 2 | ANAL-03, OUT-04, OUT-05 |

All imports of `hybridock_pep.analysis.*` are lazy (inside test method bodies). The file collects cleanly under pytest and all 11 tests fail with `ModuleNotFoundError: No module named 'hybridock_pep.analysis.clustering'` — confirming RED gate.

## TDD Gate Compliance

- RED phase: confirmed — `pytest tests/test_clustering.py -x` exits non-zero (ModuleNotFoundError)
- GREEN phase: pending — Wave 1 plans (06-02, 06-03) must make these pass
- REFACTOR: N/A for Wave 0

## Deviations from Plan

### Minor Count Discrepancy (auto-resolved)

The plan's `<implementation>` section says "10 stub test methods" and the acceptance criteria `grep -c "def test_"` says 10, but the plan's verbatim code block contains both `test_ci95_two_values` and `test_ci95_single_value` as separate tests (splitting what the GREEN list calls `test_ci95`). Following the verbatim code template produced 11 tests. This is a self-consistent deviation — the plan's provided code is authoritative, and 11 >= 10 satisfies the minimum.

No Rule 1/2/3/4 deviations applied.

## Known Stubs

None — this is a test file. All test methods are intentional RED-gate stubs; their emptiness is the plan's goal.

## Threat Flags

None — test files operate on synthetic in-memory data only; no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check

- [x] `tests/test_clustering.py` exists (221 lines, >= 120)
- [x] Three test classes: TestClustering, TestStatistics, TestPlotting
- [x] All hybridock_pep imports lazy (no module-level imports)
- [x] `python3 -m py_compile tests/test_clustering.py` exits 0
- [x] `pytest tests/test_clustering.py -x` fails RED (ModuleNotFoundError)
- [x] Commit 34ae9cb confirmed in git log

## Self-Check: PASSED
