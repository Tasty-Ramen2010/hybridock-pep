---
phase: 06-analysis-plots
plan: "03"
subsystem: analysis
tags: [statistics, ci95, csv, scipy, tdd]
dependency_graph:
  requires: [06-01, 06-02]
  provides: [compute_cluster_stats, write_cluster_summary_csv, _ci95]
  affects: [output/csv_writer.py, driver.py]
tech_stack:
  added: []
  patterns: [scipy.stats.t.interval, numpy ddof=1 std, module-level try/except import guard]
key_files:
  created: []
  modified:
    - src/hybridock_pep/analysis/statistics.py
decisions:
  - "scipy.stats.t.interval used directly (not t.ppf) per plan spec — scale=SEM (std/sqrt(n)), df=n-1"
  - "Module-level scipy import with ImportError guard — fallback to 1.96*SEM with warning log"
  - "n=1 degenerate case returns (value, value) — plan spec + test verbatim"
  - "compute_cluster_stats uses numpy for mean/std, delegates CI to _ci95"
metrics:
  duration: 12s
  completed: 2026-04-25
  tasks_completed: 1
  files_modified: 1
---

# Phase 6 Plan 03: statistics.py Implementation Summary

**One-liner:** Full statistics.py with scipy.stats.t.interval CI (df=n-1, scale=SEM), numpy-backed mean/std, and ordered CSV writer — all 4 TestStatistics tests GREEN.

## What Was Built

`src/hybridock_pep/analysis/statistics.py` (172 lines) — the ensemble statistics module for Phase 6.

Three public functions:

1. `_ci95(values)` — 95% confidence interval using `scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=SEM)`. Degenerates to `(value, value)` for n=1. Falls back to `mean ± 1.96*SEM` with a warning if scipy is absent.

2. `compute_cluster_stats(scored_poses)` — groups poses by `cluster_id`, computes 7-key dicts (`cluster_id`, `n_poses`, `mean_hybrid_score`, `std_hybrid_score`, `ci95_lower`, `ci95_upper`, `best_pose_idx`), returns list sorted by cluster_id ascending.

3. `write_cluster_summary_csv(stats, output_path)` — writes CSV with fixed column order matching the plan spec; creates parent dirs; logs via `logger.info`.

## Test Results

```
tests/test_clustering.py::TestStatistics::test_ci95_two_values       PASSED
tests/test_clustering.py::TestStatistics::test_ci95_single_value      PASSED
tests/test_clustering.py::TestStatistics::test_compute_cluster_stats_keys PASSED
tests/test_clustering.py::TestStatistics::test_cluster_summary_csv    PASSED
4/4 passed
```

TestClustering (06-02 scope) still GREEN — 5/5 pass, no regressions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced manual ppf-based CI with t.interval() as required**
- **Found during:** Task 1
- **Issue:** The stub created by plan 06-02 computed CI via `t_dist.ppf(0.975, df=n-1)` (mathematically correct but not the API the plan specifies). The plan acceptance criteria requires `t_dist.interval(0.95, df=n-1, loc=mean, scale=sem)` directly.
- **Fix:** Rewrote `_ci95()` to use `t_dist.interval()` with numpy-based mean/SEM computation. Also moved scipy import to module level with try/except guard per plan action spec.
- **Files modified:** `src/hybridock_pep/analysis/statistics.py`
- **Commit:** 6214690

## Known Stubs

None — all three functions are fully implemented and wired.

## Self-Check: PASSED

- `src/hybridock_pep/analysis/statistics.py` exists: FOUND
- Commit 6214690 exists: FOUND
- 4/4 TestStatistics tests GREEN: VERIFIED
- `grep "df=n - 1"` matches: VERIFIED
- `grep "t_dist.interval"` matches: VERIFIED
- Line count 172 >= 90 minimum: VERIFIED
