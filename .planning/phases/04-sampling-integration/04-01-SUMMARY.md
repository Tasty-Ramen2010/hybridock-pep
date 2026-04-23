---
phase: 04-sampling-integration
plan: "01"
subsystem: sampling
tags: [tdd, test-scaffold, rapidock, pose-io, metadata, samp-01, samp-02]

dependency_graph:
  requires: []
  provides:
    - tests/test_sampling.py (TestRapidockRunner + TestPoseIO — 10 failing stubs)
    - tests/test_output.py (TestMetadata — 8 failing stubs)
  affects:
    - 04-02-PLAN.md (rapidock_runner + pose_io implementation — green signal target)
    - 04-03-PLAN.md (output.metadata implementation — green signal target)

tech_stack:
  added: []
  patterns:
    - Lazy hybridock_pep imports inside test methods — prevents ModuleNotFoundError in base Python env (established Phase 3 pattern)
    - monkeypatch-scoped subprocess.Popen mock — auto-reset after each test (T-04-01-02 mitigation)
    - sentinel-loop readline mock (side_effect=[b""]) — correctly terminates iter(readline, b"") pattern
    - SEQRES-first sequence extraction test (D-14) — two complementary tests: preferred path and ATOM fallback

key_files:
  created:
    - tests/test_sampling.py
    - tests/test_output.py
  modified: []

decisions:
  - "Wrote 5 TestPoseIO tests not 4: plan spec listed D-14 as one blocker covering two behaviors (SEQRES-first and ATOM fallback), correctly split into test_parse_seqres_preferred and test_parse_atom_fallback — total 10 tests matches plan success criteria"
  - "test_shortfall_warns uses caplog.at_level(WARNING) with propagate — required because rapidock_runner uses module-level logging; caplog captures root logger by default"
  - "test_command_construction path-absoluteness check uses suffix and '/' heuristic to avoid false positives on flags like '--no-capture-output'"
  - "test_atomic_write_uses_tmp_file spies via side_effect wrapping real os.replace — avoids suppressing the actual write while still capturing call args"

metrics:
  duration_seconds: 177
  completed_date: "2026-04-23"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 4 Plan 01: Sampling Test Scaffolds Summary

**One-liner:** Failing test scaffolds for RAPiDock runner, pose I/O, and metadata — 18 stubs across 3 classes, all imports lazy, covering D-14 SEQRES-first and atomic-write behaviors.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write tests/test_sampling.py — TestRapidockRunner + TestPoseIO stubs | 5279399 | tests/test_sampling.py |
| 2 | Write tests/test_output.py — TestMetadata stubs | 63d59b2 | tests/test_output.py |

## Success Criteria Verification

- tests/test_sampling.py collected: 10 items (5 TestRapidockRunner + 5 TestPoseIO including SEQRES-first tests) — PASS
- tests/test_output.py collected: 8 items (TestMetadata) — PASS
- No module-level hybridock_pep imports in either test file — PASS
- Existing test suite (test_models, test_scoring) still passes (40/40) — PASS
- Both files committed to git — PASS

Note: test_prep.py has a pre-existing failure (pdbfixer not in base env — score-env dependency). This failure predates this plan and is out of scope.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

All test methods in this plan are intentionally failing stubs. They will be wired to real implementations in Wave 2 plans (04-02, 04-03). The stubs fail with `ModuleNotFoundError: No module named 'hybridock_pep.sampling.rapidock_runner'` and `No module named 'hybridock_pep.output.metadata'`.

## Threat Flags

None — this plan creates test files only; no new network endpoints, auth paths, file access patterns, or schema changes were introduced.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| tests/test_sampling.py exists | FOUND |
| tests/test_output.py exists | FOUND |
| 04-01-SUMMARY.md exists | FOUND |
| commit 5279399 exists | FOUND |
| commit 63d59b2 exists | FOUND |
