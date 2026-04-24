---
phase: "05"
plan: "03"
subsystem: cli-driver
tags: [driver, orchestration, pipeline, run_dock]
dependency_graph:
  requires:
    - 05-01  # DockConfig, ScoredPose models + test_driver.py RED gate
    - 05-02  # cli.py dispatch calling run_dock
    - 04-01  # run_sampling, parse_poses
    - 03-01  # prepare_receptor, generate_ad4_maps, prepare_ligand_batch
    - 03-02  # score_vina_batch
    - 03-03  # score_ad4_batch
    - 03-04  # load_calibration, apply_hybrid_score
    - 04-03  # write_metadata_skeleton, finalize_metadata
  provides:
    - hybridock_pep.driver.run_dock  # full pipeline orchestrator
  affects:
    - tests/test_driver.py  # 6 tests now GREEN
tech_stack:
  added: []
  patterns:
    - Bypass pattern (input_poses_dir != None skips run_sampling)
    - Stage-ordered orchestration with per-stage failure logging
    - pdbqt_by_stem dict for PoseRecord → ScoredPose pairing by filename stem
key_files:
  created:
    - src/hybridock_pep/driver.py
  modified: []
decisions:
  - "RuntimeError only when records > 0 and all ligand prep fails — zero records (bypass with empty dir) is not an error condition"
metrics:
  duration: "3 min"
  completed: "2026-04-24"
  tasks_completed: 1
  files_created: 1
  files_modified: 0
---

# Phase 05 Plan 03: driver.py Pipeline Orchestrator Summary

**One-liner:** `run_dock()` two-stage orchestrator wiring RAPiDock sampling through Vina/AD4/entropy scoring with metadata bookends and input-poses bypass for macOS.

## What Was Built

`src/hybridock_pep/driver.py` — the top-level pipeline function called by `cli._run_dock()`. Executes the following stages in mandatory order:

- **Stage 0:** `write_metadata_skeleton()` before any subprocess
- **Stage 1:** `run_sampling(config)` OR `parse_poses(input_poses_dir)` (D-01 bypass)
- **Stage 2a:** `prepare_receptor()` → `generate_ad4_maps()`
- **Stage 2b:** `prepare_ligand_batch()` for all pose PDB files
- **Stage 2c:** Pair `PoseRecord` objects with `pdbqt_path` by filename stem → build `ScoredPose` list
- **Stage 2d:** `score_vina_batch()` → `score_ad4_batch()` → `apply_hybrid_score()` per pose
- **Stage 3 stub:** Log handoff; clustering/output is Phase 6/7 scope
- **Finalize:** `finalize_metadata()` after scoring completes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RuntimeError guard conditioned on `records` being non-empty**

- **Found during:** First test run
- **Issue:** `test_input_poses_skips_run_sampling` passes `parse_poses` returning `([], [])` and `prepare_ligand_batch` returning `([], [])`. Zero records + zero pdbqt_paths triggered RuntimeError, but the test did not expect one — an empty bypass run (no poses to score) is valid, not an error.
- **Fix:** Changed `if not pdbqt_paths:` to `if not pdbqt_paths and records:` — RuntimeError only fires when poses existed but all failed prep.
- **Files modified:** `src/hybridock_pep/driver.py`
- **Commit:** 2c4302b (included in same commit)

## Test Results

- `tests/test_driver.py`: 6/6 passed
- Full suite: 136 passed, 1 skipped (slow integration test)

## Known Stubs

- Stage 3 logging stub: `"Clustering and output: Phase 6/7 not yet implemented"` — intentional; analysis/clustering wired in Phase 6.

## Threat Flags

None — driver.py introduces no new network endpoints, auth paths, or trust boundary crossings. All I/O is local filesystem via existing prep/scoring modules.

## Self-Check: PASSED

- `src/hybridock_pep/driver.py` exists: FOUND
- Commit 2c4302b: confirmed in git log
- 136 tests pass, 0 regressions
