---
phase: "05"
plan: "01"
subsystem: cli-driver
tags: [tdd, red-gate, test-scaffold, cli, driver]
dependency_graph:
  requires: [04-04]
  provides: [05-01-red-gate]
  affects: []
tech_stack:
  added: []
  patterns: [tdd-red-green-refactor, pytest-fixtures, unittest.mock.patch]
key_files:
  created:
    - tests/test_cli.py
    - tests/test_driver.py
  modified: []
decisions:
  - "RED gate only — no implementation created; driver.py and cli._build_parser left for Wave 1/2"
metrics:
  duration: "63s"
  completed_date: "2026-04-24"
---

# Phase 05 Plan 01: TDD RED Gate Test Scaffolds Summary

**One-liner:** TDD RED gate scaffolds for CLI subcommands/flags/validation and driver.run_dock() orchestration contract — 16 tests collected, 9 pass on existing code, 7 fail as expected.

## What Was Built

Two test files establishing the behavioral contract for Phase 5 Wave 1 (cli.py) and Wave 2 (driver.py) implementations:

**tests/test_cli.py** (16 tests total, 9 collected here):
- `TestSubcommands` — verifies dock/calibrate/prep/benchmark subcommands exist and print help (exit 0)
- `TestDockSubcommand` — verifies `cli._build_parser()` exposes all required flags including `--input-poses`, `--calibration`, `--scoring`, `--seed` (currently FAILS — `_build_parser` not yet exported)
- `TestValidation` — verifies invalid peptide chars exit 2, missing receptor exits 2, `--input-poses` + `--n-samples` mutual exclusion exits 2
- `TestSeed` — verifies `DockConfig.seed` stores value and defaults to None (PASSES — models.py already has DockConfig with seed field)

**tests/test_driver.py** (7 tests, all FAIL correctly):
- `TestInputPosesBypass` — verifies `run_dock()` skips `run_sampling` when `input_poses_dir` is provided, calls it when not
- `TestDriverOrchestration` — verifies return type is `list[ScoredPose]`, stage call ordering (metadata_skeleton before sampling, prepare_receptor before generate_ad4_maps, ligand_batch before vina_batch before ad4_batch before finalize), prep failures propagate as RuntimeError, metadata written once at start and once at end

## RED Gate Verification

```
7 failed, 9 passed in 0.21s
```

Failures are all `ImportError: cannot import name 'driver' from 'hybridock_pep'` (driver.py not yet written) and `AttributeError` on `cli._build_parser` (not yet exported). Both are correct RED gate failures.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

RED gate commit exists: f7a5414 — `test(05-01): add RED gate test scaffolds for CLI and driver`

GREEN gate commit will be created in Plans 05-02 (cli) and a Wave 2 plan (driver).

## Self-Check: PASSED

- tests/test_cli.py: FOUND
- tests/test_driver.py: FOUND
- Commit f7a5414: FOUND
