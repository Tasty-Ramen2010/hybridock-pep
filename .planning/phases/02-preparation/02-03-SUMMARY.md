---
phase: 02-preparation
plan: "03"
subsystem: prep
tags: [prep, grids, autogrid4, gpf, ad4, tdd]
dependency_graph:
  requires:
    - hybridock_pep.prep.PrepError
    - hybridock_pep.models.DockConfig
    - src/hybridock_pep/prep/receptor.py (produces receptor.pdbqt consumed here)
  provides:
    - hybridock_pep.prep.grids.generate_ad4_maps
    - hybridock_pep.prep.grids._build_gpf
    - output_dir/maps/receptor.gpf (programmatic, from DockConfig)
    - output_dir/maps/receptor.HD.map (existence verified post-autogrid4)
  affects:
    - scoring/ad4.py (consumes maps from output_dir/maps/)
    - driver.py Stage 2 (calls generate_ad4_maps after receptor prep)
tech_stack:
  added: []
  patterns:
    - Programmatic GPF construction from validated Pydantic model fields
    - Subprocess wrapper with cwd=maps_dir, capture_output=True, timeout=120s
    - Hard-abort HD map guard (PrepError) after subprocess completes
    - TDD red-green with pytest + unittest.mock.patch on subprocess.run
key_files:
  created:
    - src/hybridock_pep/prep/grids.py
  modified:
    - tests/test_prep.py (added 3 test classes, 19 tests)
decisions:
  - D-04 honored: GPF constructed entirely in Python from DockConfig — no template file on disk
  - D-05 honored: PrepError raised with verbatim message if receptor.HD.map absent post-autogrid4
  - D-06 honored: _LIGAND_TYPES = "C A N O S H HD"; _RECEPTOR_TYPES = "C A N O SA S H HD"
  - D-07 honored: all autogrid4 outputs written to output_dir/maps/
  - Grid spacing 0.375 Å — AutoDock standard, spec does not override
  - receptor.pdbqt copied into maps_dir before autogrid4 so relative filenames work with cwd
metrics:
  duration: "~5 min"
  completed: "2026-04-20"
  tasks_completed: 1
  files_created: 1
  files_modified: 1
---

# Phase 2 Plan 3: Grid Parameter File Builder and autogrid4 Wrapper Summary

**One-liner:** Programmatic GPF generation from DockConfig + autogrid4 subprocess wrapper with HD map existence guard that hard-aborts with PrepError if receptor.HD.map is absent.

## What Was Built

### Task 1: prep/grids.py (TDD)

`src/hybridock_pep/prep/grids.py` implements PREP-03 with two exported symbols:

**`_build_gpf(config: DockConfig, maps_dir: Path) -> str`**

Constructs the autogrid4 Grid Parameter File content entirely in Python from `DockConfig` fields — no template file on disk (D-04). Key computed values:

- `npts = int(config.box_size / 0.375)` — cubic box, integer grid points
- `gridcenter = config.site_coords[0] config.site_coords[1] config.site_coords[2]`
- `_RECEPTOR_TYPES = "C A N O SA S H HD"` — SA for aromatic carbon (standard autogrid4)
- `_LIGAND_TYPES = "C A N O S H HD"` — HD here is what causes receptor.HD.map to be generated
- `receptor receptor.pdbqt` — filename only (autogrid4 runs with cwd=maps_dir, relative paths required)

**`generate_ad4_maps(config: DockConfig, receptor_pdbqt: Path) -> Path`**

Full autogrid4 pipeline:

1. Creates `output_dir/maps/` (D-07) with `mkdir(parents=True, exist_ok=True)`.
2. Copies `receptor_pdbqt` into `maps_dir/receptor.pdbqt` — autogrid4 needs it alongside the GPF.
3. Builds GPF via `_build_gpf()`, writes to `maps_dir/receptor.gpf`.
4. Logs the full command at INFO level before execution.
5. Runs `autogrid4 -p receptor.gpf -l receptor.glg` with `cwd=str(maps_dir)`, `timeout=120`, `capture_output=True`.
6. Non-zero `returncode` → `PrepError(f"autogrid4 failed (exit {returncode}):\n{stderr}")`.
7. HD map guard (D-05): if `maps_dir/receptor.HD.map` does not exist → `PrepError("receptor.HD.map not found after autogrid4 — AD4 scoring will fail. Check your atom types in the GPF.")` — verbatim message, hard abort.
8. Returns `maps_dir` on success.

## Test Coverage

19 new tests across 3 classes in `tests/test_prep.py`:

- `TestGridsImports` (5 tests): importable, `_build_gpf` importable, `from __future__ import annotations` first, no bare except, no template reference
- `TestBuildGpf` (6 tests): HD in ligand_types, gridcenter matches site_coords, npts = int(box_size/0.375), receptor line is filename-only, HD map line present, spacing = 0.375
- `TestGenerateAd4Maps` (8 tests): returns maps_dir, creates dir if absent, GPF written, HD map guard (match + exact D-05 message), cwd passed correctly, non-zero exit raises PrepError, receptor.pdbqt copied

Total test suite: 61 tests pass (42 pre-existing + 19 new).

## TDD Gate Compliance

- RED commit: `f3c6a58` — `test(02-03): add failing tests for grids.py (PREP-03)` — 32 passing, 1 failing (import error on missing module)
- GREEN commit: `6444342` — `feat(02-03): implement prep/grids.py — GPF builder + autogrid4 wrapper + HD map guard` — 61 passing, 0 failing
- REFACTOR: not needed — implementation was clean on first pass

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. `generate_ad4_maps()` and `_build_gpf()` are fully implemented with no placeholder values or TODO markers.

## Threat Surface Scan

No new network endpoints or auth paths introduced. The subprocess boundary (Python → autogrid4) was already in the plan's threat model:

- T-02-08 (DoS/timeout): `timeout=120` on `subprocess.run` — raises `subprocess.TimeoutExpired` which propagates to caller. Mitigated.
- T-02-09 (GPF tampering): GPF generated from validated Pydantic model; no user string interpolation. Accepted.
- T-02-10 (PATH injection): `cmd` is a list, `shell=False` (default). Accepted.
- T-02-11 (stderr disclosure): stderr surfaced only in local PrepError message. Accepted.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| src/hybridock_pep/prep/grids.py | FOUND |
| tests/test_prep.py (modified) | FOUND |
| f3c6a58 (test RED) | FOUND |
| 6444342 (feat GREEN) | FOUND |
