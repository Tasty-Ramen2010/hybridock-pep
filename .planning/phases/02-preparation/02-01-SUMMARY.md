---
phase: 02-preparation
plan: "01"
subsystem: prep
tags: [prep, receptor, pdbfixer, subprocess, exception, fixtures, tdd]
dependency_graph:
  requires: []
  provides:
    - hybridock_pep.prep.PrepError
    - hybridock_pep.prep.receptor.prepare_receptor
    - hybridock_pep.prep.receptor._filter_pdb_lines
    - tests/fixtures/receptor_tiny.pdb
    - tests/fixtures/pose_tiny.pdb
  affects:
    - prep/ligand.py (uses PrepError)
    - prep/grids.py (uses PrepError)
    - tests/test_prep.py (uses fixtures and PrepError)
tech_stack:
  added:
    - pdbfixer>=1.9 (score-env, already in env)
    - openmm.app.PDBFile (score-env, already in env)
  patterns:
    - TDD red-green with pytest + unittest.mock.patch
    - Subprocess wrapper with capture_output=True, check=False, hard abort on non-zero
    - PDB pre-filtering via raw text scan before pdbfixer
    - tempfile.NamedTemporaryFile for intermediate PDB files with unlink(missing_ok=True) cleanup
key_files:
  created:
    - src/hybridock_pep/prep/errors.py
    - src/hybridock_pep/prep/receptor.py
    - tests/fixtures/receptor_tiny.pdb
    - tests/fixtures/pose_tiny.pdb
    - tests/test_prep.py
  modified:
    - src/hybridock_pep/prep/__init__.py
decisions:
  - D-01 honored: pdbfixer runs findMissingResidues, findMissingAtoms, addMissingHydrogens(7.4) unconditionally
  - D-02 honored: no pdbqt_path.exists() guard — always regenerates
  - D-03 honored: non-zero returncode raises PrepError with full stderr captured immediately
  - Pre-filtering via _filter_pdb_lines() before pdbfixer — strips altLoc B/C/... and non-water HETATM in a single text pass to prevent "Unknown Receptor Type" in autogrid4
metrics:
  duration: "~3 min"
  completed: "2026-04-20"
  tasks_completed: 2
  files_created: 5
  files_modified: 1
---

# Phase 2 Plan 1: PrepError, Receptor Prep, and Test Fixtures Summary

**One-liner:** PrepError(RuntimeError) exception + pdbfixer→prepare_receptor4.py receptor pipeline with altLoc/HETATM pre-filtering, both in TDD with 16 passing unit tests.

## What Was Built

### Task 1: PrepError + prep/__init__.py + fixtures (TDD)

- `src/hybridock_pep/prep/errors.py`: `PrepError(RuntimeError)` exception class. Raised on `prepare_receptor4.py` non-zero exit and (future) autogrid4 HD map missing.
- `src/hybridock_pep/prep/__init__.py`: exports `PrepError` from `errors` module.
- `tests/fixtures/receptor_tiny.pdb`: deterministic 3-ATOM minimal receptor (ALA backbone).
- `tests/fixtures/pose_tiny.pdb`: deterministic 15-ATOM ALA-ALA-ALA peptide pose with all backbone + CB atoms.

### Task 2: prep/receptor.py (TDD)

`prepare_receptor(config: DockConfig) -> Path` implements the full PREP-01 pipeline:

1. `_filter_pdb_lines()` scans raw PDB text and drops non-water HETATM records and alternate-occupancy atoms (altLoc B/C/...). Waters (HOH/WAT) are kept.
2. Filtered text is written to a first temp file.
3. `PDBFixer` loads the cleaned PDB, then `findMissingResidues()`, `findMissingAtoms()`, `addMissingHydrogens(7.4)` run unconditionally (D-01).
4. pdbfixer output is written via `PDBFile.writeFile()` to a second temp file.
5. `subprocess.run(["prepare_receptor4.py", "-r", ..., "-o", ...])` is called with the full command logged at INFO before execution.
6. Non-zero `returncode` → `raise PrepError(f"prepare_receptor4.py failed (exit {result.returncode}):\n{result.stderr}")` (D-03).
7. Both temp files are cleaned up via `unlink(missing_ok=True)` in `finally` blocks.
8. Returns `output_dir / "receptor.pdbqt"`.

No `pdbqt_path.exists()` guard anywhere — always regenerates (D-02).

## Test Coverage

16 tests in `tests/test_prep.py`:
- `TestPrepError` (3 tests): import, isinstance(RuntimeError), message preservation
- `TestFixtures` (4 tests): existence and content of both fixture PDBs
- `TestFilterPdbLines` (5 tests): passthrough ATOM, drop non-water HETATM, keep HOH, drop altLoc B, keep altLoc A
- `TestPrepareReceptor` (4 tests): success path returns pdbqt_path, non-zero exit raises PrepError, no caching guard, pdbfixer 3-step sequence

All 26 tests pass (10 from test_models.py + 16 from test_prep.py).

## Deviations from Plan

None — plan executed exactly as written.

The plan noted that the initial code snippet for receptor.py had dead code (a first PDBFixer instantiation before the pre-filter). The actual implementation in the plan's `<action>` block explicitly instructs to remove it. The implementation follows the clean path described there.

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns introduced beyond what the plan's threat model covers.

- T-02-03 (PATH injection): cmd is a list, not a string — no shell=True. Accepted.
- T-02-04 (stderr disclosure): stderr only surfaced in PrepError message to local user. Accepted.

## Self-Check: PASSED

All created files verified present on disk. All task commits verified in git log.

| Item | Status |
|------|--------|
| src/hybridock_pep/prep/errors.py | FOUND |
| src/hybridock_pep/prep/receptor.py | FOUND |
| src/hybridock_pep/prep/__init__.py | FOUND |
| tests/fixtures/receptor_tiny.pdb | FOUND |
| tests/fixtures/pose_tiny.pdb | FOUND |
| tests/test_prep.py | FOUND |
| ab25209 (test RED) | FOUND |
| 97c567d (feat task1) | FOUND |
| 7ffa3ac (feat task2) | FOUND |
