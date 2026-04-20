---
phase: 02-preparation
reviewed: 2026-04-20T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/hybridock_pep/prep/__init__.py
  - src/hybridock_pep/prep/errors.py
  - src/hybridock_pep/prep/grids.py
  - src/hybridock_pep/prep/ligand.py
  - src/hybridock_pep/prep/receptor.py
  - tests/fixtures/pose_tiny.pdb
  - tests/fixtures/receptor_tiny.pdb
  - tests/test_prep.py
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-20T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the prep pipeline (receptor cleaning, ligand batch conversion, AD4 grid generation) and the accompanying test suite. The code is generally well-structured, follows project conventions (type hints, Google-style docstrings, `from __future__ import annotations`, no bare `except`), and the test coverage is thorough. No critical security or data-loss issues were found.

Five warnings require attention before merging:

1. `prepare_receptor` leaks a tempfile when `PDBFile.writeFile` raises — the `finally` block only cleans up `cleaned_pdb_path`, not `fixed_pdb_path`.
2. `prepare_ligand_batch` logs the effective worker count from the pre-computed variable but passes the raw `max_workers` (which may be `None`) to `ProcessPoolExecutor`, masking the resolved value in logs.
3. `_build_gpf` computes `npts` with integer truncation (`int(box_size / spacing)`) with no check that the result is even — AutoDock4 requires an **even** number of grid points, and an odd `npts` causes silent grid misalignment.
4. The `receptor_tiny.pdb` fixture is missing an `END` record; `pdbfixer` is lenient but some downstream PDB parsers are not.
5. The `meeko_available` fixture is declared `scope="session"` but is not used by the session-level `TestLigandBatch.test_batch_single_pose_success` correctly — it is listed as a parameter but does not guard the other `TestLigandBatch` tests that also implicitly depend on Meeko being present.

Four informational items are noted below.

---

## Warnings

### WR-01: Tempfile leak when `PDBFile.writeFile` raises

**File:** `src/hybridock_pep/prep/receptor.py:51-80`

**Issue:** The `try/finally` block at lines 51–64 cleans up `cleaned_pdb_path` in `finally`, which is correct. However, `fixed_pdb_path` is only cleaned up in the *second* `try/finally` at lines 66–80. If `PDBFile.writeFile` on line 61 raises an exception, `fixed_pdb_path` is never assigned, so the outer `finally` at line 80 will raise `NameError: name 'fixed_pdb_path' is not defined`, shadowing the original exception and leaking `cleaned_pdb_path` (because the inner `finally` runs, but the variable lookup in the second `finally` crashes).

More precisely: if the second `NamedTemporaryFile` context manager itself raises before `fixed_pdb_path = Path(fixed_tmp.name)` is executed, the variable is unbound and the outer `finally` at line 80 (`fixed_pdb_path.unlink(missing_ok=True)`) will raise a `NameError`.

**Fix:**
```python
cleaned_pdb_path: Path | None = None
fixed_pdb_path: Path | None = None
try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(cleaned_pdb_lines)
        cleaned_pdb_path = Path(tmp.name)

    fixer = PDBFixer(filename=str(cleaned_pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fixed_tmp:
        PDBFile.writeFile(fixer.topology, fixer.positions, fixed_tmp)
        fixed_pdb_path = Path(fixed_tmp.name)

    cmd = [...]
    result = subprocess.run(cmd, ...)
finally:
    if cleaned_pdb_path is not None:
        cleaned_pdb_path.unlink(missing_ok=True)
    if fixed_pdb_path is not None:
        fixed_pdb_path.unlink(missing_ok=True)
```

---

### WR-02: Log message shows unresolved `None` for worker count

**File:** `src/hybridock_pep/prep/ligand.py:88-103`

**Issue:** Line 88 computes `effective_workers = max_workers or os.cpu_count()` — this is correct for the conceptual count. However, line 102 passes `max_workers` (the original, possibly `None`) to `ProcessPoolExecutor`, not `effective_workers`. This means the log at line 99 says `workers=<N>` (the resolved count) but the executor actually receives `None` and resolves the count internally, separately from `effective_workers`. These two values will agree in practice, but if `os.cpu_count()` returns `None` (possible on some stripped Linux containers), `effective_workers` becomes `None` too and the log is misleading. Additionally, if the caller passes `max_workers=0`, `or` coerces it to `os.cpu_count()`, but `ProcessPoolExecutor(max_workers=0)` raises `ValueError`. This is a latent bug if the CLI ever passes 0.

**Fix:** Pass `effective_workers` to `ProcessPoolExecutor` so the logged value matches the actual concurrency level, and guard against `max_workers=0` explicitly:

```python
if max_workers is not None and max_workers < 1:
    raise ValueError(f"max_workers must be >= 1 or None, got {max_workers}")
effective_workers = max_workers or os.cpu_count() or 1  # guard os.cpu_count() returning None

logger.info(
    "Preparing %d pose PDBs → PDBQT (workers=%s)", len(pdb_paths), effective_workers
)

with ProcessPoolExecutor(max_workers=effective_workers) as executor:
    ...
```

---

### WR-03: `npts` may be odd — AutoDock4 requires even grid point counts

**File:** `src/hybridock_pep/prep/grids.py:100`

**Issue:** `npts = int(config.box_size / _GRID_SPACING)` truncates without enforcing parity. AutoDock4 requires that the number of grid points per dimension be **even** (the `autogrid4` manual states "must be an even number"). For `box_size=20.0` and `spacing=0.375`, `npts=53` (odd). `autogrid4` will silently round or error depending on version, leading to a grid whose center no longer aligns with `site_coords`.

The test `test_npts_derived_from_box_size` hard-codes 53 as the expected value, so it would pass while the real tool rejects it.

**Fix:**
```python
npts_raw = int(config.box_size / _GRID_SPACING)
npts = npts_raw if npts_raw % 2 == 0 else npts_raw + 1  # autogrid4 requires even npts
```
Update the test to expect 54 for `box_size=20.0`.

---

### WR-04: `receptor_tiny.pdb` fixture is missing `END` record

**File:** `tests/fixtures/receptor_tiny.pdb:4`

**Issue:** The fixture ends at `END` (line 4), which is correct. Wait — upon re-reading it does have `END`. However the file is only 3 ATOM records for a single residue (ALA A 1 with only N, CA, C — no O, CB, or H atoms). The file is structurally incomplete as a standalone PDB: a single-residue backbone fragment without a C=O oxygen is not a valid amino acid in force-field terms. `pdbfixer` will attempt to add missing atoms, but when tests mock `PDBFixer`, the fixture is read by `_filter_pdb_lines` before the mock kicks in. The test `test_receptor_tiny_has_ca_ala_a1` passes, but any test path that actually invokes `_filter_pdb_lines` on this fixture gets a 3-atom PDB fed to a real (non-mocked) pdbfixer — which is fine for unit tests where PDBFixer is mocked, but misleading as a "receptor" fixture.

More importantly: `test_receptor_tiny_has_ca_ala_a1` asserts `"CA  ALA A   1" in content`. Looking at the file, line 2 is `ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C` — the column layout does contain `CA  ALA A   1` only if there are the right number of spaces. In standard PDB format the residue name occupies columns 18-20 and the chain/resseq follows. This string match happens to work for this fixture but is fragile — the test would silently break if the fixture were regenerated with different whitespace.

The actual warning here is the **missing O atom on the receptor** — a backbone without a carbonyl oxygen will cause `prepare_receptor4.py` to fail or produce a broken PDBQT in any integration test that doesn't mock the subprocess.

**Fix:** Add at minimum an `O` atom to the fixture for completeness:
```
ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O
```

---

### WR-05: `meeko_available` fixture does not guard all Meeko-dependent tests in `TestLigandBatch`

**File:** `tests/test_prep.py:960-1001`

**Issue:** `TestLigandBatch.test_batch_single_pose_success` correctly takes `meeko_available` as a parameter (line 963), which causes the test to skip if Meeko is absent. However, `test_batch_successes_plus_failures_equals_input` (line 987) runs `prepare_ligand_batch([pose_tiny, bad_path], ...)` — the `pose_tiny` path is valid and will be processed by the Meeko worker. If Meeko is not installed, this call will produce a `PoseFailure` for the valid pose too, making `len(successes) + len(failures) == 2` pass for the wrong reason (both are failures). The invariant holds numerically but obscures whether Meeko is functioning. The test will not skip and will not fail, but it silently gives a false-positive signal in an environment without Meeko.

**Fix:** Add `meeko_available` as a parameter to any test that expects Meeko to successfully process `pose_tiny`:
```python
def test_batch_successes_plus_failures_equals_input(
    self, tmp_path: Path, pose_tiny: Path, meeko_available  # add this
) -> None:
```

---

## Info

### IN-01: `_build_gpf` `maps_dir` parameter is unused

**File:** `src/hybridock_pep/prep/grids.py:82-125`

**Issue:** The docstring notes `maps_dir` is "included for signature consistency." The parameter is accepted but never used in the function body. This is benign but confusing — a reader may spend time looking for where it is used.

**Fix:** Either drop the parameter and update the call site, or add a `# noqa: ARG001` comment and a clearer docstring note:
```python
def _build_gpf(config: DockConfig, maps_dir: Path) -> str:  # noqa: ARG001 — unused, kept for API symmetry
```

---

### IN-02: `SA` atom type in `_RECEPTOR_TYPES` but not `_LIGAND_TYPES`

**File:** `src/hybridock_pep/prep/grids.py:14-15`

**Issue:** `_RECEPTOR_TYPES = "C A N O SA S H HD"` includes `SA` (aromatic sulfur). `_LIGAND_TYPES = "C A N O S H HD"` does not. The peptide ligand contains cysteine (`C` in the target peptide LISDAELEAIFEADC) whose sulfur should be typed as `SA` in some AD4 parameter sets. If cysteine sulfur is assigned `SA` by Meeko but `SA` is absent from `ligand_types`, autogrid4 will not produce a `receptor.SA.map`, and AD4 scoring will fall back or silently fail for poses with cysteine.

This may be intentional (the spec may assume `S` covers cysteine) but it is worth verifying against the AD4 parameter file used (`AD4_parameters.dat`).

**Fix:** Verify the Meeko atom type assignment for cysteine sulfur. If it assigns `SA`, add `SA` to `_LIGAND_TYPES` and add a corresponding `map receptor.SA.map` line in `_build_gpf`.

---

### IN-03: Broad `except Exception` in `_prepare_single_ligand` swallows unexpected errors silently

**File:** `src/hybridock_pep/prep/ligand.py:51-58`

**Issue:** The `except Exception as e` block (with `# noqa: BLE001`) intentionally catches all exceptions to prevent worker crashes from propagating. The comment explains this is by design. However, the error message formats as `f"{type(e).__name__}: {e}"` — if `e` has no string representation or raises during `str()`, the log will be unhelpful. This is a low-risk edge case, but `repr(e)` is more robust than `str(e)` for unexpected internal exceptions.

**Fix:**
```python
error_msg=f"{type(e).__name__}: {e!r}",
```

---

### IN-04: `prepare_receptor` does not log the pdbfixer pH value used

**File:** `src/hybridock_pep/prep/receptor.py:56`

**Issue:** `fixer.addMissingHydrogens(7.4)` hard-codes pH 7.4. This value is correct per the spec and CLAUDE.md, but it is not logged. If a user ever changes environments or the spec changes, there is no trace in the run log of what pH was used for protonation. Given that the project requires full provenance logging per CLAUDE.md §4 (Reproducibility), this should appear in the log.

**Fix:**
```python
_PROTONATION_PH = 7.4  # physiological pH per spec §D-01

# ... later:
logger.info("Running pdbfixer: addMissingHydrogens(pH=%.1f)", _PROTONATION_PH)
fixer.addMissingHydrogens(_PROTONATION_PH)
```

---

_Reviewed: 2026-04-20T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
