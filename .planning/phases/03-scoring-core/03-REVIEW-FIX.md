---
phase: 03-scoring-core
fixed_at: 2026-04-21T00:00:00Z
fix_scope: critical_warning
findings_in_scope: 6
fixed: 6
skipped: 0
iteration: 1
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed:** 2026-04-21
**Scope:** Critical + Warning (Info excluded)
**Status:** all_fixed — 6/6 findings resolved, 30 tests passing

---

## Fixes Applied

### CR-01 — `vina.py`: Guard `pdbqt_path=None` before `check_grid_boundary`
**Commit:** `d061999`
Added an explicit `if pose.pdbqt_path is None: raise ValueError(...)` guard at the top of the pose
loop, before `check_grid_boundary` is called. The `ValueError` carries a diagnostic message pointing
to `prep/ligand.py` as the missing step. Caught by the existing broad `except Exception` with a
meaningful error log rather than an opaque `AttributeError`.

Also added `import os` and converted `_append_clipped_pose` to atomic write (CR-02 below — both
landed in `d061999` since they share the file).

### CR-02 — `vina.py`: Atomic write in `_append_clipped_pose`
**Commit:** `d061999` (same commit as CR-01)
Replaced `path.write_text(json.dumps(...))` with write-to-`.tmp`-then-`os.replace()`. `os.replace`
is atomic on POSIX (rename syscall) and overwrites the destination, preventing JSON truncation on
SIGTERM or disk-full mid-write.

### WR-01 — `entropy.py`: Detect L-BFGS-B non-convergence
**Commit:** `67e0018`
Added `if not result.success: _log.warning(...)` after `minimize()` returns. A failed fit now
surfaces as a WARNING with `result.message` rather than silently propagating out-of-bound parameters
to `write_calibration`.

### WR-02 — `entropy.py`: Wrap `KeyError` in `load_calibration`
**Commit:** `67e0018`
Wrapped `cal["alpha"]` / `cal["beta"]` in `try/except KeyError as exc: raise ValueError(...) from exc`.
The `ValueError` message quotes the missing key and instructs the user to re-run `calibrate_alpha.py`.
This aligns with the documented `Raises` contract.

### WR-03 — `calibrate_alpha.py`: `KeyError` → `ValueError` for missing `pdb_id`
**Commit:** `1927a71`
Changed `raise KeyError(...)` to `raise ValueError(...)` at line 147 of `calibrate_alpha.py`.
Now matches the docstring's `Raises: ValueError` contract so callers catching `ValueError` for bad
user input don't silently miss this case.

### WR-04 — `test_scoring.py`: Force `reader.fieldnames` before consuming rows
**Commit:** `ad99e72`
Added `_ = reader.fieldnames` before `rows = list(reader)` in `test_training_csv_schema`.
Prevents `TypeError: argument of type 'NoneType' is not iterable` on empty CSV inputs where
`fieldnames` stays `None` until the first row is consumed.

---

## Test Results

```
30 passed in 43.22s
```

All 30 scoring tests pass after fixes. No regressions.

---

## Info Findings (not in scope)

- **IN-01** (`entropy.py:30`): `_RT` constant derivation comment — deferred, Info only
- **IN-02** through **IN-04**: deferred per fix scope (critical_warning only)
