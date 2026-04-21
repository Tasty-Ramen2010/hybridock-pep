---
phase: 03-scoring-core
reviewed: 2026-04-21T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/hybridock_pep/scoring/vina.py
  - src/hybridock_pep/scoring/ad4.py
  - src/hybridock_pep/scoring/entropy.py
  - src/hybridock_pep/scoring/__init__.py
  - scripts/calibrate_alpha.py
  - tests/test_scoring.py
findings:
  critical: 2
  warning: 4
  info: 4
  total: 10
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-04-21T00:00:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

The six scoring-module files implement SCORE-01 (Vina batch scorer), SCORE-02 (AD4 batch scorer),
SCORE-03 (entropy correction + hybrid formula), and the calibration CLI. The overall structure is
solid: per-pose exception isolation (D-07), lazy Vina import, correct `load_maps` usage in the AD4
path, and the D-01 formula are all implemented correctly. Two critical issues require fixes before
merging: a `None`-guard gap on `pdbqt_path` that produces an opaque error and silent metadata
corruption, and a non-atomic JSON write that can corrupt `run_metadata.json` on an interrupted run.
Four warnings cover a silent optimization non-convergence, an unhandled `KeyError` in
`load_calibration`, a `KeyError`/`ValueError` contract mismatch in the calibration CLI, and a
fragile test that accesses `fieldnames` after exhausting the CSV reader.

---

## Critical Issues

### CR-01: `score_vina_batch` — `None` pdbqt_path produces opaque error and corrupts metadata JSON

**File:** `src/hybridock_pep/scoring/vina.py:158` and `vina.py:97`

**Issue:** `ScoredPose.pdbqt_path` is typed `Path | None` and defaults to `None`. Line 158 calls
`check_grid_boundary(pose.pdbqt_path, ...)` unconditionally; inside that function, line 53 calls
`pdbqt_path.read_text()` which raises `AttributeError: 'NoneType' object has no attribute
'read_text'`. This is swallowed by the broad `except Exception` at line 172 and logged as a
`PoseFailure` — containment is correct — but the error message is opaque rather than diagnostic.
More seriously, if the `is_clipped` path is somehow reached with `pdbqt_path=None`,
`_append_clipped_pose` at line 97 stores `str(None)` = `"None"` into the metadata JSON, silently
corrupting it.

**Fix:**
```python
for pose in poses:
    try:
        if pose.pdbqt_path is None:
            raise ValueError(
                f"Pose {pose.pose_idx} has pdbqt_path=None; "
                "was prep/ligand.py run before scoring?"
            )
        pose.is_clipped = check_grid_boundary(
            pose.pdbqt_path, config.site_coords, config.box_size
        )
        ...
```

---

### CR-02: `_append_clipped_pose` — non-atomic file write can corrupt `run_metadata.json`

**File:** `src/hybridock_pep/scoring/vina.py:100-101`

**Issue:** The function reads existing JSON, appends to it in memory, then calls
`path.write_text(json.dumps(data, indent=2))`. `write_text` opens the file for writing (truncating
it immediately) and then writes the content. If the process is interrupted between the truncate and
the write completing (SIGTERM, disk full, power loss), the metadata file is left empty or
truncated. All previously recorded clipped-pose entries are lost.

**Fix:** Write to a sibling `.tmp` file first, then rename atomically (rename is atomic on POSIX
and on NTFS):
```python
import os

path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2))
os.replace(tmp, path)  # atomic on POSIX; overwrites destination
```

---

## Warnings

### WR-01: `fit_calibration` — L-BFGS-B non-convergence is never detected; bad fit silently propagated

**File:** `src/hybridock_pep/scoring/entropy.py:200-201`

**Issue:** `scipy.optimize.minimize` returns a `result` object with `result.success: bool` and
`result.message: str`. The code extracts `result.x` without checking `result.success`. If the
optimizer fails to converge (degenerate training set, single data point with `n_residues=0`,
numerical issues), the parameters returned are whatever the optimizer stopped at — not the actual
minimum. The downstream `load_calibration()` self-check in `calibrate_alpha.py` only validates
the range, not convergence, so a failed fit can silently reach production.

**Fix:**
```python
result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
if not result.success:
    _log.warning(
        "L-BFGS-B optimization did not converge: %s. "
        "Proceeding with best-found parameters — verify calibration manually.",
        result.message,
    )
alpha, beta = float(result.x[0]), float(result.x[1])
```

---

### WR-02: `load_calibration` — bare `KeyError` if `alpha` or `beta` key absent; undocumented raise

**File:** `src/hybridock_pep/scoring/entropy.py:57-58`

**Issue:** `cal["alpha"]` and `cal["beta"]` raise `KeyError` if the keys are missing from the JSON
(hand-edited file, schema mismatch, or truncated write). The function docstring documents only
`ValueError` and `FileNotFoundError`. A `KeyError` with just the key name is an opaque error for
end users.

**Fix:**
```python
try:
    alpha = cal["alpha"]
    beta = cal["beta"]
except KeyError as exc:
    raise ValueError(
        f"Calibration file {path} is missing required key {exc}. "
        "Re-run calibrate_alpha.py to regenerate a valid calibration file."
    ) from exc
```

---

### WR-03: `calibrate_alpha.py` — `KeyError` raised where docstring promises `ValueError`

**File:** `scripts/calibrate_alpha.py:147-150`

**Issue:** The `main()` docstring at line 103 states:
`ValueError: If a pdb_id in the training CSV is missing from the scores JSON`.
The actual raise at line 147 is `KeyError`. Callers (and the post-write `try/except` in any future
driver code) catching `ValueError` for bad user input will miss this exception entirely.

**Fix:**
```python
raise ValueError(
    f"pdb_id '{pdb_id}' from training CSV not found in scores JSON. "
    f"Available pdb_ids: {sorted(scores.keys())}"
)
```

---

### WR-04: `test_scoring.py` — `reader.fieldnames` read after reader is exhausted; fails on empty CSV

**File:** `tests/test_scoring.py:613-616`

**Issue:**
```python
reader = csv.DictReader(fh)
rows = list(reader)          # exhausts the reader, but fieldnames already populated if rows > 0
...
assert list(reader.fieldnames) == expected_columns, ...
```
`csv.DictReader` populates `fieldnames` lazily on the first row. If the CSV has a header but zero
data rows, `list(reader)` returns `[]` and `fieldnames` remains `None` in some Python builds,
causing `TypeError: argument of type 'NoneType' is not iterable` instead of the expected
assertion failure. The test passes today only because the real CSV has data rows.

**Fix:** Access `fieldnames` before consuming the reader:
```python
reader = csv.DictReader(fh)
_ = reader.fieldnames   # force header read; safe even on empty files
rows = list(reader)
assert list(reader.fieldnames) == expected_columns, ...
```

---

## Info

### IN-01: `entropy.py` — `_RT` constant lacks derivation comment; auditing fragile

**File:** `src/hybridock_pep/scoring/entropy.py:30`

**Issue:** `_RT = 0.592` is correct (R=1.987×10⁻³ kcal/mol/K × 298.15 K ≈ 0.5922) but the
rounding and the temperature assumption are not documented inline. A future reader or auditor
cannot verify the value without external lookup.

**Fix:**
```python
# R = 1.987e-3 kcal/mol/K, T = 298.15 K → RT ≈ 0.5922 kcal/mol (D-09, hardcoded v1)
_RT = 0.592
```

---

### IN-02: `vina.py` — `OSError` on metadata read silently drops previous clipped-pose entries

**File:** `src/hybridock_pep/scoring/vina.py:92-94`

**Issue:**
```python
except (json.JSONDecodeError, OSError):
    data = {}
```
A permission-denied or disk-full `OSError` on `path.read_text()` silently resets `data` to `{}`,
discarding all previously recorded clipped poses. No warning is emitted.

**Fix:**
```python
except (json.JSONDecodeError, OSError) as e:
    _log.warning(
        "Could not read existing metadata at %s (%s); starting fresh — "
        "previous clipped-pose entries may be lost.",
        path, e,
    )
    data = {}
```

---

### IN-03: `test_scoring.py` — `test_pkd_to_delta_g_conversion` uses `n_residues_list=[0]`; alpha is unconstrained and the ΔG formula is not verified

**File:** `tests/test_scoring.py:579-592`

**Issue:** With `n_residues=0`, `alpha * 0` vanishes and alpha is unconstrained by the objective.
L-BFGS-B will pin it to the lower bound (0.2). The test only asserts that `"alpha"` and `"beta"`
keys exist in the result — not that the pKd→ΔG conversion `ΔG = -RT * pKd` is correct. The test
name implies it validates the conversion but it does not.

**Suggestion:** Add an assertion on the hybrid score or use a non-degenerate `n_residues` so the
formula is exercised. For example, with `vina=ad4=ΔG` and `n_residues=0`, assert
`result["pearson_r"]` is NaN (single point):
```python
import math
assert math.isnan(result["pearson_r"]), "Single-point fit should yield NaN Pearson r"
```

---

### IN-04: `calibrate_alpha.py` — relative `Path` defaults resolve against caller's cwd, not repo root

**File:** `scripts/calibrate_alpha.py:57` and `scripts/calibrate_alpha.py:79`

**Issue:** `default=Path("data/training_complexes.csv")` and `default=Path("data/calibration.json")`
are relative paths that resolve against whatever directory the user is in when they invoke the
script. Invocations from a job scheduler, a CI runner, or a subdirectory will silently use the
wrong path and fail with a confusing `FileNotFoundError`.

**Fix:** Anchor defaults to the repo root:
```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
...
default=_REPO_ROOT / "data" / "training_complexes.csv",
...
default=_REPO_ROOT / "data" / "calibration.json",
```
At minimum, update the help text to say "relative to cwd".

---

_Reviewed: 2026-04-21T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
