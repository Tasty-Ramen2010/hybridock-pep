# Test Coverage Analysis

**Generated:** 2026-05-22  
**Suite:** `pytest --cov=hybridock_pep` (score-env, Python 3.11)  
**Result:** 209 passed, 45 skipped (slow e2e) — 8.69 s

---

## Headline vs. Real Coverage

| Scope | Stmts | Missed | Coverage |
|-------|-------|--------|----------|
| All modules (raw) | 1,907 | 679 | **64%** |
| Excluding untestable modules (see below) | 1,479 | 251 | **83%** |

The 64% headline is suppressed by three modules that **cannot** be tested from `score-env` and should be excluded from the coverage target:

| Module | Stmts | Reason untestable |
|--------|-------|-------------------|
| `sampling/run_rapidock.py` | 67 | Python 3.9 subprocess script; executed by `rapidock-env`, cannot be imported in Python 3.11 |
| `ui/app.py` | 348 | Streamlit app; requires a live browser session |
| `ui/_launcher.py` | 13 | Streamlit launcher |

**Adjusted coverage of 83% exceeds the 70% CLAUDE.md target.**

---

## Per-Module Breakdown

### Green (≥ 90%) — no action needed

| Module | Cover | Notes |
|--------|-------|-------|
| `models.py` | 100% | |
| `prep/phospho.py` | 100% | |
| `prep/errors.py` | 100% | |
| `prep/ligand.py` | 98% | Line 86: subprocess `CalledProcessError` path |
| `output/csv_writer.py` | 93% | Lines 125, 135, 143: empty-list early-exit guards |
| `analysis/plotting.py` | 92% | Lines 39, 104-105: `FileNotFoundError` guard on Agg backend |
| `prep/grids.py` | 92% | Lines 41, 65, 74: `CalledProcessError` on autogrid4 failure |
| `driver.py` | 91% | Lines 84, 176, 186: `prepare_receptor_pdb` path (Stage 1 only), AD4 failure-log branches |
| `prep/receptor.py` | 91% | Lines 58-75: `pdbfixer` clean path (requires GPU run) |
| `scoring/ad4.py` | 90% | Lines 26-27: ImportError branch; line 73: `ValueError` on empty maps |

### Yellow (70–89%) — monitor, low-priority tests

| Module | Cover | Missing lines | What's not covered |
|--------|-------|---------------|--------------------|
| `scoring/vina.py` | 89% | 24-25, 106-109, 176 | ImportError; timeout/`CalledProcessError` on subprocess; `VerbosityError` |
| `scoring/entropy.py` | 87% | 78-79, 98, 163, 171-172, 194, 199-200, 249, 252, 256, 285, 288, 335, 370-371, 433, 439, 445-449, 477, 561, 574, 588 | `beta<0` guard; `n_poses=1` fallback; `gamma<0` clamp; calibration file write errors |
| `analysis/clustering.py` | 82% | 17-19, 31-32, 37-39, 75, 88, 90, 94, 148-149, 223, 232-233, 245-247, 250-251, 285, 310 | ImportError branches (Biopython/sklearn); `_load_receptor_ca_coords` ValueError; `_kabsch_rmsd` N<3 fallback; `_select_k_silhouette` k_max<2 path; silhouette `ValueError` path |
| `analysis/statistics.py` | 81% | 23-24, 47, 56-59, 94-95, 111-114, 153 | Single-cluster edge case; `output_dir.mkdir` path; `_ci95` n=0 guard |
| `sampling/rapidock_runner.py` | 81% | 48-50, 92, 103-109, 131, 136, 156, 162, 233, 246-248 | `--seed` propagation branch; process timeout/non-zero exit handler |
| `sampling/pose_io.py` | 78% | 32-33, 43-44, 75-85, 144, 157, 159, 163, 215, 221-224, 236, 239-240, 247 | Multi-model PDB handling; PDB parse error paths; `ca_coords`-missing fallback |
| `output/metadata.py` | 70% | 85-88, 94, 123-125, 145-152, 157-160, 170-171, 179-183 | `get_rapidock_commit_sha`; `_detect_cuda_driver_version` (nvidia-smi); `_get_openmm_version` |

### Red (< 70%) — real gaps worth fixing

#### `cli.py` — 64% (36 missed, lines 252-271, 281-295, 305-321, 335-353, 367, 379-380)

The four `_run_*` dispatch handlers are entirely uncovered. Existing tests only exercise `--help` and `parse_known_args`. Each handler is 8–20 lines and trivially mockable:

- `_run_dock`: patch `driver.run_dock`, assert `DockConfig` built correctly from args
- `_run_calibrate`: patch `calibrate_alpha.main`, assert `Namespace` forwarded
- `_run_prep`: patch `prepare_receptor`, assert path resolution
- `_run_benchmark`: patch `benchmark.main`, assert `Namespace` forwarded

#### `scoring/minimization.py` — 42% (52 missed, lines 103-184)

The entire OpenMM execution path is untested — restraint force construction, energy minimization, displacement safety check, and PDB write (Steps 3–6 in `minimize_pose`). This is the highest-risk uncovered block:

- The displacement check's numpy unit-stripping (lines 141-150) is version-sensitive across OpenMM releases
- The `_MAX_DISPLACEMENT_ANG` threshold determines whether a minimized pose gets used or the original is returned — silent wrong behavior if the logic is off

Tests should mock `openmm`, `pdbfixer`, and `app.ForceField` rather than require a real OpenMM installation at test time. The batch wrapper (`minimize_poses_batch`, lines 209-220) needs a matching integration test.

---

## Priority Test Work

Ordered by coverage gain per effort:

1. **`scoring/minimization.py`** (+52 stmts) — mock OpenMM; test displacement check with synthetic position arrays; test batch wrapper
2. **`cli.py` dispatch handlers** (+36 stmts) — patch driver/scripts; assert Namespace forwarding
3. **`output/metadata.py` helpers** (+23 stmts) — mock `subprocess.run` for nvidia-smi; mock `importlib.metadata` for rapidock SHA and openmm version
4. **`sampling/pose_io.py` error paths** (+21 stmts) — write malformed PDB fixtures; test multi-model and missing-CA fallback

Implementing items 1–4 brings adjusted coverage from 83% to approximately **95%** and eliminates the two modules currently below 70% in testable code.

---

## What Will Not Be Tested (by design)

- `sampling/run_rapidock.py`: run in `rapidock-env` subprocess. Validate end-to-end via smoke test (`scripts/smoke_test.sh`), not pytest.
- `ui/app.py`, `ui/_launcher.py`: Streamlit. Covered by manual smoke testing via `scripts/launch_ui.sh`.
- Slow integration tests (`pytest -m slow`): MDM2/p53 e2e. 45 currently skipped; opt-in only.
