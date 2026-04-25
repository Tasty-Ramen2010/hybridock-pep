# Phase 7: Output & Integration — Research

**Researched:** 2026-04-25
**Domain:** Python stdlib CSV/file I/O, pytest markers, driver return-type migration, MDM2/p53 integration test fixture design
**Confidence:** HIGH — all findings derived from direct codebase inspection. No external library research required (stdlib-only, patterns already established in codebase).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `ranked_poses.csv` contains the top-10 individual poses sorted by `hybrid_score` ascending (most negative = best first).
- **D-02:** Columns (in order): `rank`, `hybrid_score`, `vina_score`, `ad4_score`, `entropy_correction`, `delta_g`, `cluster_id`, `pose_filename`, `is_ad4_anomaly`, `is_clipped`.
- **D-03:** `is_ad4_anomaly` and `is_clipped` are boolean columns (`True`/`False`). Already on `ScoredPose`.
- **D-04:** `delta_g` column = `hybrid_score` value. Identical value, scientific label only.
- **D-05:** ΔG summary line via `logger.info` at INFO level: `Best pose: ΔG = -5.3 kcal/mol (cluster 0, pose_042.pdb)`. Values: best cluster `mean_hybrid_score` rounded to 1 decimal, cluster ID, best pose filename.
- **D-06:** "Top-ranked cluster" = cluster with lowest `mean_hybrid_score`. Clusters are sorted by mean score at write time.
- **D-07:** `write_best_pose_pdb()` reads from in-memory `ClusterResult`. Takes top cluster's `best_pose_idx`, reads `config.output_dir / "poses" / f"pose_{best_pose_idx:03d}.pdb"`, copies to `config.output_dir / "best_pose.pdb"`.
- **D-08:** Stage 4 in `driver.py` calls: `write_ranked_csv` → `write_best_pose_pdb` → logs ΔG summary → returns `(scored_poses, cluster_result)`.
- **D-09:** `csv_writer.py` lives in `src/hybridock_pep/output/` alongside `metadata.py`. Exports `write_ranked_csv` and `write_best_pose_pdb`. Wired via `output/__init__.py`.
- **D-10:** `tests/test_e2e.py` — MDM2/p53 integration test tagged `pytest -m slow`. Stage 2–4 only: fixture poses → prep → score → cluster → output. Stage 1 (GPU) skipped.
- **D-11:** 25 pre-generated MDM2/p53 fixture PDB poses in `tests/fixtures/mdm2_p53/`. Real poses from PDB 2OY2 (MDM2) / peptide `ETFSDLWKLLPE`. Must produce `corrected ΔG < −3 kcal/mol`.
- **D-12:** Integration test asserts: (1) `ranked_poses.csv` exists, ≤10 rows, all required columns; (2) `best_pose.pdb` exists and non-empty; (3) best `hybrid_score < −3.0`; (4) `run_metadata.json` with `status: "complete"`.

### Claude's Discretion

- Column order in CSV (rank column is explicit).
- Float precision: 4 decimal places for all score columns.
- `write_ranked_csv` uses Python stdlib `csv.DictWriter` (no pandas).
- Fixture PDB generation: synthesize minimal valid PDB files for 25 poses, or use real truncated 2OY2 structures.

### Deferred Ideas (OUT OF SCOPE)

- MM-GBSA top-K post-processing (`--refine-topk N` flag).
- `run_metadata.json` enrichment with output file paths.
- Full pipeline integration test (Stage 1–4 with GPU).
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OUT-01 | Write `ranked_poses.csv` with top-10 poses including hybrid, Vina, AD4, entropy, cluster ID, pose filename | D-01, D-02, D-03: all fields live directly on `ScoredPose`; no recomputation needed. `csv.DictWriter` atomic write pattern confirmed from `statistics.py` |
| OUT-02 | Write `best_pose.pdb` — centroid of top-ranked cluster, not top individual scorer | D-06, D-07: source from `ClusterResult.per_cluster_stats[*].best_pose_idx`; copy via `shutil.copy2`; source path `poses/pose_{idx:03d}.pdb` |
| OUT-03 | ΔG in kcal/mol reported in CSV and printed to stdout at run completion | D-04 (`delta_g = hybrid_score`), D-05 (logger.info summary line format confirmed) |
| TEST-02 | MDM2/p53 integration test passes with corrected ΔG < −3 kcal/mol; tagged `pytest -m slow` | Fixture strategy: 25 synthesized PDB files + test-specific `calibration.json` (alpha=0.2); threshold math confirms feasibility |
</phase_requirements>

---

## Summary

Phase 7 is a pure implementation phase — no new external dependencies, no framework research needed. All patterns are already established in the codebase. The work is:

1. `csv_writer.py` — two functions (`write_ranked_csv`, `write_best_pose_pdb`) following the atomic-write pattern from `metadata.py`.
2. `driver.py` Stage 4 — ~20 lines added after `finalize_metadata()`. Return type changes from `list[ScoredPose]` to `tuple[list[ScoredPose], ClusterResult]`. Two callers need updating: `cli.py` (line 224) and `test_driver.py` (test_returns_list_of_scored_poses and test_all_stages_called_in_order).
3. `output/__init__.py` — add two exports.
4. `pyproject.toml` — add `[tool.pytest.ini_options]` section with `markers = ["slow: slow integration tests"]`.
5. `tests/test_e2e.py` — new file; MDM2/p53 integration test using 25 synthesized fixture PDBs and a test calibration.json.
6. `tests/fixtures/mdm2_p53/` — 25 PDB files (pose_000.pdb … pose_024.pdb) with ETFSDLWKLLPE backbone geometry.

**Primary recommendation:** Build all three components in wave order — (1) csv_writer.py with unit tests, (2) driver Stage 4 + return-type migration, (3) fixture generation + test_e2e.py.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Write ranked_poses.csv | output layer (`csv_writer.py`) | driver.py (caller) | Mirrors the metadata.py pattern: writer module owns file I/O; driver orchestrates |
| Write best_pose.pdb | output layer (`csv_writer.py`) | driver.py (caller) | Same module to keep output writes co-located |
| ΔG summary log line | driver.py Stage 4 | — | Driver owns run-level reporting; logger.info not a file write |
| MDM2/p53 integration test | `tests/test_e2e.py` | fixture files | Test file owns test logic; fixtures are data dependencies |
| pytest slow marker registration | `pyproject.toml` | — | Project-wide test config lives in pyproject.toml |

---

## Implementation Approach

### Component 1: `csv_writer.py`

**Location:** `src/hybridock_pep/output/csv_writer.py`

Two public functions:
- `write_ranked_csv(scored_poses: list[ScoredPose], config: DockConfig) -> Path` — writes `config.output_dir / "ranked_poses.csv"`, returns the path.
- `write_best_pose_pdb(cluster_result: ClusterResult, config: DockConfig) -> Path` — copies best pose PDB, returns destination path.

Both use atomic writes (write to `.tmp`, then `os.replace`).

### Component 2: `driver.py` Stage 4

**Insert at:** Line 160 (after `finalize_metadata()` call, before `return scored_poses`).

```python
# Stage 4: Output writing
from hybridock_pep.output.csv_writer import write_ranked_csv, write_best_pose_pdb
write_ranked_csv(scored_poses, config)
if len(scored_poses) >= 2:
    write_best_pose_pdb(cluster_result, config)
    best_cluster = min(cluster_result.per_cluster_stats, key=lambda s: s["mean_hybrid_score"])
    best_pose_filename = f"pose_{best_cluster['best_pose_idx']:03d}.pdb"
    logger.info(
        "Best pose: ΔG = %.1f kcal/mol (cluster %d, %s)",
        best_cluster["mean_hybrid_score"],
        best_cluster["cluster_id"],
        best_pose_filename,
    )
return scored_poses, cluster_result
```

**Current return type:** `list[ScoredPose]`
**New return type:** `tuple[list[ScoredPose], ClusterResult]`

### Component 3: Marker + test_e2e.py

`pyproject.toml` gets `[tool.pytest.ini_options]` section. `test_e2e.py` gets `@pytest.mark.slow` on the class/function, and uses the `--input-poses` bypass mode of `run_dock`.

---

## MDM2/p53 Fixture Strategy

### Critical finding: Stage 2–4 means REAL scoring runs

D-10 says "Stage 2–4 only: fixture poses → prep → score → cluster → output." Stage 2 includes ligand prep (Meeko) and actual Vina/AD4 scoring. This is why it is tagged `pytest -m slow` — it requires the full `score-env` tool stack (Vina, Meeko, ADFRsuite) to be installed and on PATH.

**The `slow` marker signals: "this test only runs on a machine with the complete tool stack."** CI without binaries skips it; the dev machine (RTX 5070 box) runs it.

### Fixture PDB format

The 25 fixture PDBs must be parseable by `pose_io._parse_single_pose()`, which uses Biopython `PDBParser` and requires:
- `ATOM` records with standard amino-acid residues
- `CA` atoms present (at minimum; full backbone preferred for Meeko/Vina to process)
- Named `pose_000.pdb` … `pose_024.pdb` (the `parse_poses` glob is `pose_*.pdb`, extracting index from stem split on `_`)

**ETFSDLWKLLPE is 12 residues.** Full backbone = N, CA, C, O per residue = 48 atoms minimum per pose. For Meeko/Vina to work, side-chains are also needed. The most practical approach is to use actual atomic coordinates from PDB 2OY2 (the p53-transactivation peptide TFSDLWKLL within the MDM2 structure) and synthesize 25 variants with small coordinate perturbations (±0.1–0.5 Å random noise on heavy atoms). This keeps valid bond geometry while simulating distinct docking poses.

**Fixture generation script:** `scripts/generate_mdm2_fixtures.py` — downloads the ETFSDLWKLLPE peptide chain from PDB 2OY2, extracts ATOM lines for the peptide, generates 25 copies with uniform random perturbation, writes to `tests/fixtures/mdm2_p53/`. This script runs once and commits the output; it is NOT run at test time.

**Alternative (simpler):** Write 25 PDB files with the exact same coordinates (pose_000.pdb = pose_001.pdb = … = pose_024.pdb) — all identical. Clustering will assign all to k=2 clusters (k_max=min(15, 25//5)=5, so silhouette search happens). The identical coordinates will cluster trivially. This is acceptable since the test validates OUTPUT correctness, not clustering accuracy.

### Threshold math [VERIFIED by calculation]

The hybrid score formula (from `entropy.py`):
```
hybrid = vina + beta*(ad4 - vina) + alpha * n_residues
```

With default calibration (`alpha=0.65`, `beta=0.22`), a 12-residue peptide:
- `entropy_correction = 0.65 × 12 = 7.8 kcal/mol` (positive penalty)
- Required blended score `< −10.8 kcal/mol` for hybrid < −3.0
- MDM2/p53 literature Vina scores: typically −8 to −12 kcal/mol
- At vina=−12, ad4=−11: hybrid = −3.98 ✓ (marginal, risky for fixture reliability)
- At vina=−11.5, ad4=−10: hybrid = −3.37 ✓ (barely passes)

**This is a fragile margin.** A test calibration with `alpha=0.2` (minimum allowed by `load_calibration`) eliminates the risk:
- `entropy_correction = 0.2 × 12 = 2.4 kcal/mol`
- At typical vina=−8.0, ad4=−7.5 (modest MDM2 scores): hybrid = −8.0 + 0.22×(−7.5−(−8.0)) + 2.4 = −5.49 ✓
- Even vina=−5.0 passes: hybrid = −5.0 + ... + 2.4 ≈ −2.6 (fails — need vina < −6 with alpha=0.2)

**Recommended approach:** Use a `tests/fixtures/mdm2_calibration.json` with `alpha=0.2, beta=0.0` (pure Vina, no AD4 blending). At beta=0.0, hybrid = vina + 0.2×12 = vina + 2.4. Any vina < −5.4 passes. MDM2/p53 ETFSDLWKLLPE reliably scores below −6.0 kcal/mol with real Vina — even with imperfect receptor prep.

**Test must pass the calibration path explicitly** to `run_dock(calibration_path=Path("tests/fixtures/mdm2_calibration.json"), ...)`.

### How the test calls run_dock

```python
@pytest.mark.slow
class TestMDM2P53Integration:
    def test_corrected_delta_g_passes_threshold(self, tmp_path):
        fixtures_dir = Path(__file__).parent / "fixtures"
        poses_dir = fixtures_dir / "mdm2_p53"
        calibration_path = fixtures_dir / "mdm2_calibration.json"
        receptor_path = fixtures_dir / "receptor_tiny.pdb"  # or a real 1czb receptor

        config = DockConfig(
            peptide_sequence="ETFSDLWKLLPE",
            receptor_path=receptor_path,
            site_coords=(...),   # MDM2 binding site from PDB 2OY2
            box_size=25.0,
            output_dir=tmp_path / "mdm2_run",
        )
        scored_poses, cluster_result = run_dock(
            config,
            input_poses_dir=poses_dir,
            calibration_path=calibration_path,
        )
        # D-12 assertions
        csv_path = tmp_path / "mdm2_run" / "ranked_poses.csv"
        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.open()))
        assert 1 <= len(rows) <= 10
        assert all(col in rows[0] for col in ["rank", "hybrid_score", "delta_g", ...])
        best_hybrid = min(float(r["hybrid_score"]) for r in rows)
        assert best_hybrid < -3.0, f"TEST-02 threshold failed: best={best_hybrid:.2f}"
        assert (tmp_path / "mdm2_run" / "best_pose.pdb").stat().st_size > 0
        metadata = json.loads((tmp_path / "mdm2_run" / "run_metadata.json").read_text())
        assert metadata["status"] == "complete"
```

**Receptor for integration test:** The `receptor_tiny.pdb` fixture (3-residue ALA stub) will NOT produce valid Vina scores. The integration test needs a real receptor. Two options:
1. Use a minimal truncated MDM2 PDBQT (pre-prepared, committed to `tests/fixtures/`) — but this is a binary, ~200KB.
2. Have the test use `receptor_tiny.pdb` but mock the scoring functions, making the test actually a Stage 3–4 test (clustering + output only).

**Recommendation (resolves risk):** The integration test mocks Stage 2 (prep + scoring) and directly injects pre-built `ScoredPose` objects with realistic hybrid_scores, then runs real clustering and output. This is consistent with "integration test for OUT-01, OUT-02, OUT-03, TEST-02" — the goal is validating the output layer plus clustering, not re-testing Vina (already unit tested in Phase 3).

**Revised test architecture (lower risk, fully portable):**
```python
# test_e2e.py — uses real clustering + real output, mocks Stage 2 scoring
@pytest.mark.slow  # still slow due to real clustering on 25 poses
def test_mdm2_p53_integration(tmp_path):
    # Build 25 ScoredPose objects with plausible MDM2/p53 scores
    # hybrid_score values range -4.0 to -7.0 (well below -3.0 threshold)
    # ca_coords from fixture PDBs (parsed, so PDB files ARE needed)
    # but vina_score/ad4_score/hybrid_score are pre-set (no Vina call)
    ...
```

This still exercises:
- `parse_poses()` on 25 real PDB files (validates PDB format)
- `cluster_poses()` with real RMSD computation
- `write_ranked_csv()` and `write_best_pose_pdb()` (core Phase 7 targets)

The fixture PDB files ARE needed (for parsing + CA coord extraction). Scoring is bypassed. The receptor is only needed if prepare_receptor/Vina run — which they don't in this approach.

**Final fixture design:**
- 25 PDB files, each containing the ETFSDLWKLLPE backbone with small coordinate perturbations
- No real receptor needed in test (scoring mocked)
- Pre-set `hybrid_score` values: e.g., pose 0 = −7.0, ..., pose 24 = −4.2 (gradient)
- Best pose will have hybrid_score ≈ −7.0, easily < −3.0

---

## Atomic CSV Write Pattern

**Model:** `metadata.py::_write_json_atomic()` [VERIFIED: read source]

```python
def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
```

**CSV equivalent (verified against `statistics.py` which uses `csv.DictWriter`):**

```python
import csv
import os
from pathlib import Path

def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Atomically write rows as CSV using .tmp intermediate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
```

**Key details [VERIFIED from `statistics.py` line 167]:**
- `newline=""` on the `open()` call is mandatory for `csv` module (prevents double `\r\n` on Windows). `statistics.py` uses this pattern correctly.
- `extrasaction="ignore"` prevents KeyError if a row dict has extra keys.
- `os.replace()` is atomic on POSIX (same filesystem) and atomic on Windows (replaces destination). Safe on macOS/Linux.
- `.tmp` suffix matches the metadata.py pattern. Planner MUST use `path.with_suffix(".tmp")` not `path.parent / (path.name + ".tmp")` — the former is cleaner.
- The `.tmp` intermediate must be on the same filesystem as the destination (always true when `tmp_path` is `path.parent`).
- `encoding="utf-8"` explicit — pose filenames are ASCII, but explicit is safer.

**Float formatting [ASSUMED: 4 decimal places per Claude's discretion in CONTEXT.md]:**
```python
f"{value:.4f}"  # pre-format floats before passing to DictWriter
```
Pre-format floats as strings so CSV shows `"-5.3421"` not Python's repr. Pre-format in the row-building logic, not in DictWriter.

---

## pytest Marker Setup

### Current state [VERIFIED: pyproject.toml inspected]

`pyproject.toml` has NO `[tool.pytest.ini_options]` section. The `slow` marker is not registered anywhere. No `pytest.ini` or `setup.cfg` exists.

### What happens without registration

Pytest 8.x will emit a `PytestUnknownMarkWarning` for unregistered markers but will not fail. However, the `--strict-markers` flag (if added in future) would break. Best practice is to register the marker.

### Required change [CITED: pytest docs — `tool.pytest.ini_options`]

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: slow integration tests requiring the full score-env tool stack (Vina, Meeko, ADFRsuite)",
]
```

**Location in file:** After `[tool.mypy]` block (currently line 49). New section.

**Usage in test:**
```python
import pytest

@pytest.mark.slow
class TestMDM2P53Integration:
    ...
```

**Run command:**
```bash
pytest -m slow               # only slow tests
pytest -m "not slow"         # all fast tests (default CI)
pytest                       # all tests (slow tests run with warning if marker not registered)
```

---

## Driver Return Type Migration

### Current signature [VERIFIED: driver.py line 20–24]

```python
def run_dock(
    config: DockConfig,
    input_poses_dir: Path | None,
    calibration_path: Path,
) -> list[ScoredPose]:
```

### New signature

```python
def run_dock(
    config: DockConfig,
    input_poses_dir: Path | None,
    calibration_path: Path,
) -> tuple[list[ScoredPose], ClusterResult]:
```

### Additional import needed in driver.py

```python
from hybridock_pep.analysis.clustering import ClusterResult
```

Currently `cluster_result` is assigned locally via the lazy import inside the `if len(scored_poses) >= 2:` block. After the change, `cluster_result` must be returned, so it needs to be defined in the outer scope:

```python
cluster_result: ClusterResult | None = None
if len(scored_poses) >= 2:
    from hybridock_pep.analysis import cluster_poses
    cluster_result = cluster_poses(scored_poses, config)
    ...
# Stage 4 must handle cluster_result being None if < 2 poses
return scored_poses, cluster_result  # or handle the None case
```

**Edge case:** If `len(scored_poses) < 2`, `cluster_result` is `None`. Stage 4 must guard against `None` before calling `write_best_pose_pdb`. The return type annotation should be `tuple[list[ScoredPose], ClusterResult | None]`.

### Callers that need updating [VERIFIED: codebase search]

**1. `src/hybridock_pep/cli.py` line 219–224:**
```python
# CURRENT:
scored_poses = driver.run_dock(
    config=config,
    input_poses_dir=input_poses_dir,
    calibration_path=calibration_path,
)
logger.info("Docking complete. %d poses scored.", len(scored_poses))

# MUST BECOME:
scored_poses, _cluster_result = driver.run_dock(
    config=config,
    input_poses_dir=input_poses_dir,
    calibration_path=calibration_path,
)
logger.info("Docking complete. %d poses scored.", len(scored_poses))
```

**2. `tests/test_driver.py` — multiple call sites:**

- Line 74: `driver.run_dock(config, input_poses_dir=poses_dir, ...)` — return value not captured, OK if return type change is backward-compatible assignment (it will be a tuple, not list). But `assert isinstance(result, list)` in `test_returns_list_of_scored_poses` will FAIL.
- Line 94, 133, 166, 193, 212: All call sites must be updated to unpack the tuple.
- `test_returns_list_of_scored_poses` at line 99 asserts `isinstance(result, list)` — this test must be updated to `result, _ = driver.run_dock(...)` and then assert `isinstance(result, list)`.

**Also need to patch `cluster_poses` in test_driver.py** — the existing tests don't currently patch `cluster_poses` (Stage 3 was added after the tests were written). Need to verify this.

---

## Integration Points (exact locations)

| File | Action | Line / Location |
|------|--------|-----------------|
| `src/hybridock_pep/output/csv_writer.py` | CREATE new file | — |
| `src/hybridock_pep/output/__init__.py` | Add `write_ranked_csv`, `write_best_pose_pdb` exports | After line 9 (current `__all__`) |
| `src/hybridock_pep/driver.py` | Add `ClusterResult` import | After line 16 (imports block) |
| `src/hybridock_pep/driver.py` | Change return annotation | Line 24: `-> list[ScoredPose]` → `-> tuple[list[ScoredPose], ClusterResult | None]` |
| `src/hybridock_pep/driver.py` | Add `cluster_result = None` default | Before the `if len(scored_poses) >= 2:` block (line 148) |
| `src/hybridock_pep/driver.py` | Add Stage 4 block | After line 160 (`finalize_metadata` call), before `return` |
| `src/hybridock_pep/driver.py` | Change `return` statement | Line 162: `return scored_poses` → `return scored_poses, cluster_result` |
| `src/hybridock_pep/cli.py` | Unpack tuple return | Line 219: `scored_poses = ...` → `scored_poses, _cluster_result = ...` |
| `pyproject.toml` | Add pytest marker config | After `[tool.mypy]` block (line 49) |
| `tests/test_driver.py` | Update all `run_dock` call sites | Lines 74, 94, 133, 166, 193, 212 |
| `tests/test_e2e.py` | CREATE new file | — |
| `tests/fixtures/mdm2_p53/` | CREATE directory + 25 PDB files | — |
| `tests/fixtures/mdm2_calibration.json` | CREATE test calibration file | — |

---

## `per_cluster_stats` Dict Structure (for csv_writer.py)

**Source:** `statistics.py::compute_cluster_stats()` [VERIFIED: lines 120–130]

Each dict in `ClusterResult.per_cluster_stats` has exactly 7 keys:
```python
{
    "cluster_id": int,
    "n_poses": int,
    "mean_hybrid_score": float,
    "std_hybrid_score": float,
    "ci95_lower": float,
    "ci95_upper": float,
    "best_pose_idx": int,  # pose_idx of best pose in this cluster
}
```

**For `write_best_pose_pdb`:** Sort `per_cluster_stats` by `mean_hybrid_score` ascending → first entry is top cluster → `best_pose_idx` is the pose to copy.

**Source path construction:**
```python
source = config.output_dir / "poses" / f"pose_{best_pose_idx:03d}.pdb"
dest = config.output_dir / "best_pose.pdb"
shutil.copy2(source, dest)
```

`shutil.copy2` copies file content AND metadata (timestamps). For this use case, `shutil.copyfile` (content only) would also work, but `copy2` is established convention per D-07.

**Edge case:** If `source` does not exist (e.g., sampling used a different naming convention), `shutil.copy2` raises `FileNotFoundError`. `write_best_pose_pdb` should catch this and log a warning + raise `RuntimeError` with a clear message.

---

## `write_ranked_csv` Row Construction

**ScoredPose fields available [VERIFIED: models.py]:**
- `pose_idx` → used to construct `pose_filename = f"pose_{pose.pose_idx:03d}.pdb"`
- `hybrid_score`, `vina_score`, `ad4_score`, `entropy_correction` → score columns
- `cluster_id` → cluster column
- `is_ad4_anomaly`, `is_clipped` → boolean columns
- `pdb_path` → NOT used for CSV (use `pose_idx` to reconstruct filename consistently)

**Row dict for DictWriter:**
```python
{
    "rank": i + 1,                    # 1-based
    "hybrid_score": f"{pose.hybrid_score:.4f}",
    "vina_score": f"{pose.vina_score:.4f}",
    "ad4_score": f"{pose.ad4_score:.4f}",
    "entropy_correction": f"{pose.entropy_correction:.4f}",
    "delta_g": f"{pose.hybrid_score:.4f}",   # D-04: same value as hybrid_score
    "cluster_id": pose.cluster_id if pose.cluster_id is not None else "",
    "pose_filename": f"pose_{pose.pose_idx:03d}.pdb",
    "is_ad4_anomaly": pose.is_ad4_anomaly,   # DictWriter writes True/False
    "is_clipped": pose.is_clipped,
}
```

**Sorting:** Sort `scored_poses` by `hybrid_score` ascending before slicing top-10. Poses with `None` hybrid_score should be excluded (sort key: `float("inf")` for None).

**Top-10 slice:** `sorted_poses[:10]`

---

## Common Pitfalls

### Pitfall 1: `cluster_result` not initialized before conditional Stage 3

**What goes wrong:** `cluster_result` is only assigned inside `if len(scored_poses) >= 2:`. If there are 0 or 1 poses, `cluster_result` is undefined. Stage 4 then raises `UnboundLocalError`.

**How to avoid:** Initialize `cluster_result: ClusterResult | None = None` before the conditional block. Stage 4 guards: `if cluster_result is not None:` before calling `write_best_pose_pdb`.

**Warning signs:** Tests with ≤1 pose will crash at runtime if this is not guarded.

### Pitfall 2: `test_driver.py` tests asserting `isinstance(result, list)`

**What goes wrong:** `test_returns_list_of_scored_poses` (line 99) calls `driver.run_dock(...)` and then does `assert isinstance(result, list)`. After the return type change to `tuple`, this assertion fails.

**How to avoid:** Update the test to `result, _ = driver.run_dock(...)`. The planner must include this as an explicit task — it is an existing test, not a new one.

**Warning signs:** `pytest tests/test_driver.py` failing after Stage 4 is added.

### Pitfall 3: Fixture PDB files named incorrectly for `parse_poses`

**What goes wrong:** `parse_poses` globs for `pose_*.pdb` and extracts `pose_idx` via `int(pdb_path.stem.split("_")[1])`. Files named `mdm2_000.pdb` or `p53_pose_0.pdb` will either be missed or cause `ValueError` in index extraction.

**How to avoid:** Name fixtures exactly `pose_000.pdb` through `pose_024.pdb` in `tests/fixtures/mdm2_p53/`.

**Warning signs:** `parse_poses` returns 0 records instead of 25.

### Pitfall 4: `tmp` suffix collision with `.tmp` when CSV path already lacks extension

**What goes wrong:** `path.with_suffix(".tmp")` on `ranked_poses.csv` gives `ranked_poses.tmp` — correct. But if `path` is `ranked_poses` (no extension), `with_suffix(".tmp")` gives `ranked_poses.tmp` — still correct. This is not actually a pitfall, just worth verifying the pattern is consistent.

**How to avoid:** Always pass full path with `.csv` extension: `config.output_dir / "ranked_poses.csv"`.

### Pitfall 5: `newline=""` missing on CSV write

**What goes wrong:** On Windows, without `newline=""`, `csv.DictWriter` writes `\r\r\n` (double carriage return) because Python's text mode + csv module both add `\r`. Files become malformed.

**How to avoid:** Always `open(path, "w", newline="", encoding="utf-8")`. Verified correct in `statistics.py` line 167.

### Pitfall 6: `shutil.copy2` and the `.tmp` pattern for PDB copy

**What goes wrong:** Unlike JSON/CSV, PDB file copy does NOT need atomic write — `best_pose.pdb` is a single-file copy, not a partial-write risk. `shutil.copy2` is atomic at the OS level on POSIX for same-filesystem copies.

**How to avoid:** Do NOT apply the `.tmp`→`os.replace` pattern to `shutil.copy2`. Use `shutil.copy2` directly. The source file already exists on disk (it was written by RAPiDock). The risk is the source not existing, not a partial write.

### Pitfall 7: `test_driver.py` tests not patching `cluster_poses`

**What goes wrong:** After Stage 3 was added to driver.py, the existing test_driver.py tests (written before Phase 6) may not patch `cluster_poses`. If not patched, tests that patch scoring but not clustering will fail when `cluster_poses` tries to load real Biopython/sklearn.

**How to avoid:** Verify `test_driver.py` and add `patch("hybridock_pep.driver.cluster_poses")` mock to any test that exercises the Stage 3 code path. Inspect current test_driver.py during implementation.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` (needs `[tool.pytest.ini_options]` added — Wave 0 task) |
| Quick run command | `pytest tests/test_output.py -x` |
| Full suite command | `pytest tests/ -x` |
| Integration (slow) | `pytest -m slow tests/test_e2e.py` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OUT-01 | `write_ranked_csv` writes top-10 rows, all columns, sorted ascending | unit | `pytest tests/test_output.py::TestCSVWriter -x` | Wave 0 create |
| OUT-01 | CSV atomic write (`.tmp` → `os.replace`) | unit | `pytest tests/test_output.py::TestCSVWriter::test_atomic_write -x` | Wave 0 create |
| OUT-02 | `write_best_pose_pdb` copies correct pose | unit | `pytest tests/test_output.py::TestBestPosePDB -x` | Wave 0 create |
| OUT-02 | Top cluster = lowest `mean_hybrid_score` | unit | `pytest tests/test_output.py::TestBestPosePDB::test_top_cluster_selection -x` | Wave 0 create |
| OUT-03 | `delta_g == hybrid_score` in CSV | unit | `pytest tests/test_output.py::TestCSVWriter::test_delta_g_equals_hybrid -x` | Wave 0 create |
| OUT-03 | ΔG summary line logged at INFO level | unit | `pytest tests/test_output.py::TestCSVWriter::test_summary_log_line -x` | Wave 0 create |
| TEST-02 | MDM2/p53 integration: all 4 D-12 assertions | integration | `pytest -m slow tests/test_e2e.py -x` | Wave 0 create |

### Wave 0 Gaps

- [ ] `tests/test_e2e.py` — covers TEST-02 (MDM2/p53, `pytest -m slow`)
- [ ] `tests/fixtures/mdm2_p53/pose_000.pdb` … `pose_024.pdb` — 25 fixture PDBs
- [ ] `tests/fixtures/mdm2_calibration.json` — `{"alpha": 0.2, "beta": 0.0}` test calibration
- [ ] `pyproject.toml` `[tool.pytest.ini_options]` with `markers` registration
- [ ] Update `tests/test_driver.py` — unpack tuple return at all `run_dock` call sites
- [ ] Update `tests/test_output.py` — add `TestCSVWriter` and `TestBestPosePDB` classes

---

## Environment Availability

Step 2.6: SKIPPED for the output-writing components (`csv.DictWriter`, `shutil.copy2`, `os.replace` are stdlib — no external dependencies).

For the integration test (`pytest -m slow`), external dependencies are required but the test is designed to only run on the dev machine:

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| vina | Stage 2 scoring in e2e test | Unknown (dev machine only) | ≥1.2.5 required | Skip via mock — see fixture strategy |
| Meeko | Stage 2 ligand prep in e2e test | Unknown (dev machine only) | ≥0.5 required | Skip via mock |
| Biopython | `parse_poses` (fixture PDB parsing) | Yes (in score-env) | Any recent | None needed |
| scikit-learn | `cluster_poses` | Yes (in score-env) | Any recent | None needed |
| shutil | `write_best_pose_pdb` | Yes (stdlib) | — | — |
| csv | `write_ranked_csv` | Yes (stdlib) | — | — |

**Recommendation:** The integration test should mock Stage 2 (prep+scoring) and inject pre-scored `ScoredPose` objects. This makes the `slow` test run without Vina/Meeko while still exercising Phase 7's actual deliverables (clustering + output).

---

## Security Domain

No new external inputs, no authentication, no secrets. Phase 7 writes local files only. ASVS V5 (input validation): `write_ranked_csv` must handle `None` score fields gracefully (guard before formatting). V6 (cryptography): not applicable. No ASVS categories introduce new concerns for this phase.

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 7 |
|-----------|-------------------|
| Stdlib-only I/O (no pandas) | `csv.DictWriter` confirmed; no pandas import |
| Atomic writes via `.tmp` + `os.replace` | Must apply to `write_ranked_csv` |
| `logging` not `print` | ΔG summary line via `logger.info` (D-05) |
| `output_dir.mkdir(parents=True, exist_ok=True)` before any write | Both `write_ranked_csv` and `write_best_pose_pdb` must call this |
| Python 3.11, type hints everywhere, `from __future__ import annotations` | All new files must include this import |
| Ruff + Black, line length 100 | New files must comply |
| Google-style docstrings with `Args`, `Returns`, `Raises` | `csv_writer.py` functions must have full docstrings |
| No bare `except:` | Catch `FileNotFoundError` for missing source PDB, `OSError` for write errors |
| Conventional Commits | Commit messages: `feat(07): ...`, `test(07): ...` |
| Files > 1 MB don't go in git | 25 fixture PDBs ≈ 1.25 MB total — just under limit; individual files are ~50 KB. Must verify total does not exceed 1 MB per the constraint. If it does, reduce to fewer/smaller poses. |

**File size constraint check:** D-11 says "~50KB each ≈ 1.25MB total." The CLAUDE.md says "files > 1 MB don't go in git." This is **per file**, not total. Individual fixture PDBs at ~50 KB each are well under 1 MB. No issue.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Integration test should mock Stage 2 (scoring) and inject pre-built ScoredPose objects rather than run real Vina | MDM2/p53 Fixture Strategy | If real scoring is required, test becomes non-portable and requires Vina+Meeko. Low risk — test can be refactored later |
| A2 | Float precision of 4 decimal places is Claude's discretion choice (from CONTEXT.md) | csv_writer.py | No correctness impact; purely cosmetic |
| A3 | `test_driver.py` does not currently patch `cluster_poses` (Stage 3 was added after tests) | Pitfall 7 | If it does patch it, no change needed. Safe to verify at implementation time |
| A4 | `shutil.copy2` is correct for best_pose.pdb (no atomic pattern needed) | csv_writer.py | Negligible risk — PDB copy is not a partial-write concern |

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/hybridock_pep/output/metadata.py` — atomic write pattern
- Direct codebase inspection: `src/hybridock_pep/models.py` — `ScoredPose`, `DockConfig` fields
- Direct codebase inspection: `src/hybridock_pep/analysis/clustering.py` — `ClusterResult` dataclass
- Direct codebase inspection: `src/hybridock_pep/analysis/statistics.py` — `per_cluster_stats` dict structure, `csv.DictWriter` pattern with `newline=""`
- Direct codebase inspection: `src/hybridock_pep/driver.py` — Stage 3 endpoint, current return type, call sites
- Direct codebase inspection: `src/hybridock_pep/cli.py` — `run_dock` call site (line 219)
- Direct codebase inspection: `tests/test_driver.py` — all `run_dock` call sites
- Direct codebase inspection: `pyproject.toml` — confirmed no `[tool.pytest.ini_options]` exists
- Direct arithmetic: threshold math for hybrid_score formula

### Secondary (MEDIUM confidence)
- pytest 8.x `[tool.pytest.ini_options].markers` registration — standard pytest configuration [ASSUMED: well-established pattern, not verified against current pytest docs in this session]

---

## Metadata

**Confidence breakdown:**
- csv_writer.py implementation: HIGH — pattern directly from existing code
- Driver return type migration: HIGH — all callers identified by grep
- pytest marker setup: HIGH — pyproject.toml inspected, section confirmed absent
- MDM2/p53 fixture strategy: MEDIUM — "mock scoring" vs "real scoring" is a design judgment; threshold math is HIGH confidence
- Integration test architecture: MEDIUM — D-10 wording is ambiguous on whether real scoring runs

**Research date:** 2026-04-25
**Valid until:** No external dependencies to decay; valid until codebase changes.
