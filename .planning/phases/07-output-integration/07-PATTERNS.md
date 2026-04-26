# Phase 7: Output & Integration - Pattern Map

**Mapped:** 2026-04-25
**Files analyzed:** 8 (6 new, 2 modified)
**Analogs found:** 8 / 8

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/hybridock_pep/output/csv_writer.py` | output-writer | file-I/O | `src/hybridock_pep/output/metadata.py` | exact |
| `src/hybridock_pep/output/__init__.py` | config | — | `src/hybridock_pep/output/__init__.py` (current) | exact |
| `src/hybridock_pep/driver.py` | orchestrator | request-response | `src/hybridock_pep/driver.py` (current Stage 3 block) | exact |
| `pyproject.toml` | config | — | `pyproject.toml` (current `[tool.*]` sections) | exact |
| `tests/test_e2e.py` | test | request-response | `tests/test_driver.py` + `tests/test_output.py` | role-match |
| `tests/test_csv_writer.py` | test | file-I/O | `tests/test_output.py` | exact |
| `tests/fixtures/mdm2_p53/pose_000.pdb … pose_024.pdb` | fixture | file-I/O | `tests/test_driver.py` `_make_pose_record()` inline write | partial-match |
| `tests/fixtures/mdm2_calibration.json` | fixture | file-I/O | entropy.py calibration format (alpha/beta dict) | partial-match |

---

## Pattern Assignments

### `src/hybridock_pep/output/csv_writer.py` (output-writer, file-I/O)

**Analog:** `src/hybridock_pep/output/metadata.py`

**Imports pattern** (metadata.py lines 15–28):
```python
from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path

from hybridock_pep.models import DockConfig, ScoredPose
from hybridock_pep.analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)
```
Note: replace `json`/`hashlib` imports with `csv` and `shutil`. All other conventions identical.

**Atomic write pattern** (metadata.py lines 128–133):
```python
def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomically write data as JSON to path using a .tmp intermediate file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
```
Apply same pattern for CSV: write to `path.with_suffix(".tmp")`, then `os.replace(tmp, path)`.

**CSV write pattern** (statistics.py lines 155–172 — the only existing csv.DictWriter usage):
```python
FIELDNAMES = [
    "cluster_id",
    "n_poses",
    ...
]

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(stats)

logger.info("Wrote cluster_summary.csv to %s", output_path)
```
For `csv_writer.py`, adapt with these FIELDNAMES (D-02):
```python
FIELDNAMES = [
    "rank", "hybrid_score", "vina_score", "ad4_score",
    "entropy_correction", "delta_g", "cluster_id",
    "pose_filename", "is_ad4_anomaly", "is_clipped",
]
```
Use atomic write (`.tmp` + `os.replace`) instead of direct open, unlike `statistics.py` which writes directly. Follow `metadata.py`'s atomic pattern.

**Logger pattern** (metadata.py lines 61, 97–101):
```python
logger.info("Metadata skeleton written to %s", metadata_path)

logger.info(
    "Metadata finalized: %d/%s poses, status=%s",
    poses_generated,
    data.get("poses_requested", "?"),
    status,
)
```
Use `%`-style lazy formatting. Match the same pattern: one `logger.info` call per public function, at the end, confirming the write path.

**mkdir guard pattern** (metadata.py line 130):
```python
path.parent.mkdir(parents=True, exist_ok=True)
```
Call this at the top of every write function before opening any file handle.

**`write_ranked_csv` function shape:**
- Signature: `write_ranked_csv(scored_poses: list[ScoredPose], config: DockConfig) -> Path`
- Sort poses by `hybrid_score` ascending (most negative first), take top 10
- Build row dicts: `rank` = 1-based index, `hybrid_score` = `pose.hybrid_score`, `delta_g` = `pose.hybrid_score` (identical, D-04), `pose_filename` = `pose.pdb_path.name`, floats to 4 decimal places (Claude's Discretion)
- Write to `config.output_dir / "ranked_poses.csv"` atomically
- Return the destination path

**`write_best_pose_pdb` function shape:**
- Signature: `write_best_pose_pdb(cluster_result: ClusterResult, config: DockConfig) -> Path`
- Find top cluster: `min(cluster_result.per_cluster_stats, key=lambda s: s["mean_hybrid_score"])`
- Source: `config.output_dir / "poses" / f"pose_{best_pose_idx:03d}.pdb"` (D-07)
- Dest: `config.output_dir / "best_pose.pdb"`
- Copy with `shutil.copy2(src, dest)` — no atomic needed (copy is crash-safe enough for a PDB)
- Return dest path

---

### `src/hybridock_pep/output/__init__.py` (config)

**Analog:** current `src/hybridock_pep/output/__init__.py` (lines 1–11)

**Current file** (lines 1–11):
```python
from hybridock_pep.output.metadata import (
    finalize_metadata,
    get_rapidock_commit_sha,
    write_metadata_skeleton,
)

__all__ = [
    "write_metadata_skeleton",
    "finalize_metadata",
    "get_rapidock_commit_sha",
]
```

**Modified file — add two imports and two `__all__` entries:**
```python
from hybridock_pep.output.csv_writer import (
    write_best_pose_pdb,
    write_ranked_csv,
)
```
Add `"write_ranked_csv"` and `"write_best_pose_pdb"` to `__all__`. Keep existing entries unchanged.

---

### `src/hybridock_pep/driver.py` (orchestrator — Stage 4 addition)

**Analog:** `src/hybridock_pep/driver.py` lines 147–162 (current Stage 3 block)

**Current Stage 3 pattern** (lines 147–162):
```python
    # Stage 3: Clustering and analysis
    if len(scored_poses) >= 2:
        from hybridock_pep.analysis import cluster_poses
        cluster_result = cluster_poses(scored_poses, config)
        logger.info(
            "Stage 3 complete: k=%d clusters, silhouette=%.3f",
            cluster_result.k_optimal,
            cluster_result.silhouette_score,
        )
    else:
        logger.warning("Stage 3 skipped: no scored poses to cluster")

    # Finalize metadata AFTER scoring
    finalize_metadata(metadata_path, poses_generated=len(records))

    return scored_poses
```

**Stage 4 block to insert after line 160 (after `finalize_metadata(...)`) and before `return`:**
```python
    # Stage 4: Output writing
    from hybridock_pep.output.csv_writer import write_ranked_csv, write_best_pose_pdb
    write_ranked_csv(scored_poses, config)
    if len(scored_poses) >= 2:
        write_best_pose_pdb(cluster_result, config)
        best_cluster = min(
            cluster_result.per_cluster_stats,
            key=lambda s: s["mean_hybrid_score"],
        )
        best_pose_filename = f"pose_{best_cluster['best_pose_idx']:03d}.pdb"
        logger.info(
            "Best pose: ΔG = %.1f kcal/mol (cluster %d, %s)",
            best_cluster["mean_hybrid_score"],
            best_cluster["cluster_id"],
            best_pose_filename,
        )

    return scored_poses, cluster_result
```

**Return type change:** The function signature at line 24 must change:
```python
# Before (line 24):
) -> list[ScoredPose]:

# After:
) -> tuple[list[ScoredPose], ClusterResult]:
```

Add `ClusterResult` to the top-level import at line 6:
```python
# Before (line 6):
from hybridock_pep.models import DockConfig, PoseRecord, ScoredPose

# After:
from hybridock_pep.models import DockConfig, PoseRecord, ScoredPose
from hybridock_pep.analysis.clustering import ClusterResult
```

**Two callers need updating** (identified in RESEARCH.md):
1. `src/hybridock_pep/cli.py` line 224: unpack the tuple `scored_poses, cluster_result = run_dock(...)`
2. `tests/test_driver.py` lines 133–135: `result = driver.run_dock(...)` — tests checking `isinstance(result, list)` need to assert on the first element of the tuple instead

**logger.info pattern** for ΔG summary (from metadata.py lines 97–101):
```python
logger.info(
    "Best pose: ΔG = %.1f kcal/mol (cluster %d, %s)",
    best_cluster["mean_hybrid_score"],
    best_cluster["cluster_id"],
    best_pose_filename,
)
```
Use `%`-style lazy formatting, consistent with all other `logger.info` calls in `driver.py`.

---

### `pyproject.toml` (config — pytest marker registration)

**Analog:** existing `[tool.mypy]` / `[tool.ruff]` / `[tool.black]` sections (pyproject.toml lines 42–50)

**Pattern:** new `[tool.pytest.ini_options]` section appended after `[tool.mypy]`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: slow integration tests requiring full score-env tool stack (Vina, Meeko, ADFRsuite)",
]
```
This is the standard pytest marker registration pattern. Without it, `pytest -m slow` raises `PytestUnknownMarkWarning`.

---

### `tests/test_csv_writer.py` (test, file-I/O)

**Analog:** `tests/test_output.py` (TestMetadata class)

**Module-level structure** (test_output.py lines 1–11):
```python
"""Tests for hybridock_pep.output.metadata (SAMP-02)."""
from __future__ import annotations

import json
import os
import unittest.mock as mock
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
```
Copy exactly: `from __future__ import annotations`, lazy imports inside test methods, `FIXTURES_DIR` constant, class-based test grouping.

**Fixture pattern** (test_output.py lines 17–27):
```python
@pytest.fixture()
def config(self, tmp_path: Path):
    from hybridock_pep.models import DockConfig

    return DockConfig(
        peptide_sequence="ALA",
        receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=tmp_path / "out",
    )
```
Use the same `tmp_path`-based `DockConfig` fixture. For `test_csv_writer.py`, `peptide_sequence` can be `"ACDEF"` (5 residues → round entropy numbers). Use `FIXTURES_DIR / "receptor_tiny.pdb"` for receptor.

**Helper for synthetic ScoredPose** (test_clustering.py lines 33–50):
```python
def _make_scored_poses(tmp_path: Path):
    """Return 10 ScoredPose objects with populated ca_coords and hybrid_score."""
    from hybridock_pep.models import ScoredPose

    poses = []
    for i, (coords, score) in enumerate(
        zip(_GROUP_A_COORDS + _GROUP_B_COORDS, _SCORES_A + _SCORES_B)
    ):
        pose = ScoredPose(
            pose_idx=i,
            pdb_path=tmp_path / f"pose_{i}.pdb",
            sequence="ACDEF",
            ca_coords=coords,
            pdbqt_path=tmp_path / f"pose_{i}.pdbqt",
        )
        pose.hybrid_score = score
        poses.append(pose)
    return poses
```
For `test_csv_writer.py`, build a small `_make_scored_poses()` helper that produces 3–5 `ScoredPose` objects with `hybrid_score`, `vina_score`, `ad4_score`, `entropy_correction`, `cluster_id`, `is_ad4_anomaly`, `is_clipped` all populated. pdb_path files do NOT need to exist on disk for csv_writer unit tests (csv_writer only reads `.name`, not the file content).

**Assert pattern** (test_output.py lines 60–62):
```python
assert metadata_path.exists(), "run_metadata.json must be created"
data = json.loads(metadata_path.read_text())
assert data["status"] == "running"
```
For CSV tests, use `csv.DictReader`:
```python
import csv
assert csv_path.exists()
rows = list(csv.DictReader(csv_path.open()))
assert len(rows) <= 10
assert "hybrid_score" in rows[0]
```

**Key test cases for `test_csv_writer.py`:**
1. `test_write_ranked_csv_creates_file` — file exists after call
2. `test_write_ranked_csv_columns` — all 10 D-02 columns present
3. `test_write_ranked_csv_sorted_ascending` — row 0 has lowest `hybrid_score`
4. `test_write_ranked_csv_top10_limit` — with 15 poses, only 10 rows written
5. `test_write_ranked_csv_delta_g_equals_hybrid` — `delta_g == hybrid_score` for every row
6. `test_write_best_pose_pdb_copies_file` — `best_pose.pdb` exists and is non-empty
7. `test_write_best_pose_pdb_selects_best_cluster` — copies from the cluster with lowest `mean_hybrid_score`

---

### `tests/test_e2e.py` (test, request-response — integration)

**Analog:** `tests/test_driver.py` (class-based structure, `_make_config()` pattern)

**Module structure** (test_driver.py lines 1–9):
```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
```

**`@pytest.mark.slow` class decorator** — this is the key addition not present in existing tests:
```python
@pytest.mark.slow
class TestMDM2P53Integration:
    def test_corrected_delta_g_passes_threshold(self, tmp_path: Path) -> None:
        ...
```
All methods inside inherit the `slow` marker from the class.

**`run_dock` call pattern** (test_driver.py lines 73–75):
```python
driver.run_dock(config, input_poses_dir=poses_dir, calibration_path=tmp_path / "cal.json")
```
For `test_e2e.py`, call the real `run_dock` (no mocking) with:
- `input_poses_dir = FIXTURES_DIR / "mdm2_p53"` (25 real fixture PDBs)
- `calibration_path = FIXTURES_DIR / "mdm2_calibration.json"` (alpha=0.2, beta=0.0)

**D-12 assertions to implement:**
```python
scored_poses, cluster_result = run_dock(config, input_poses_dir=poses_dir, calibration_path=cal_path)

csv_path = tmp_path / "mdm2_run" / "ranked_poses.csv"
assert csv_path.exists()
rows = list(csv.DictReader(csv_path.open()))
assert 1 <= len(rows) <= 10
required_cols = {"rank", "hybrid_score", "vina_score", "ad4_score",
                 "entropy_correction", "delta_g", "cluster_id",
                 "pose_filename", "is_ad4_anomaly", "is_clipped"}
assert required_cols.issubset(rows[0].keys())

best_hybrid = min(float(r["hybrid_score"]) for r in rows)
assert best_hybrid < -3.0, f"TEST-02 threshold failed: best={best_hybrid:.2f}"

best_pdb = tmp_path / "mdm2_run" / "best_pose.pdb"
assert best_pdb.exists() and best_pdb.stat().st_size > 0

metadata = json.loads((tmp_path / "mdm2_run" / "run_metadata.json").read_text())
assert metadata["status"] == "complete"
```

**MDM2 binding site coords** (from PDB 2OY2, MDM2 active site):
```python
site_coords=(26.4, 3.5, -5.6)  # approximate MDM2 binding groove center from 2OY2
```
These will need to be confirmed from the actual fixture PDB coordinates or set to the centroid of the fixture peptide's CA atoms.

---

### `tests/fixtures/mdm2_p53/pose_000.pdb … pose_024.pdb` (fixture files)

**Analog:** inline PDB write in test_driver.py `_make_pose_record()` (lines 13–14):
```python
pdb = tmp_path / f"pose_{idx}.pdb"
pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")
```

**Required format** (from RESEARCH.md fixture strategy):
- Must be parseable by `pose_io._parse_single_pose()` via Biopython `PDBParser`
- `ATOM` records with standard residues; `CA` atoms mandatory; full backbone (N, CA, C, O) preferred
- Named `pose_000.pdb` … `pose_024.pdb`
- Peptide: `ETFSDLWKLLPE` (12 residues)

**Generation approach** (from RESEARCH.md): Use `scripts/generate_mdm2_fixtures.py` to extract ETFSDLWKLLPE backbone from PDB 2OY2 and write 25 variants with ±0.1–0.5 Å random perturbation. The script runs once; its output is committed. The alternative (25 identical files) also works since the test validates output correctness, not clustering accuracy.

**Minimal valid PDB ATOM line format:**
```
ATOM      1  N   GLU A   1      26.000   3.000  -5.000  1.00  0.00           N
ATOM      2  CA  GLU A   1      26.500   3.500  -5.500  1.00  0.00           C
ATOM      3  C   GLU A   1      27.000   4.000  -6.000  1.00  0.00           C
ATOM      4  O   GLU A   1      27.500   4.500  -6.500  1.00  0.00           O
...
END
```

---

### `tests/fixtures/mdm2_calibration.json` (fixture — calibration data)

**Analog:** `entropy.py` `load_calibration()` return shape (alpha/beta dict).

**Format** (from RESEARCH.md threshold math and entropy.py):
```json
{
  "alpha": 0.2,
  "beta": 0.0
}
```
`alpha=0.2` (minimum allowed by `load_calibration` validator) reduces the entropy penalty to `0.2 × 12 = 2.4 kcal/mol` for ETFSDLWKLLPE. `beta=0.0` means pure Vina scoring (`hybrid = vina + 2.4`). Any Vina score below `-5.4` passes the `< -3.0` threshold. MDM2/p53 with real scoring reliably produces Vina < -6.0.

---

## Shared Patterns

### Atomic File Write
**Source:** `src/hybridock_pep/output/metadata.py` lines 128–133
**Apply to:** `csv_writer.py` `write_ranked_csv()`
```python
def _write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Atomically write rows as CSV to path using a .tmp intermediate file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
```

### Logger Pattern (module-level, `%`-style)
**Source:** `src/hybridock_pep/output/metadata.py` line 29; `driver.py` line 17
**Apply to:** `csv_writer.py`, `driver.py` Stage 4 block
```python
logger = logging.getLogger(__name__)
```
All log calls use `%`-style lazy formatting: `logger.info("msg %s", value)` not f-strings.

### Lazy Imports in Tests
**Source:** `tests/test_driver.py` lines 56, 101, etc.; `tests/test_clustering.py` line 36
**Apply to:** `tests/test_csv_writer.py`, `tests/test_e2e.py`
```python
# Import hybridock_pep modules inside test methods, not at module level
def test_something(self, tmp_path):
    from hybridock_pep.output.csv_writer import write_ranked_csv
    ...
```
This is the established project convention for all test files.

### `FIXTURES_DIR` Constant
**Source:** `tests/test_output.py` line 11; `tests/test_clustering.py` line 15
**Apply to:** `tests/test_csv_writer.py`, `tests/test_e2e.py`
```python
FIXTURES_DIR = Path(__file__).parent / "fixtures"
```

### `_make_config()` Helper
**Source:** `tests/test_driver.py` lines 23–37; `tests/test_clustering.py` lines 53–60
**Apply to:** `tests/test_csv_writer.py`, `tests/test_e2e.py`
```python
def _make_config(tmp_path: Path):
    from hybridock_pep.models import DockConfig
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    return DockConfig(
        peptide_sequence="AAAA",
        receptor_path=receptor,
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=tmp_path,
        seed=42,
    )
```
For `test_e2e.py`, use `FIXTURES_DIR / "receptor_tiny.pdb"` instead of the inline-written receptor, since the real test needs a valid receptor for actual Vina/AD4 scoring.

### `from __future__ import annotations`
**Source:** Every source file and test file in the project
**Apply to:** All new files (`csv_writer.py`, `test_csv_writer.py`, `test_e2e.py`)
This is the first line of every module, per CLAUDE.md §4.

### Google-style Docstrings
**Source:** `metadata.py` lines 32–43, `statistics.py` lines 65–88
**Apply to:** `csv_writer.py` public functions
```python
def write_ranked_csv(scored_poses: list[ScoredPose], config: DockConfig) -> Path:
    """Write top-10 poses ranked by hybrid_score to ranked_poses.csv.

    Args:
        scored_poses: All scored poses from the pipeline. Sorted internally;
            top 10 by hybrid_score (ascending) are written.
        config: Run configuration. output_dir is used as the write destination.

    Returns:
        Path to the written ranked_poses.csv file.
    """
```

---

## No Analog Found

No files in Phase 7 are without an analog. All patterns are established in the codebase.

---

## Metadata

**Analog search scope:** `src/hybridock_pep/output/`, `src/hybridock_pep/analysis/`, `src/hybridock_pep/driver.py`, `src/hybridock_pep/models.py`, `tests/`, `pyproject.toml`
**Files scanned:** 9 source files + pyproject.toml
**Pattern extraction date:** 2026-04-25

### Critical Implementation Notes for Planner

1. **`driver.py` return type change breaks two callers.** `cli.py` (line ~224) and `test_driver.py` tests `test_returns_list_of_scored_poses` (line 134) and `test_all_stages_called_in_order` both need updating. The planner should include these as explicit sub-tasks in the driver plan.

2. **`cluster_result` variable scoping in driver.py.** Currently `cluster_result` is only defined inside the `if len(scored_poses) >= 2:` branch (line 150). The Stage 4 block also has a guard `if len(scored_poses) >= 2:` before calling `write_best_pose_pdb`. The return statement `return scored_poses, cluster_result` must handle the else-branch — either set `cluster_result = None` before the Stage 3 block, or use `ClusterResult(k_optimal=0, silhouette_score=0.0)` as a sentinel. The return type annotation must reflect this: `tuple[list[ScoredPose], ClusterResult | None]`.

3. **statistics.py `write_cluster_summary_csv` does NOT use atomic writes** (lines 165–170 write directly). `csv_writer.py` MUST use atomic writes (`.tmp` + `os.replace`) per the metadata.py pattern — this is the stated requirement in CONTEXT.md and RESEARCH.md.

4. **Float formatting in CSV rows.** Use `f"{value:.4f}"` for all score fields when building the row dict before passing to `csv.DictWriter`. `csv.DictWriter` writes whatever string is in the dict — it does not auto-format floats.

5. **`pose_filename` column.** Source: `pose.pdb_path.name` (just the filename, not the full path). This is what users need to locate poses in the output directory.
