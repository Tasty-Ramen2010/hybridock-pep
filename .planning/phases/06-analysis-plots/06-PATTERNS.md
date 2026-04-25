# Phase 6: Analysis & Plots - Pattern Map

**Mapped:** 2026-04-25
**Files analyzed:** 6 (4 new, 2 modified)
**Analogs found:** 6 / 6

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/hybridock_pep/analysis/clustering.py` | service | batch + transform | `src/hybridock_pep/scoring/entropy.py` | role-match (in-place mutation pattern + batch semantics) |
| `src/hybridock_pep/analysis/statistics.py` | utility | transform | `src/hybridock_pep/output/metadata.py` | role-match (structured data write, csv/json) |
| `src/hybridock_pep/analysis/plotting.py` | utility | batch + file-I/O | `src/hybridock_pep/output/metadata.py` | partial (file write pattern) |
| `src/hybridock_pep/analysis/__init__.py` | config | — | `src/hybridock_pep/scoring/vina.py` (module __init__ pattern) | partial |
| `src/hybridock_pep/driver.py` (modify lines 147-148) | service | request-response | self | exact (stub replacement) |
| `tests/test_clustering.py` | test | — | `tests/test_scoring.py` | exact (class-per-module, lazy imports, tmp_path fixtures) |

---

## Pattern Assignments

### `src/hybridock_pep/analysis/clustering.py` (service, batch + transform)

**Analog:** `src/hybridock_pep/scoring/entropy.py` (in-place mutation pattern) + `src/hybridock_pep/sampling/pose_io.py` (batch-failure tuple return, Biopython parsing)

**Imports pattern** — copy from `scoring/entropy.py` lines 14-27 and `sampling/pose_io.py` lines 17-26:
```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except ImportError:
    AgglomerativeClustering = None  # type: ignore[assignment,misc]
    silhouette_score = None  # type: ignore[assignment,misc]

_log = logging.getLogger(__name__)
```

**Lazy import pattern** — copy from `scoring/vina.py` lines 22-25:
```python
try:
    from vina import Vina
except ImportError:  # score-env not active (e.g. during unit tests with mocks)
    Vina = None  # type: ignore[assignment,misc]
```
Apply same pattern for `sklearn` and `scipy` in clustering.py and statistics.py.

**In-place mutation pattern** — copy from `scoring/entropy.py` lines 112-153 (`apply_hybrid_score`):
```python
def apply_hybrid_score(
    pose: ScoredPose,
    *,
    alpha: float,
    beta: float,
    n_residues: int,
) -> None:
    """Apply the D-01 hybrid score formula to a ScoredPose in place.
    ...
    """
    assert pose.vina_score is not None, "vina_score must be set before apply_hybrid_score"
    assert pose.ad4_score is not None, "ad4_score must be set before apply_hybrid_score"

    pose.entropy_correction = alpha * n_residues
    pose.hybrid_score = (
        pose.vina_score + beta * (pose.ad4_score - pose.vina_score) + pose.entropy_correction
    )
    _log.debug(
        "Pose %d: vina=%.3f ad4=%.3f ec=%.3f hybrid=%.3f",
        pose.pose_idx,
        ...
    )
```
For `cluster_poses()`: mutate `pose.cluster_id = int(labels[i])` in-place on every pose, following this exact pattern (mutation is the primary side effect; function returns `ClusterResult`).

**Batch-failure tuple return** — copy from `sampling/pose_io.py` lines 29-87 (`parse_poses`):
```python
def parse_poses(poses_dir: Path) -> tuple[list[PoseRecord], list[PoseFailure]]:
    records: list[PoseRecord] = []
    failures: list[PoseFailure] = []
    ...
    for pdb_path in pdb_files:
        try:
            record = _parse_single_pose(pose_idx, pdb_path)
            records.append(record)
        except Exception as e:  # noqa: BLE001
            failures.append(PoseFailure(pose_idx=pose_idx, stage="parsing", error_msg=f"{type(e).__name__}: {e}"))
            logger.warning("Pose %d parse failed: %s", pose_idx, e)
    return records, failures
```
`cluster_poses()` does NOT return a failures tuple (it returns `ClusterResult`), but uses the same `try/except` per-item guard pattern internally. Receptor parse failure → fall back gracefully (not a `PoseFailure` at the batch level).

**Biopython Cα extraction** — copy from `sampling/pose_io.py` lines 118-148 (`_parse_single_pose`):
```python
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

parser = PDBParser(QUIET=True)
structure = parser.get_structure(f"pose_{pose_idx}", str(pdb_path))
model = next(iter(structure))
for chain in model:
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        if "CA" not in residue:
            continue
        ca_coords_list.append(list(residue["CA"].get_vector().get_array()))
ca_coords = np.array(ca_coords_list, dtype=np.float64)  # shape [n_res, 3]
```
Use this exact pattern in `_load_receptor_ca_coords()`. Note: use `protein_letters_3to1` import workaround from line 118 if needed for residue name conversion.

**@dataclass definition** — copy from `src/hybridock_pep/models.py` lines 94-110 (`PoseRecord`):
```python
@dataclass
class PoseRecord:
    """Parsed peptide pose with pre-extracted C-alpha coordinates.

    Args:
        pose_idx: Zero-based index of this pose within the sampling run.
        pdb_path: Absolute path to the raw PDB file produced by RAPiDock.
        sequence: Single-letter amino acid sequence parsed from the PDB.
        ca_coords: Shape (n_residues, 3) float64 array of C-alpha XYZ
            coordinates. Populated at parse time; never re-read from disk.
    """
    pose_idx: int
    pdb_path: Path
    sequence: str
    ca_coords: np.ndarray
```
For `ClusterResult`: same `@dataclass` style, Google-style docstring with `Args` block. **Important:** `ClusterResult` is defined in `analysis/clustering.py`, NOT `models.py` (D-09 decision). Fields:
```python
@dataclass
class ClusterResult:
    k_optimal: int
    silhouette_score: float
    per_cluster_stats: list[dict]
```

**Logging pattern** — copy from `scoring/entropy.py` line 27 and `sampling/pose_io.py` line 26:
```python
_log = logging.getLogger(__name__)   # entropy.py style (module-level _log)
logger = logging.getLogger(__name__) # pose_io.py style (module-level logger)
```
Use `logger` (not `_log`) to match the more common project convention in pose_io.py and vina.py.

---

### `src/hybridock_pep/analysis/statistics.py` (utility, transform)

**Analog:** `src/hybridock_pep/output/metadata.py`

**Imports pattern** — copy from `output/metadata.py` lines 15-29:
```python
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from hybridock_pep.models import ScoredPose

try:
    from scipy.stats import t as t_dist
except ImportError:
    t_dist = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
```

**File write pattern** — copy from `output/metadata.py` lines 44-61 (`write_metadata_skeleton`):
```python
data = {
    "status": "running",
    "timestamp_start": datetime.now(tz=timezone.utc).isoformat(),
    ...
}
_write_json_atomic(metadata_path, data)
logger.info("Metadata skeleton written to %s", metadata_path)
```
For `write_cluster_summary_csv()`: use `csv.DictWriter` with explicit `fieldnames` (ordered per D-10 decision: `cluster_id, n_poses, mean_hybrid_score, std_hybrid_score, ci95_lower, ci95_upper, best_pose_idx`). Log the output path with `logger.info(...)`.

**Read-modify-write pattern** — copy from `output/metadata.py` lines 64-80 (`finalize_metadata`):
```python
def finalize_metadata(metadata_path: Path, poses_generated: int, ...) -> None:
    """Overwrite run_metadata.json with final counts and status (D-15).

    Uses read-modify-write to preserve any clipped_poses entries ...
    """
```
Not directly needed in statistics.py, but the pattern of "write to a path, log success, use parent.mkdir(parents=True, exist_ok=True)" is standard throughout the project.

**Path creation guard** — copy from `scoring/entropy.py` lines 106-108 (`write_calibration`):
```python
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w") as fh:
    json.dump(payload, fh, indent=2)
logger.info("Wrote calibration to %s (alpha=%.3f, beta=%.3f)", path, alpha, beta)
```
Apply same pattern in `write_cluster_summary_csv()`:
```python
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[...])
    writer.writeheader()
    writer.writerows(rows)
logger.info("Wrote cluster_summary.csv to %s", output_path)
```

---

### `src/hybridock_pep/analysis/plotting.py` (utility, batch + file-I/O)

**Analog:** No direct analog in codebase. Use RESEARCH.md Pattern 5 (Matplotlib Agg) as primary reference.

**Import + backend pattern** — from RESEARCH.md Pattern 5 (verified against matplotlib docs):
```python
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # MUST be before any import of matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
```

**Figure lifecycle pattern** (critical — no analog in codebase, but firmly established by RESEARCH.md):
```python
fig, ax = plt.subplots(figsize=(8, 5))
# ... ax.plot(), ax.fill_between(), ax.set_xlabel(), ax.set_ylabel(), ax.legend() ...
fig.tight_layout()
fig.savefig(output_path, dpi=150)
plt.close(fig)  # CRITICAL: release memory; avoids ResourceWarning in tests
```
`plt.close(fig)` after every `savefig` is mandatory (RESEARCH.md anti-pattern list).

**Function signature style** — copy from `scoring/entropy.py` lines 33-51 (Google-style docstring with Args/Returns/Raises):
```python
def plot_convergence(
    scored_poses: list[ScoredPose],
    output_path: Path,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 150,
) -> None:
    """Generate convergence plot (score-sorted running mean ± σ).

    Args:
        scored_poses: List of scored poses (hybrid_score must be populated).
        output_path: Absolute path to write the PNG file.
        figsize: Matplotlib figure size in inches. Default (8, 5).
        dpi: Output resolution. Default 150.
    """
```

---

### `src/hybridock_pep/analysis/__init__.py` (config)

**Analog:** Pattern from other `__init__.py` files. The file currently exists and is empty (1 line).

**Export pattern** — add a single export, consistent with how driver.py imports from subpackages (driver.py lines 7-15):
```python
from hybridock_pep.analysis.clustering import cluster_poses

__all__ = ["cluster_poses"]
```
Keep it minimal — only `cluster_poses` needs to be re-exported since that is the public API consumed by `driver.py`.

---

### `src/hybridock_pep/driver.py` — modify lines 147-148 (stub replacement)

**Current stub** (`driver.py` lines 147-148):
```python
    # Stage 3 stub: Clustering and output writing are Phase 6/7 scope
    logger.info("Clustering and output: Phase 6/7 not yet implemented")
```

**Replacement pattern** — follow the established Stage 2 call pattern from `driver.py` lines 137-145:
```python
    calibration = load_calibration(calibration_path.resolve())
    alpha: float = calibration["alpha"]
    ...
    for pose in scored_poses:
        apply_hybrid_score(pose, alpha=alpha, beta=beta, n_residues=n_residues)

    logger.info("Stage 2 complete: %d poses scored", len(scored_poses))
```
Replace stub with:
```python
    # Stage 3: Clustering and analysis
    from hybridock_pep.analysis import cluster_poses
    cluster_result = cluster_poses(scored_poses, config)
    logger.info(
        "Stage 3 complete: k=%d clusters, silhouette=%.3f",
        cluster_result.k_optimal,
        cluster_result.silhouette_score,
    )
```
Import `cluster_poses` inside the function (or at module top alongside other imports at lines 7-15) — use inline import to defer the sklearn dependency until clustering actually runs, consistent with the lazy-import philosophy demonstrated in `vina.py` lines 22-25.

The `finalize_metadata` call at line 151 stays in place (no structural change to driver.py beyond replacing lines 147-148).

---

### `tests/test_clustering.py` (test)

**Analog:** `tests/test_scoring.py` (exact match — same class-per-module structure, lazy imports, tmp_path fixtures, DockConfig construction)

**File header pattern** — copy from `tests/test_scoring.py` lines 1-12:
```python
"""Tests for Phase 6 analysis modules — clustering, statistics, plotting (ANAL-01..03).

All hybridock_pep imports are lazy (inside test functions) per STATE.md decision:
"All hybridock_pep imports kept lazy in test files."
"""
from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
```

**FIXTURES_DIR pattern** — copy from `tests/test_sampling.py` line 11:
```python
FIXTURES_DIR = Path(__file__).parent / "fixtures"
```

**pytest fixture pattern** — copy from `tests/test_sampling.py` lines 17-29 (`config` fixture):
```python
@pytest.fixture()
def config(self, tmp_path: Path):
    from hybridock_pep.models import DockConfig

    receptor = FIXTURES_DIR / "receptor_tiny.pdb"
    return DockConfig(
        peptide_sequence="ALA",
        receptor_path=receptor,
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=tmp_path / "out",
        n_samples=5,
    )
```
For test_clustering.py: use `FIXTURES_DIR / "receptor_tiny.pdb"` as the receptor file (already exists in fixtures). Build synthetic `ScoredPose` objects with `np.zeros((n, 3))` or small deterministic `ca_coords` arrays inline.

**ScoredPose construction in tests** — copy from `tests/test_scoring.py` lines 96-103:
```python
pose = ScoredPose(
    pose_idx=0,
    pdb_path=tmp_path / "pose_0.pdb",
    sequence="ACDE",
    ca_coords=np.zeros((4, 3)),
    pdbqt_path=Path("/tmp/nonexistent_xyz_abc.pdbqt"),
)
```
For clustering tests: construct multiple `ScoredPose` objects with distinct `ca_coords` to create two clearly separable clusters. Set `hybrid_score` directly on each pose.

**Lazy import pattern in tests** — copy from `tests/test_scoring.py` lines 38-41:
```python
def test_check_grid_boundary_inside(self, tmp_path: Path) -> None:
    """Atom at exact center → not clipped → returns False."""
    from hybridock_pep.scoring.vina import check_grid_boundary
    ...
```
Every test method imports from `hybridock_pep.*` inside the function body, not at module top.

**Class-per-module structure**:
```python
class TestClustering:
    """Tests for cluster_poses(), ClusterResult, RMSD matrix, silhouette loop."""

class TestStatistics:
    """Tests for compute_cluster_stats(), write_cluster_summary_csv()."""

class TestPlotting:
    """Tests for plot_convergence(), plot_silhouette()."""
```

---

## Shared Patterns

### `from __future__ import annotations`
**Source:** Every module in the project (`models.py` line 1, `entropy.py` line 14, `pose_io.py` line 17, `vina.py` line 14, `driver.py` line 1)
**Apply to:** All four new/modified files (`clustering.py`, `statistics.py`, `plotting.py`, `test_clustering.py`)
```python
from __future__ import annotations
```
First line of every module. No exceptions.

### Logger naming
**Source:** `sampling/pose_io.py` line 26, `scoring/vina.py` line 28, `output/metadata.py` line 29
**Apply to:** `clustering.py`, `statistics.py`, `plotting.py`
```python
logger = logging.getLogger(__name__)
```
Use `logger` (not `_log`) — the dominant pattern across newer modules.

### Lazy optional-dependency imports with `try/except ImportError`
**Source:** `scoring/vina.py` lines 22-25
**Apply to:** `clustering.py` (sklearn), `statistics.py` (scipy)
```python
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except ImportError:
    AgglomerativeClustering = None  # type: ignore[assignment,misc]
    silhouette_score = None         # type: ignore[assignment,misc]
```

### Google-style docstrings with Args/Returns/Raises
**Source:** `scoring/entropy.py` lines 33-79, `sampling/pose_io.py` lines 39-47
**Apply to:** All public functions in `clustering.py`, `statistics.py`, `plotting.py`
Minimum sections: `Args`, `Returns` (or note side effects), `Raises` if applicable.

### Path resolution before passing to sub-calls
**Source:** `driver.py` line 35 ("All paths passed to sub-module calls are resolved to absolute before use")
**Apply to:** `clustering.py` `_load_receptor_ca_coords(config.receptor_path.resolve())`

### No bare `except:`
**Source:** CLAUDE.md §4, `pose_io.py` line 72 comment `# noqa: BLE001`
**Apply to:** All exception handling in clustering.py. If catching broad `Exception`, add `# noqa: BLE001` comment explaining why.

### `PoseFailure(stage="clustering")`
**Source:** `models.py` lines 142-155 — `PoseFailure.stage` is `Literal["parsing", "prep", "scoring", "clustering"]`; `"clustering"` is already defined as a valid literal.
**Apply to:** Any per-pose failure inside `cluster_poses()`. The literal is pre-registered — no models.py change needed.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/hybridock_pep/analysis/plotting.py` | utility | file-I/O | No matplotlib usage anywhere in current codebase. RESEARCH.md Pattern 5 is the authoritative reference. |

---

## Metadata

**Analog search scope:** `src/hybridock_pep/scoring/`, `src/hybridock_pep/sampling/`, `src/hybridock_pep/output/`, `src/hybridock_pep/models.py`, `src/hybridock_pep/driver.py`, `tests/`
**Files scanned:** 10 source files
**Pattern extraction date:** 2026-04-25
