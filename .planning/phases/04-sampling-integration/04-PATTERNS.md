# Phase 4: Sampling Integration - Pattern Map

**Mapped:** 2026-04-21
**Files analyzed:** 6 (4 source + 2 test)
**Analogs found:** 5 / 6 (run_rapidock.py has no direct analog)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/hybridock_pep/sampling/rapidock_runner.py` | service | streaming + batch | `src/hybridock_pep/prep/receptor.py` (subprocess) + `src/hybridock_pep/scoring/vina.py` (batch loop) | role-match |
| `src/hybridock_pep/sampling/run_rapidock.py` | utility | request-response | none | no-analog |
| `src/hybridock_pep/sampling/pose_io.py` | service | batch + file-I/O | `src/hybridock_pep/prep/ligand.py` | exact |
| `src/hybridock_pep/output/metadata.py` | utility | file-I/O | `src/hybridock_pep/scoring/vina.py` `_append_clipped_pose` (lines 77-104) | partial-match |
| `tests/test_sampling.py` | test | — | `tests/test_prep.py` + `tests/test_scoring.py` | role-match |
| `tests/test_output.py` | test | — | `tests/test_scoring.py` (JSON write tests) | role-match |

---

## Pattern Assignments

### `src/hybridock_pep/sampling/rapidock_runner.py` (service, streaming)

**Analogs:** `src/hybridock_pep/prep/receptor.py` (subprocess pattern) and `src/hybridock_pep/scoring/vina.py` (batch + logger pattern)

**Imports pattern** — from `src/hybridock_pep/prep/receptor.py` lines 1-14:
```python
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)
```

For `rapidock_runner.py`, add `threading` and `glob`/`re` for file renaming:
```python
from __future__ import annotations

import logging
import re
import subprocess
import threading
from pathlib import Path

from hybridock_pep.models import DockConfig, PoseRecord

logger = logging.getLogger(__name__)
```

**Subprocess invocation + log-before-call pattern** — from `src/hybridock_pep/prep/receptor.py` lines 67-88:
```python
cmd = [
    "prepare_receptor4.py",
    "-r", str(fixed_pdb_path),
    "-o", str(pdbqt_path),
]
logger.info("Running: %s", " ".join(cmd))
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
finally:
    fixed_pdb_path.unlink(missing_ok=True)

if result.returncode != 0:
    raise PrepError(
        f"prepare_receptor4.py failed (exit {result.returncode}):\n{result.stderr}"
    )
```

**IMPORTANT DIFFERENCE for `rapidock_runner.py`:** Do NOT use `subprocess.run()` or `communicate()`. Use `subprocess.Popen` with a `readline()` loop (D-01, D-02). The receptor.py pattern shows `subprocess.run` — rapidock_runner.py deviates from this intentionally for real-time OOM surfacing.

**Popen + stderr daemon thread pattern** — from CONTEXT.md specifics (no existing analog; use this exact shape):
```python
def _stream_stderr(stderr_pipe) -> None:
    """Drain stderr line-by-line on a daemon thread; emit to logger at DEBUG."""
    for raw_line in iter(stderr_pipe.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("[rapidock stderr] %s", line)

def run_sampling(config: DockConfig) -> list[Path]:
    ...
    shim_path = Path(__file__).resolve().parent / "run_rapidock.py"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "rapidock-env",
        "python", str(shim_path),
        "--peptide", config.peptide_sequence,
        "--receptor", str(config.receptor_path.resolve()),
        "--output-dir", str((config.output_dir / "poses_raw").resolve()),
        "--n-samples", str(config.n_samples),
        "--rapidock-dir", str(_find_rapidock_dir()),
        "--model-dir", str(_model_dir().resolve()),
        "--ckpt", _ckpt_name(),
        "--scoring-function", "confidence",
    ]
    if config.seed is not None:
        cmd += ["--seed", str(config.seed)]

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    t = threading.Thread(target=_stream_stderr, args=(proc.stderr,), daemon=True)
    t.start()

    for raw_line in iter(proc.stdout.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("[rapidock stdout] %s", line)

    proc.wait()
    t.join()

    if proc.returncode != 0:
        raise RuntimeError(
            f"RAPiDock subprocess exited with code {proc.returncode}"
        )
```

**Non-zero returncode error pattern** — from `src/hybridock_pep/prep/receptor.py` lines 82-88:
```python
if result.returncode != 0:
    raise PrepError(
        f"prepare_receptor4.py failed (exit {result.returncode}):\n{result.stderr}"
    )
```
Adapt for `rapidock_runner.py` as `RuntimeError` (D-03), not `PrepError`.

**File renaming after subprocess exits** — no existing analog; use glob + re:
```python
raw_dir = config.output_dir / "poses_raw" / "poses_raw"  # RAPiDock creates {output_dir}/{complex_name}/
poses_dir = config.output_dir / "poses"
poses_dir.mkdir(parents=True, exist_ok=True)

rank_files = sorted(
    raw_dir.glob("rank*.pdb"),
    key=lambda p: int(re.search(r"rank(\d+)", p.stem).group(1)),
)
renamed: list[Path] = []
for i, src in enumerate(rank_files):
    dst = poses_dir / f"pose_{i}.pdb"
    src.rename(dst)
    renamed.append(dst)
logger.info("Renamed %d rank*.pdb → pose_*.pdb in %s", len(renamed), poses_dir)
return renamed
```

**Pose count shortfall pattern** — D-09/D-11; no existing analog:
```python
if len(renamed) == 0:
    raise RuntimeError(
        f"RAPiDock produced 0 poses in {raw_dir}. Check stderr logs above."
    )
if len(renamed) < config.n_samples:
    logger.warning(
        "RAPiDock pose shortfall: requested %d, generated %d",
        config.n_samples,
        len(renamed),
    )
```

---

### `src/hybridock_pep/sampling/run_rapidock.py` (utility, request-response)

**No direct analog.** This file is a Python 3.9 shim executed inside `rapidock-env`. It has unique constraints that no existing file in the codebase shares.

**CRITICAL CONSTRAINTS (D-06):**
- No `match`/`case`, no `X | Y` type unions, no walrus operator in comprehensions, no `TypeAlias`
- Use `Optional[X]` instead of `X | None`
- Use `Union[A, B]` instead of `A | B`
- `from __future__ import annotations` is safe to include (Python 3.7+ compatible)

**Python 3.9-compatible imports pattern** — from RESEARCH.md example:
```python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional
```

**Seed propagation pattern** — from RESEARCH.md (must happen BEFORE any torch/numpy call):
```python
def _seed_everything(seed: int) -> None:
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
```

**RAPiDock invocation pattern** — from RESEARCH.md:
```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rapidock-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scoring-function", default="confidence")
    args = parser.parse_args()

    if args.seed is not None:
        _seed_everything(args.seed)

    rapidock_dir = str(Path(args.rapidock_dir).resolve())
    if rapidock_dir not in sys.path:
        sys.path.insert(0, rapidock_dir)

    from utils.inference_parsing import get_parser as rd_get_parser
    import inference as rd_inference

    rd_args = rd_get_parser().parse_args([])
    rd_args.protein_description = args.receptor
    rd_args.peptide_description = args.peptide
    rd_args.output_dir = args.output_dir
    rd_args.complex_name = "poses_raw"
    rd_args.N = args.n_samples
    rd_args.model_dir = args.model_dir
    rd_args.ckpt = args.ckpt
    rd_args.scoring_function = args.scoring_function
    rd_args.fastrelax = False       # CLAUDE.md §2.5 — skip PyRosetta relax
    rd_args.save_visualisation = False
    rd_args.config = None

    rd_inference.main(rd_args)

if __name__ == "__main__":
    main()
```

**Absolute path resolution across conda boundary** — CLAUDE.md §7 and D-07:
```python
# All paths must be resolved to absolute before use
receptor_abs = str(Path(args.receptor).resolve())
output_dir_abs = str(Path(args.output_dir).resolve())
```

---

### `src/hybridock_pep/sampling/pose_io.py` (service, batch + file-I/O)

**Analog:** `src/hybridock_pep/prep/ligand.py` — exact match for collect-all-failures batch pattern.

**Imports pattern** — from `src/hybridock_pep/prep/ligand.py` lines 1-10, adapted:
```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hybridock_pep.models import PoseFailure, PoseRecord

logger = logging.getLogger(__name__)
```

**Collect-all-failures batch function signature** — from `src/hybridock_pep/prep/ligand.py` lines 61-86:
```python
def parse_poses(
    poses_dir: Path,
) -> tuple[list[PoseRecord], list[PoseFailure]]:
    """Parse all pose_*.pdb files in poses_dir into PoseRecord objects.

    All poses are processed regardless of individual failures. Failures are
    collected into PoseFailure records and returned alongside successes. The
    caller decides how many failures are acceptable — this function never
    raises on per-pose parse errors (D-12).

    Args:
        poses_dir: Directory containing pose_0.pdb ... pose_N.pdb files.

    Returns:
        Tuple of (records, failures) where:
        - records: Successfully parsed PoseRecord objects with ca_coords populated.
        - failures: PoseFailure records for any file that could not be parsed.
    """
```

**Per-item success/failure isolation pattern** — from `src/hybridock_pep/prep/ligand.py` lines 95-117:
```python
successes: list[PoseRecord] = []
failures: list[PoseFailure] = []

pdb_files = sorted(poses_dir.glob("pose_*.pdb"))
logger.info("Parsing %d pose PDB files from %s", len(pdb_files), poses_dir)

for pdb_path in pdb_files:
    pose_idx = int(pdb_path.stem.split("_")[1])
    try:
        record = _parse_single_pose(pose_idx, pdb_path)
        successes.append(record)
    except Exception as e:  # noqa: BLE001
        failures.append(PoseFailure(
            pose_idx=pose_idx,
            stage="parsing",
            error_msg=f"{type(e).__name__}: {e}",
        ))
        logger.warning("Pose %d parse failed: %s", pose_idx, e)

logger.info(
    "Pose parsing complete: %d succeeded, %d failed", len(successes), len(failures)
)
return successes, failures
```

Note: `ligand.py` uses `ProcessPoolExecutor`; `pose_io.py` does NOT — parsing is sequential (no subprocess, no heavy CPU work per pose). This is the key structural difference.

**PoseRecord construction with ca_coords** — D-13, no existing analog for Biopython PDB parsing; use this shape:
```python
def _parse_single_pose(pose_idx: int, pdb_path: Path) -> PoseRecord:
    """Parse one PDB file into a PoseRecord with Cα coordinates.

    Raises:
        ValueError: If no Cα atoms found or sequence unparseable (D-14).
    """
    from Bio.PDB import PDBParser  # local import — optional dep, cleaner error

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(f"pose_{pose_idx}", str(pdb_path))

    ca_coords_list = []
    residues_from_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords_list.append(residue["CA"].get_vector().get_array())
                    residues_from_atoms.append(residue.resname)

    if not ca_coords_list:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    ca_coords = np.array(ca_coords_list, dtype=np.float64)

    # Sequence: try SEQRES first (D-14), fallback to residue names from ATOM records
    sequence = _extract_sequence(pdb_path, fallback_residues=residues_from_atoms)

    return PoseRecord(
        pose_idx=pose_idx,
        pdb_path=pdb_path.resolve(),
        sequence=sequence,
        ca_coords=ca_coords,
    )
```

**Warning + continue on failure pattern** — from `src/hybridock_pep/prep/ligand.py` line 111:
```python
logger.warning("Pose %d prep failed: %s", result.pose_idx, result.error_msg)
```

---

### `src/hybridock_pep/output/metadata.py` (utility, file-I/O)

**Analog:** `src/hybridock_pep/scoring/vina.py` `_append_clipped_pose` (lines 77-104) — JSON read-modify-write + atomic replace pattern.

**Atomic JSON write pattern** — from `src/hybridock_pep/scoring/vina.py` lines 89-104:
```python
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2))
os.replace(tmp, path)  # atomic on POSIX; overwrites destination
```

**JSON read-modify-write pattern** — from `src/hybridock_pep/scoring/vina.py` lines 88-104:
```python
if not path.exists():
    data: dict = {}
else:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        data = {}
```

**Imports pattern** — from `src/hybridock_pep/scoring/vina.py` lines 14-29, adapted:
```python
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hybridock_pep.models import DockConfig

logger = logging.getLogger(__name__)
```

**Two-write pattern** — D-15; no existing analog; implement as two distinct functions:
```python
def write_metadata_skeleton(config: DockConfig, metadata_path: Path) -> None:
    """Write initial run_metadata.json with status='running' before sampling starts.

    Args:
        config: Validated DockConfig for this run.
        metadata_path: Absolute path to write the JSON file.
    """
    data = {
        "status": "running",
        "timestamp_start": datetime.now(tz=timezone.utc).isoformat(),
        "poses_requested": config.n_samples,
        "seed": config.seed,
        "cli_args": config.model_dump(),
        "git_sha": _get_git_sha(),
        "rapidock_commit_sha": _get_rapidock_sha(),
        "receptor_sha256": _sha256_file(config.receptor_path),
        "peptide_sequence_hash": hashlib.sha256(
            config.peptide_sequence.encode()
        ).hexdigest(),
        "vina_version": _get_vina_version(),
        "openmm_version": _get_openmm_version(),
        "cuda_version": None,  # filled from rapidock-env query
    }
    _write_json_atomic(metadata_path, data)
    logger.info("Metadata skeleton written: %s", metadata_path)


def finalize_metadata(
    metadata_path: Path,
    poses_generated: int,
    cuda_version: str | None = None,
) -> None:
    """Overwrite run_metadata.json with status='complete' and final counts.

    Args:
        metadata_path: Path to the existing skeleton JSON.
        poses_generated: Actual count of successfully parsed poses.
        cuda_version: CUDA version string from rapidock-env (may be None).
    """
    try:
        data = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        data = {}

    data["status"] = "complete"
    data["timestamp_end"] = datetime.now(tz=timezone.utc).isoformat()
    data["poses_generated"] = poses_generated
    if cuda_version is not None:
        data["cuda_version"] = cuda_version

    _write_json_atomic(metadata_path, data)
    logger.info(
        "Metadata finalized: %d/%d poses", poses_generated, data.get("poses_requested")
    )
```

**Version-query helper shape** — modelled on `_append_clipped_pose`'s subprocess-tolerant pattern:
```python
def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None

def _get_vina_version() -> str | None:
    try:
        result = subprocess.run(
            ["vina", "--version"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() or result.stderr.strip() or None
    except FileNotFoundError:
        return None
```

**Atomic write helper** — extracted from vina.py pattern:
```python
def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
```

---

### `tests/test_sampling.py` (test)

**Analog:** `tests/test_prep.py` (subprocess mock pattern, monkeypatch style) and `tests/test_scoring.py` (mock.patch style).

**File header + lazy import convention** — from `tests/test_scoring.py` lines 1-12:
```python
"""Tests for hybridock_pep.sampling — rapidock_runner and pose_io (SAMP-01)."""
from __future__ import annotations

import threading
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

# NOTE: All hybridock_pep imports are lazy (inside test functions) per STATE.md:
# "All hybridock_pep imports kept lazy in test files."

FIXTURES_DIR = Path(__file__).parent / "fixtures"
```

**Subprocess mock pattern (monkeypatch)** — from `tests/test_prep.py` lines 825-849 (TestReceptorPrep):
```python
def test_run_sampling_logs_command(self, config, monkeypatch, tmp_path: Path) -> None:
    """run_sampling logs the full conda run command before launching."""
    from hybridock_pep.sampling.rapidock_runner import run_sampling

    log_messages: list[str] = []

    def fake_popen(cmd, **kwargs):
        log_messages.append(" ".join(cmd))
        proc = mock.MagicMock()
        proc.stdout = iter([])   # empty — no lines
        proc.stderr = iter([])
        proc.returncode = 0
        proc.wait.return_value = 0
        return proc

    monkeypatch.setattr("hybridock_pep.sampling.rapidock_runner.subprocess.Popen", fake_popen)
    # ... create pose files, call run_sampling, assert "conda run" in log_messages[0]
```

**Class structure convention** — from `tests/test_prep.py` (TestReceptorPrep, TestLigandBatch, TestGrids):
```python
class TestRapidockRunner:
    """SAMP-01: subprocess orchestration, streaming, renaming."""
    ...

class TestPoseIO:
    """SAMP-01: PDB parsing → PoseRecord, collect-all-failures semantics."""
    ...
```

**Fixture convention** — from `tests/test_prep.py` lines 133-145:
```python
@pytest.fixture()
def config(self, tmp_path: Path):
    from hybridock_pep.models import DockConfig
    receptor = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
    return DockConfig(
        peptide_sequence="LISDAELEAIFEADC",
        receptor_path=receptor,
        site_coords=(22.5, 14.1, 38.7),
        box_size=20.0,
        output_dir=tmp_path / "out",
    )
```

**Source-inspection convention** — from `tests/test_prep.py` lines 299-310:
```python
def test_uses_popen_not_run(self) -> None:
    """Source of rapidock_runner.py must use subprocess.Popen, not subprocess.run (D-01)."""
    source = (
        Path(__file__).parent.parent
        / "src" / "hybridock_pep" / "sampling" / "rapidock_runner.py"
    )
    content = source.read_text()
    assert "subprocess.Popen" in content
    assert "subprocess.run" not in content
```

**Key test cases required for rapidock_runner:**
- Non-zero returncode → `RuntimeError` (D-03)
- Zero poses after subprocess → `RuntimeError` (D-11)
- Shortfall (fewer than n_samples) → warning logged, no exception (D-09)
- `conda run --no-capture-output` in command (D-04)
- All paths in command are absolute (D-07)
- Seed flag present when `config.seed` is set (D-08)
- No seed flag when `config.seed` is None (D-08)

**Key test cases required for pose_io:**
- Returns `(list[PoseRecord], list[PoseFailure])` — never raises
- Malformed PDB → `PoseFailure(stage="parsing")`
- `PoseRecord.ca_coords` shape is `(n_residues, 3)` float64 (D-13)
- `len(records) + len(failures) == len(glob("pose_*.pdb"))`

---

### `tests/test_output.py` (test)

**Analog:** `tests/test_scoring.py` lines 157-197 (JSON write + read-back tests for `_append_clipped_pose`).

**File header** — from `tests/test_scoring.py` lines 1-12:
```python
"""Tests for hybridock_pep.output.metadata (SAMP-02)."""
from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
```

**JSON write + read-back pattern** — from `tests/test_scoring.py` lines 185-197:
```python
def test_write_metadata_skeleton_creates_file(self, tmp_path: Path) -> None:
    from hybridock_pep.output.metadata import write_metadata_skeleton

    metadata_path = tmp_path / "run_metadata.json"
    # ... build config, call write_metadata_skeleton ...
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text())
    assert data["status"] == "running"
```

**Class structure:**
```python
class TestMetadataWriter:
    """SAMP-02: run_metadata.json — skeleton + finalize two-write pattern."""

    @pytest.fixture()
    def config(self, tmp_path: Path): ...

    def test_skeleton_status_is_running(self, ...) -> None: ...
    def test_skeleton_has_required_fields(self, ...) -> None: ...
    def test_finalize_status_is_complete(self, ...) -> None: ...
    def test_finalize_adds_timestamp_end(self, ...) -> None: ...
    def test_finalize_records_poses_generated(self, ...) -> None: ...
    def test_crash_leaves_running_status(self, ...) -> None: ...
    def test_atomic_write_uses_tmp_suffix(self, ...) -> None: ...
    def test_all_required_fields_present(self, ...) -> None: ...
```

**Required fields test pattern** — from `tests/test_scoring.py` lines 193-197:
```python
def test_all_required_fields_present(self, tmp_path: Path) -> None:
    """Skeleton must contain all 14 fields from D-16."""
    from hybridock_pep.output.metadata import write_metadata_skeleton

    # ... call write_metadata_skeleton, read JSON ...
    required = {
        "git_sha", "rapidock_commit_sha", "cli_args", "seed",
        "vina_version", "openmm_version", "cuda_version",
        "receptor_sha256", "peptide_sequence_hash",
        "timestamp_start", "poses_requested", "status",
    }
    assert required.issubset(data.keys()), f"Missing fields: {required - data.keys()}"
```

---

## Shared Patterns

### `from __future__ import annotations` first line
**Source:** Every existing module in `src/hybridock_pep/`
**Apply to:** `rapidock_runner.py`, `pose_io.py`, `metadata.py`, `run_rapidock.py` (Python 3.9 safe)
**Note:** In `run_rapidock.py`, this is the ONLY 3.10+ feature that is safe to use — it was backported to 3.7.

### Logger declaration
**Source:** `src/hybridock_pep/prep/receptor.py` line 14, `src/hybridock_pep/scoring/vina.py` line 28
**Apply to:** `rapidock_runner.py`, `pose_io.py`, `metadata.py`
```python
logger = logging.getLogger(__name__)
```

### Log-before-subprocess
**Source:** `src/hybridock_pep/prep/receptor.py` line 73
**Apply to:** `rapidock_runner.py` (before `Popen`), `metadata.py` (before version-query subprocesses)
```python
logger.info("Running: %s", " ".join(cmd))
```

### No bare `except:`
**Source:** `src/hybridock_pep/prep/ligand.py` line 51 (comment: `# noqa: BLE001 — Meeko raises varied internal errors`)
**Apply to:** All per-pose loops in `pose_io.py`; per-subprocess calls in `metadata.py`
```python
except Exception as e:  # noqa: BLE001
    # reason: [explain what varied errors are possible]
```

### Collect-all-failures return signature
**Source:** `src/hybridock_pep/prep/ligand.py` lines 61-86 (`prepare_ligand_batch`)
**Apply to:** `pose_io.parse_poses()`
```python
) -> tuple[list[PoseRecord], list[PoseFailure]]:
```

### Absolute path resolution (conda boundary)
**Source:** CLAUDE.md §7; D-07
**Apply to:** `rapidock_runner.py` (all paths in cmd list), `run_rapidock.py` (receptor, output_dir)
```python
str(Path(some_path).resolve())
```

### Atomic JSON write
**Source:** `src/hybridock_pep/scoring/vina.py` lines 101-104
**Apply to:** `metadata.py` `_write_json_atomic()` helper
```python
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2))
os.replace(tmp, path)
```

### Google-style docstrings with Args/Returns/Raises
**Source:** `src/hybridock_pep/prep/receptor.py` lines 18-39, `src/hybridock_pep/prep/ligand.py` lines 61-86
**Apply to:** All public functions in all four source files

### Lazy imports in test files
**Source:** `tests/test_scoring.py` lines 1-5 (docstring note); applied throughout both test files
**Apply to:** `tests/test_sampling.py`, `tests/test_output.py`
All `from hybridock_pep.*` imports inside test methods/fixtures, not at module top level.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/hybridock_pep/sampling/run_rapidock.py` | utility | request-response | Unique Python 3.9 shim executed inside a different conda env. No other file in the codebase crosses the conda env boundary or must be 3.9-compatible. Patterns come from RESEARCH.md + D-06 constraints. |

---

## Metadata

**Analog search scope:** `src/hybridock_pep/prep/`, `src/hybridock_pep/scoring/`, `tests/`
**Files read:** `receptor.py`, `ligand.py`, `vina.py`, `models.py`, `test_prep.py`, `test_scoring.py`
**Pattern extraction date:** 2026-04-21
