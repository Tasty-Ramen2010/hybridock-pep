# Architecture Research: HybriDock-Pep

**Researched:** 2026-04-18
**Confidence:** HIGH (core structural decisions already established; research validates and fills detail)

---

## Summary

HybriDock-Pep is a two-stage pipeline: stochastic ML pose generation (RAPiDock, GPU, separate
conda env) followed by physics-based rescoring (Vina + AD4 + entropy correction, CPU, score-env).
The boundary between stages is **file-based**: 100 PDB files written to `poses/` by Stage 1,
consumed by Stage 2. Everything within Stage 2 is in-memory after the PDB files are parsed.

The driver pattern is a thin orchestrator in `score-env` that:
1. Validates inputs before spawning anything.
2. Spawns RAPiDock via `conda run -n rapidock-env python rapidock_runner.py ...` and blocks.
3. Collects the written PDB files.
4. Fans out per-pose scoring in parallel via `concurrent.futures.ProcessPoolExecutor`.
5. Collects results, tolerating partial failures, then runs clustering and output.

The proposed module boundary in `CLAUDE.md` (`prep/`, `sampling/`, `scoring/`, `analysis/`,
`output/`) is sensible and maps cleanly onto the data flow. No restructuring recommended.

---

## Component Map

### `cli.py` — Entry Point and Input Validation

**Responsibility:** Argparse subcommands (`dock`, `calibrate`, `benchmark`, `prep`). Validates
all inputs before any subprocess is spawned — receptor PDB exists and is readable, peptide
sequence contains only standard amino acids, box dimensions are positive, output directory is
writable. Any validation failure exits with a clear message before any compute runs.

**Boundary:** Produces a validated `DockConfig` dataclass. Calls `driver.run(config)`.
Knows nothing about file formats, processes, or scoring math.

**Talks to:** `driver.py` only.

---

### `driver.py` — Orchestrator

**Responsibility:** Orchestrates the two-env boundary. Calls `prep/`, launches the RAPiDock
subprocess, calls `scoring/` and `analysis/`, writes output via `output/`. Logs all subprocess
commands and timing. Writes `run_metadata.json` at the end.

**Boundary:** Receives a `DockConfig`. The only module that touches subprocess and `conda run`.
Imports from `prep/`, `sampling/`, `scoring/`, `analysis/`, `output/` but nothing imports back
into `driver.py` (one-way dependency).

**Talks to:** All other modules as a coordinator.

---

### `prep/` — Receptor and Ligand Preparation Wrappers

**receptor.py:** Wraps `prepare_receptor4.py` (ADFRsuite) to produce the receptor PDBQT.
Runs once per job. Checks if `receptor.pdbqt` already exists (caches to avoid re-running on
repeated calls against the same receptor).

**ligand.py:** Wraps Meeko's `mk_prepare_ligand` per pose PDB. Accepts a pose PDB, writes
a pose PDBQT. Called inside the per-pose worker function. Must be side-effect-free and
stateless — it will be called concurrently from multiple worker processes.

**grids.py:** Wraps `autogrid4` to pre-compute AD4 affinity maps. Runs once per receptor,
before the scoring fan-out. Output is the `.e`, `.d`, `.map` files in the grid output directory.
Blocking, not parallelized.

**Boundary:** All functions take file paths in, write file paths out. No shared state.

---

### `sampling/` — RAPiDock Subprocess

**rapidock_runner.py:** The script that runs *inside* `rapidock-env`. Not imported by anything
in `score-env`. Called via `conda run -n rapidock-env python sampling/rapidock_runner.py`.
Accepts CLI args: `--peptide`, `--receptor`, `--n-samples`, `--seed`, `--output-dir`.
Writes `pose_000.pdb` through `pose_099.pdb` to the output dir. Returns exit code 0 on
success; non-zero on failure.

**pose_io.py:** In `score-env`. Parses pose PDB files, validates atom completeness (checks
for Cα presence for clustering; flags poses with missing heavy atoms). Provides `PoseBatch`
— a list of validated `PoseRecord` objects passed to the scoring fan-out.

**Boundary:**
- `rapidock_runner.py`: subprocess-only; never imported.
- `pose_io.py`: reads PDB files, emits `PoseRecord` dataclasses.

---

### `scoring/` — Per-Pose Physics Scoring

**vina.py:** Wraps `vina --score_only`. Takes a pose PDBQT path and receptor PDBQT path,
returns a `VinaScore` float. Uses AutoDock Vina's Python API (`from vina import Vina`) to
avoid subprocess overhead per pose — Vina 1.2.x exposes a Python binding that is faster than
100 subprocess calls. Falls back to subprocess if the binding is unavailable.

**ad4.py:** Wraps `vina --scoring ad4 --score_only`. Same interface as `vina.py`. Requires
the AD4 affinity maps (from `grids.py`) to be pre-computed and passed in as a path.

**entropy.py:** Pure Python. Accepts a `PoseRecord` (has backbone torsion angles), computes
the backbone entropy correction: `ΔS = α × Σ S_i` per residue. Returns a float. Stateless.
α is loaded from `calibration.json` at driver startup and passed in — no global state.

**mmgbsa.py:** Optional. Accepts a pose PDB, receptor PDB, and `OpenMM` system setup
parameters. Returns a `MMGBSAResult` dataclass with `delta_g_kcal_mol` and
`minimized_pose_pdb_path`. Runs OpenMM minimization before scoring (replacing the skipped
PyRosetta relax step, per §16.1). Called only for top-K poses after the initial ranking.

**Boundary:** All scoring functions are pure (path in, score out). No shared state between
calls. Safe to call from worker processes.

---

### `analysis/` — Clustering and Statistics

**clustering.py:** Accepts a list of `ScoredPose` objects (has `pose_path` for Cα loading).
Builds a pairwise Cα RMSD matrix using MDAnalysis or numpy. Feeds condensed distance matrix
to `scipy.cluster.hierarchy.linkage` (average linkage, well-validated for molecular structures
per clusttraj literature). Calls `sklearn.cluster.AgglomerativeClustering` with a precomputed
RMSD cutoff of 2.0 Å. Returns cluster labels and centroid indices.

**statistics.py:** Computes per-cluster hybrid score statistics (mean, min, std), ensemble
convergence metric (score variance vs pose index), and rank correlation between Vina and AD4
scores as a quality signal.

**plotting.py:** matplotlib only. Produces `convergence.png` (running mean of hybrid score
vs N) and `dendrogram.png` from the linkage matrix. No logic — purely visualization.

**Boundary:** `clustering.py` is the only module that reads PDB files at this stage (for Cα
coordinates). Everything else operates on the `ScoredPose` list.

---

### `output/` — Result Serialization

**csv_writer.py:** Writes `ranked_poses.csv` and `cluster_summary.csv` from `ScoredPose`
list and cluster results. Uses Python's standard `csv` module (no pandas dependency — keeps
`score-env` lean).

**metadata.py:** Collects git SHA, RAPiDock commit SHA, all CLI args, random seeds,
software versions (Vina, OpenMM, CUDA detected at runtime), receptor SHA256, peptide
sequence hash, wallclock time. Writes `run_metadata.json`.

**Boundary:** Pure serialization. No computation.

---

## Data Flow

```
CLI args
    │
    ▼
cli.py: validate → DockConfig
    │
    ▼
driver.py
    │
    ├─► prep/receptor.py ──────────────────► receptor.pdbqt  (once)
    │
    ├─► prep/grids.py ─────────────────────► AD4 affinity maps  (once, before fan-out)
    │
    ├─► conda run rapidock-env
    │       sampling/rapidock_runner.py
    │       writes: poses/pose_{000..099}.pdb
    │
    ├─► sampling/pose_io.py
    │       reads: poses/*.pdb
    │       emits: List[PoseRecord]
    │
    ├─► ProcessPoolExecutor (one worker per pose, max_workers=cpu_count())
    │       per-pose worker:
    │         prep/ligand.py          → pose.pdbqt
    │         scoring/vina.py         → VinaScore
    │         scoring/ad4.py          → AD4Score
    │         scoring/entropy.py      → EntropyCorrection
    │       returns: ScoredPose or PoseFailure
    │
    ├─► driver: collect results, log failures, proceed with passing poses
    │
    ├─► [optional] scoring/mmgbsa.py on top-K
    │       reads: pose PDB, receptor PDB
    │       returns: MMGBSAResult
    │
    ├─► analysis/clustering.py
    │       reads: Cα coordinates from ScoredPose paths
    │       returns: cluster labels, centroids
    │
    ├─► analysis/statistics.py  → ensemble stats
    ├─► analysis/plotting.py    → convergence.png, dendrogram.png
    │
    └─► output/csv_writer.py    → ranked_poses.csv, cluster_summary.csv
        output/metadata.py      → run_metadata.json
        driver: copy best centroid → best_pose.pdb
```

**Interface nature summary:**

| Stage boundary | Interface type | Rationale |
|---|---|---|
| cli.py → driver.py | In-memory (`DockConfig` dataclass) | Same process, same env |
| driver → RAPiDock subprocess | File-based (pose PDBs in `poses/`) | Cross-env boundary; file is the only reliable IPC |
| pose_io → scoring fan-out | In-memory (`PoseRecord` dataclass) | Same env; avoids re-reading 100 PDBs |
| scoring workers → driver | In-memory (Future result, `ScoredPose` or exception) | ProcessPoolExecutor standard |
| scoring → analysis | In-memory (`List[ScoredPose]`) | Sorting and filtering before clustering |
| analysis → output | In-memory (cluster assignments, stats) | Serialized once at end |
| driver → output | File-based (final CSVs, JSON, PDB) | Persistent results |

---

## Interface Contracts

### `DockConfig` (dataclass, `cli.py`)

```python
@dataclass(frozen=True)
class DockConfig:
    peptide: str                  # validated amino acid sequence
    receptor_pdb: Path
    site_center: tuple[float, float, float]
    box_size: float               # angstroms
    n_samples: int                # default 100
    scoring_modes: list[str]      # ["vina", "ad4"]
    refine_topk: int | None       # None = skip MM-GBSA
    output_dir: Path
    seed: int
    calibration_json: Path
```

### `PoseRecord` (dataclass, `sampling/pose_io.py`)

```python
@dataclass
class PoseRecord:
    pose_index: int
    pdb_path: Path
    is_valid: bool                # False if Cα missing or atom count anomalous
    validation_warnings: list[str]
```

### `ScoredPose` (dataclass, `scoring/`)

```python
@dataclass
class ScoredPose:
    pose_index: int
    pdb_path: Path
    pdbqt_path: Path
    vina_score: float | None      # None if scoring failed
    ad4_score: float | None
    entropy_correction: float
    hybrid_score: float           # vina + ad4_weight*ad4 + entropy_correction
    mmgbsa_delta_g: float | None  # populated only for top-K
```

### `PoseFailure` (dataclass, `scoring/`)

```python
@dataclass
class PoseFailure:
    pose_index: int
    pdb_path: Path
    stage: str                    # "prep", "vina", "ad4", "entropy"
    error: str                    # str(exception)
```

### File formats at the Stage 1 / Stage 2 boundary

- `poses/pose_{i:03d}.pdb`: standard all-atom PDB from RAPiDock. Must have `ATOM` records with
  Cα atoms for every residue. PULCHRA v3.04 is expected to have already reconstructed side chains.
- `receptor.pdbqt`: standard AutoDock PDBQT. Generated once by `prep/receptor.py`.
- AD4 grid maps: `.e`, `.d`, `.C.map`, `.N.map`, etc. in `grids/` subdir. Named per ADFRsuite
  convention; `ad4.py` constructs the path from the receptor name and grid dir.

---

## Build Order

Dependencies flow upward — lower layers must exist before upper layers can be tested.

```
Layer 0 (no deps):
  prep/receptor.py   — wraps ADFRsuite binary, testable with fixture PDB
  prep/ligand.py     — wraps Meeko, testable with fixture pose PDB
  prep/grids.py      — wraps autogrid4, testable with receptor PDBQT fixture

Layer 1 (depends on Layer 0):
  scoring/entropy.py     — pure Python, no binary dep, build first
  scoring/vina.py        — needs receptor.pdbqt + ligand.pdbqt from Layer 0
  scoring/ad4.py         — needs grid maps from Layer 0 + ligand.pdbqt

Layer 2 (depends on Layer 1):
  sampling/pose_io.py    — pure PDB parsing, can build early (no env dep)
  sampling/rapidock_runner.py  — needs GPU env wired; build last of sampling

Layer 3 (depends on Layer 1 + 2):
  scoring/mmgbsa.py      — needs OpenMM, most complex; build after Vina/AD4 working

Layer 4 (depends on all scoring):
  analysis/clustering.py
  analysis/statistics.py
  analysis/plotting.py

Layer 5 (depends on Layer 4):
  output/csv_writer.py
  output/metadata.py

Layer 6 (depends on all):
  driver.py
  cli.py
```

**Practical build sequence for phases:**
1. `prep/` modules (receptor, ligand, grids) — establishes file format contracts
2. `scoring/entropy.py` — pure logic, validates calibration math early
3. `scoring/vina.py` + `scoring/ad4.py` — core physics signal
4. `sampling/pose_io.py` — PDB validation
5. `analysis/clustering.py` + `analysis/statistics.py` — validates end-to-end math
6. `driver.py` skeleton with stubbed subprocess (for integration testing without GPU)
7. `sampling/rapidock_runner.py` — GPU env wiring, validates Blackwell compatibility
8. `scoring/mmgbsa.py` — optional feature, build last
9. `cli.py` subcommands and `output/` serialization

---

## Subprocess Orchestration Patterns

### The `conda run` process ownership problem

Known issue (conda/conda#12894): when `conda run` spawns a process, the spawned process does
NOT have `conda run` as its parent. Calling `.kill()` on the Python subprocess object kills
`conda run` but leaves the actual RAPiDock inference process running orphaned.

**Recommended pattern for `driver.py`:**

```python
import subprocess, sys, shutil

def run_rapidock(config: DockConfig, poses_dir: Path) -> None:
    conda_exe = shutil.which("conda") or "conda"
    cmd = [
        conda_exe, "run", "--no-capture-output", "-n", "rapidock-env",
        sys.executable,                          # not "python" — use full path
        str(Path(__file__).parent / "sampling" / "rapidock_runner.py"),
        "--peptide", config.peptide,
        "--receptor", str(config.receptor_pdb),
        "--n-samples", str(config.n_samples),
        "--seed", str(config.seed),
        "--output-dir", str(poses_dir),
    ]
    log.info("RAPiDock command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=False,          # stream stdout/stderr in real-time
        timeout=600,                   # 10-minute hard cap; tune after benchmarking
        check=False,                   # inspect returncode ourselves
    )
    if result.returncode != 0:
        raise RAPiDockError(
            f"RAPiDock exited with code {result.returncode}. "
            f"Check logs above for details."
        )
```

Use `--no-capture-output` on `conda run` so RAPiDock's stdout/stderr stream directly to the
driver's terminal — this is the only way to see GPU OOM errors or CUDA init failures in real
time without a separate reader thread.

Use `subprocess.run(..., timeout=...)` rather than `Popen` for Stage 1. RAPiDock is a
blocking, serial step — no need for async management. If it hangs, the timeout raises
`subprocess.TimeoutExpired`, which driver catches and converts to a user-facing error.

Do NOT use `subprocess.run(..., shell=True)` — shell injection risk and PATH issues across
Linux/macOS/WSL2.

### Vina/ADFRsuite subprocess calls

These are single-pose, fast (< 1 s each) calls wrapped in worker processes. Use the
**Vina Python API** (`from vina import Vina`) rather than subprocess for Vina 1.2.x — the
Python binding avoids fork+exec overhead × 100. If the binding is unavailable (fallback),
use `subprocess.run(["vina", "--score_only", ...], capture_output=True, timeout=30)`.

For `autogrid4` (runs once, pre-fan-out), use `subprocess.run` with `check=True` and
`timeout=120`.

---

## Parallelization Strategy

### Approach: `concurrent.futures.ProcessPoolExecutor`

Per-pose scoring is CPU-bound (Vina Python API, Meeko preparation, entropy math). Use
`ProcessPoolExecutor` — not `ThreadPoolExecutor` (GIL contention) and not raw
`multiprocessing.Pool` (less ergonomic exception handling).

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple

def score_all_poses(
    pose_records: list[PoseRecord],
    receptor_pdbqt: Path,
    grid_dir: Path,
    alpha: float,
    max_workers: int | None = None,   # None = os.cpu_count()
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    scored: list[ScoredPose] = []
    failed: list[PoseFailure] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pose = {
            executor.submit(
                _score_single_pose,   # module-level function, picklable
                pose, receptor_pdbqt, grid_dir, alpha
            ): pose
            for pose in pose_records
            if pose.is_valid
        }
        for future in as_completed(future_to_pose):
            pose = future_to_pose[future]
            exc = future.exception()
            if exc is not None:
                log.warning("Pose %d failed: %s", pose.pose_index, exc)
                failed.append(PoseFailure(
                    pose_index=pose.pose_index,
                    pdb_path=pose.pdb_path,
                    stage=_infer_stage(exc),
                    error=str(exc),
                ))
            else:
                scored.append(future.result())

    return scored, failed
```

**Critical:** `_score_single_pose` must be a **module-level function** (not a lambda or
nested function) — ProcessPoolExecutor uses pickle for IPC, and closures do not pickle.

### Partial failure handling

The pipeline must proceed if some poses fail to score. Policy:

- Skip invalid poses before the fan-out (`pose.is_valid` check — catches PULCHRA atom issues).
- Log each failure at WARNING level with pose index, stage, and exception string.
- If `< 10` poses scored successfully out of 100, raise a `InsufficientPosesError` and stop
  — clustering on < 10 poses is not meaningful (this threshold matches the "cluster populations
  < 10 of 100" failure mode in CLAUDE.md §9).
- If `≥ 10` poses scored, continue with the subset and note the failure count in
  `run_metadata.json` under `"scoring_failures"`.
- Never silently drop failures. Every failure is logged and reported in the metadata.

### `BrokenProcessPool` guard

If a worker process is killed by the OS (OOM, segfault in Vina), `ProcessPoolExecutor` raises
`BrokenProcessPool`. Catch it at the driver level:

```python
from concurrent.futures import BrokenExecutor

try:
    scored, failed = score_all_poses(...)
except BrokenExecutor as e:
    raise ScoringEnvironmentError(
        "A scoring worker process was killed unexpectedly. "
        "Check system memory. Details: " + str(e)
    ) from e
```

### Worker count

Default to `os.cpu_count()`. Document that on a machine with the RTX 5070, the CPU cores are
not occupied by Stage 1 (RAPiDock uses GPU), so all CPU cores are available for the Stage 2
fan-out. For the 5 min runtime target on 100 poses, with Vina Python API ≈ 0.2–0.5 s/pose
and a modern 8–16 core CPU, the fan-out budget is well within target.

### MM-GBSA parallelization

MM-GBSA runs on top-K (default 10) poses only. Run sequentially in the driver — OpenMM
itself uses multiple CPU threads internally (via OpenCL or CPU platform). Do not wrap in
ProcessPoolExecutor; there is no benefit and it complicates GPU/CPU resource management.

---

## Additional Patterns to Follow

### Pose worker functions must be stateless and path-based

Worker functions receive file paths, not open file handles or shared objects. This is
required for ProcessPoolExecutor pickling. Workers open files themselves, do their work,
close files, return a dataclass.

### Pre-fan-out invariant checks

Before launching the executor, assert:
- `receptor.pdbqt` exists and is non-empty.
- AD4 grid maps exist (if AD4 scoring enabled).
- `calibration.json` is loaded and `alpha` is in [0.2, 1.2] (the out-of-bounds failure mode
  from CLAUDE.md §9).
- At least 1 valid pose exists.

Catching these before the fan-out avoids wasting time spinning up workers for a doomed run.

### Intermediate result persistence

Write `poses/` PDB files permanently (not temp files) — this enables re-running Stage 2 on
existing poses without re-running the GPU inference. The driver should check for existing
`poses/pose_*.pdb` and skip Stage 1 if `--skip-sampling` is passed. This is the key
developer-iteration affordance for the scoring/clustering code.

### Entropy correction is not per-process state

Pass `alpha` into the worker function as a plain float. Do not read `calibration.json` inside
the worker (file I/O in every worker × 100 is unnecessary and a portability risk on WSL2
network filesystems).

---

## Sources

- [Integrating ML-Based Pose Sampling with Physics Scoring - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12117556/) — DiffDock-L + Vina pipeline architecture; file-based stage boundary pattern
- [DiffDock GitHub](https://github.com/gcorso/DiffDock) — module layout reference for ML docking tool
- [conda/conda #12894 — conda run process ownership issue](https://github.com/conda/conda/issues/12894) — `conda run` subprocess lifecycle warning
- [conda_subprocess PyPI](https://pypi.org/project/conda-subprocess/) — alternative pattern (not recommended here due to added dep; plain subprocess.run is sufficient for blocking Stage 1)
- [concurrent.futures Python docs](https://docs.python.org/3/library/concurrent.futures.html) — ProcessPoolExecutor, as_completed, BrokenExecutor
- [clusttraj - RMSD-based agglomerative clustering](https://github.com/hmcezar/clusttraj) — validated approach for molecular trajectory clustering with SciPy linkage
- [Autodock Vina Python scripting docs](https://autodock-vina.readthedocs.io/en/latest/docking_python.html) — Vina Python API for score_only mode
- [Meeko docs](https://meeko.readthedocs.io/) — PDBQT preparation from PDB/SDF
- [Eli Bendersky — Interacting with long-running child processes](https://eli.thegreenplace.net/2017/interacting-with-a-long-running-child-process-in-python/) — streaming subprocess stdout pattern
