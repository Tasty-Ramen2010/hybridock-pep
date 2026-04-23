# Phase 4: Sampling Integration - Research

**Researched:** 2026-04-21
**Domain:** RAPiDock subprocess orchestration, PDB pose parsing, provenance metadata
**Confidence:** HIGH (RAPiDock API verified from live source; PyG wheels verified from data.pyg.org)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** `rapidock_runner.py` uses `subprocess.Popen` with `readline()` loop on stdout; stderr on a separate daemon thread with its own `readline()` loop. Both emit to `logger` in real time.
**D-02:** Do NOT use `asyncio` subprocess or `communicate()` — both buffer output and fail the real-time OOM requirement.
**D-03:** Every line logged at DEBUG level. Non-zero returncode after `Popen.wait()` raises `RuntimeError` with the exit code.
**D-04:** `rapidock_runner.py` calls `conda run --no-capture-output -n rapidock-env python {abs_path_to_run_rapidock}` with args as CLI flags.
**D-05:** `run_rapidock.py` lives at `src/hybridock_pep/sampling/run_rapidock.py`. Its absolute path is resolved in score-env via `Path(__file__).resolve()` and passed to `conda run`.
**D-06 (CRITICAL):** `run_rapidock.py` must be strictly Python 3.9 compatible. No `match`/`case`, no `X | Y` unions, no walrus operator in comprehensions, no `TypeAlias`.
**D-07:** All file paths across the `conda run` boundary MUST be absolute paths resolved via `Path(...).resolve()`.
**D-08:** Seed passed as CLI arg `--seed N`. If `DockConfig.seed` is None, no seed flag is passed; non-determinism noted in `run_metadata.json`.
**D-09:** Fewer poses than `DockConfig.n_samples` → warn and continue. Do NOT abort.
**D-10:** `run_metadata.json` records `poses_requested` and `poses_generated`.
**D-11:** Zero poses generated → raise `RuntimeError`.
**D-12:** `pose_io.py` returns `(list[PoseRecord], list[PoseFailure])`. Batch never raises.
**D-13:** Cα coordinates extracted at parse time into `PoseRecord.ca_coords: np.ndarray` shape `[n_residues, 3]`.
**D-14:** Sequence from SEQRES records; fallback to residue names from ATOM records. Neither parseable → `PoseFailure`.
**D-15:** `run_metadata.json` written twice: skeleton at start (`status: "running"`), overwritten at completion (`status: "complete"`).
**D-16:** Required `run_metadata.json` fields: `git_sha`, `rapidock_commit_sha`, `cli_args`, `seed`, `vina_version`, `openmm_version`, `cuda_version`, `receptor_sha256`, `peptide_sequence_hash`, `timestamp_start`, `timestamp_end`, `poses_requested`, `poses_generated`, `status`.
**D-17:** `metadata.py` lives in `src/hybridock_pep/output/`.

### Claude's Discretion

- Exact argparse flag names in `run_rapidock.py`
- Whether to use `threading.Thread` or `concurrent.futures.ThreadPoolExecutor` for the stderr daemon thread
- Exact PDB ATOM record parsing logic (Biopython vs manual)
- `rapidock_commit_sha` discovery strategy

### Deferred Ideas (OUT OF SCOPE)

- Incremental per-pose metadata writes — two writes (start + end) only for v1
- Configurable `min_poses` threshold — deferred to v2
- GPU parallelism across multiple GPUs — explicitly out of scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SAMP-01 | Pipeline runs RAPiDock N=100 stochastic inference passes in `rapidock-env` via `conda run` subprocess; seed propagated for reproducibility | RAPiDock entry point is `inference.py`; invoked via `python inference.py --protein_description ... --peptide_description ... --output_dir ... --N 100`; seed requires `torch.manual_seed()` call in `run_rapidock.py` before calling inference |
| SAMP-02 | Every run writes `run_metadata.json` containing the 14 required fields | `direct_url.json` in dist-info is the canonical source for `rapidock_commit_sha`; all other fields are available from stdlib (hashlib, subprocess, importlib) |
</phase_requirements>

---

## Summary

Phase 4 delivers three modules: a subprocess driver (`rapidock_runner.py`) that calls RAPiDock inside `rapidock-env` via `conda run`, a thin Python 3.9 shim (`run_rapidock.py`) that imports and invokes RAPiDock's inference script, a PDB parser (`pose_io.py`) that produces `list[PoseRecord]`, and a metadata writer (`output/metadata.py`) that writes provenance JSON.

The RAPiDock API has been fully inspected from source [VERIFIED: github.com/huifengzhao/RAPiDock]. The entry point is the top-level `inference.py` script, which is invoked as a subprocess via `python inference.py [args]`. It is NOT a Python package with an importable API — there is no `rapidock.Inference()` class or `rapidock.run()` function. The `run_rapidock.py` shim must therefore call `inference.py` by running it in-process (via `exec()` or by importing its `main()` directly after adding the RAPiDock source directory to `sys.path`). The output naming convention is `rank{N}_{scoring_function}.pdb` (e.g., `rank1_ref2015.pdb`, `rank2_ref2015.pdb`), NOT `pose_{i}.pdb` — **the driver must rename these files after generation**.

Both environment compatibility blockers from STATE.md are resolved: PyG cu128 prebuilt wheels DO exist for PyTorch 2.7.0 and cp39 [VERIFIED: data.pyg.org/whl/torch-2.7.0+cu128.html]. fair-esm 2.0.0 uses only ESM2 embeddings (`esm2_t33_650M_UR50D`) — NOT ESMFold — so the "python <= 3.9 required for ESMFold" restriction does not apply; however, since the ESM repo was archived August 2024, PyTorch 2.7 compatibility must be treated as LOW confidence until verified at runtime.

**Primary recommendation:** Use `threading.Thread` for the stderr daemon (not `ThreadPoolExecutor`) and Biopython `PDBParser` for pose parsing (already in score-env); rename RAPiDock output files from `rank{N}_ref2015.pdb` to `pose_{N-1}.pdb` immediately after inference completes.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| RAPiDock inference (GPU diffusion) | rapidock-env subprocess | — | GPU/PyTorch stack incompatible with score-env; must cross conda boundary |
| Subprocess orchestration + streaming | score-env (rapidock_runner.py) | — | Driver lives in score-env; `conda run` bridges envs |
| Pose PDB parsing → PoseRecord | score-env (pose_io.py) | — | Reads files written by rapidock-env subprocess; no GPU needed |
| Provenance metadata write | score-env (output/metadata.py) | — | Reads DockConfig + sampling results; writes JSON |
| Seed propagation | run_rapidock.py (rapidock-env) | rapidock_runner.py passes --seed | Only in-process `torch.manual_seed()` can seed the inference RNG |
| Output file renaming (rank→pose) | rapidock_runner.py or run_rapidock.py | — | RAPiDock names files `rank{N}_{scoring_function}.pdb`; pipeline expects `pose_{i}.pdb` |

---

## RAPiDock API

### Entry Point (CRITICAL)

**Repository:** `https://github.com/huifengzhao/RAPiDock` [VERIFIED: fetched live source 2026-04-21]

RAPiDock has **no importable Python package API**. The code lives in a flat directory structure (`inference.py`, `utils/`, `dataset/`, `train_models/`). `pip install git+...` installs the package so its modules are importable, but **the `main()` function in `inference.py` is the only defined entry point**.

**How `run_rapidock.py` must call it:**

```python
# Option A (recommended): exec the script via subprocess — CLEANEST
# rapidock_runner.py calls: conda run -n rapidock-env python /abs/path/run_rapidock.py [args]
# run_rapidock.py then: finds inference.py via the installed package location,
# constructs an argparse Namespace, and calls inference.main(args)

import sys
import importlib.util

def _find_inference_py() -> str:
    """Locate inference.py from the pip-installed RAPiDock package."""
    import rapidock  # or whatever module name pip registered
    pkg_dir = Path(rapidock.__file__).parent
    candidate = pkg_dir.parent / "inference.py"  # depends on install layout
    ...
```

**IMPORTANT FINDING — Install Layout:** When `pip install git+https://github.com/huifengzhao/RAPiDock.git@SHA` is run, the package is installed with the repo root as the package root. RAPiDock has no `setup.py` or `pyproject.toml` visible in the repo — [ASSUMED] it may install as an editable or namespace package, or pip may simply clone the directory. The `run_rapidock.py` shim should locate `inference.py` by resolving the path relative to the RAPiDock package location OR by accepting `--rapidock-dir` as an explicit CLI arg passed from `rapidock_runner.py`. **Recommend the `--rapidock-dir` approach** (see Pattern 2 below) — it is explicit, testable, and avoids import magic.

[VERIFIED: github.com/huifengzhao/RAPiDock — no setup.py found in repo root]

### inference.py Key Parameters

[VERIFIED: fetched `utils/inference_parsing.py` and `inference.py` live from GitHub 2026-04-21]

| Argument | Type | Default | Notes |
|----------|------|---------|-------|
| `--protein_description` | str | None | Path to receptor PDB **OR** amino acid sequence |
| `--peptide_description` | str | None | Path to peptide PDB **OR** amino acid sequence |
| `--output_dir` | str | `"outputs/default_result"` | Directory where subdirs per complex are created |
| `--complex_name` | str | None | Name of output subdir; if None → `"complex_0"` |
| `--N` | int | None | Number of inference passes (poses to generate) |
| `--model_dir` | str | None | Path to `train_models/CGTensorProductEquivariantModel/` |
| `--ckpt` | str | None | Checkpoint filename e.g. `rapidock_local.pt` |
| `--scoring_function` | str | None | `"confidence"` or `"ref2015"` — controls output filename suffix |
| `--batch_size` | int | None | GPU batch size for inference |
| `--no_final_step_noise` | bool | False | Remove noise from final diffusion step |
| `--inference_steps` | int | None | Denoising steps (from model YAML: 16) |
| `--actual_steps` | int | None | Steps actually performed (default = inference_steps) |
| `--conformation_type` | str | `"H"` | Initial conformation: H(elix), E(xtended), P(olyproline) |
| `--conformation_partial` | str | None | Initial conformation split ratios e.g. `"1:1:1"` |
| `--fastrelax` | bool | False | Run PyRosetta FastRelax — **DO NOT use** (CLAUDE.md §2.5) |
| `--save_visualisation` | bool | False | Save reverse diffusion frames — not needed |
| `--cpu` | int | 5 | CPU cores for multiprocessing pool (ref2015 only) |
| `--config` | FileType | None | YAML config file — values overwrite CLI args |

**No `--seed` argument exists in RAPiDock.** [VERIFIED from `utils/inference_parsing.py`]

### Seed Propagation Strategy

RAPiDock has no `--seed` parameter and no `torch.manual_seed()` or `numpy.random.seed()` call anywhere in `inference.py` or `utils/sampling.py`. [VERIFIED: live source inspection]

The `sampling()` function signature is:
```python
def sampling(data_list, model, args, inference_steps=20,
             no_random=False, ode=False, visualization_list=None,
             confidence_model=None, batch_size=32,
             no_final_step_noise=False, actual_steps=None)
```

The `no_random=False` parameter suppresses noise when True, but is not wired to a seed.

**To implement seed propagation in `run_rapidock.py`:**
```python
if args.seed is not None:
    import torch
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
```

This must happen BEFORE any ESM embedding computation or graph construction — i.e., before `main(rapidock_args)` is called. Caveat: CUDA nondeterminism (cublas, cudnn) remains even with a fixed seed unless `torch.use_deterministic_algorithms(True)` is set — flag this in `run_metadata.json` per D-08. [CITED: pytorch.org/docs/stable/notes/randomness.html]

### Output File Naming Convention

[VERIFIED: `save_predictions()` function in `inference.py`]

RAPiDock writes files to `{output_dir}/{complex_name}/`:
```
rank1_ref2015.pdb
rank2_ref2015.pdb
...
rankN_ref2015.pdb
```

When `scoring_function="confidence"`:
```
rank1_confidence.pdb
...
```

When no confidence model and `fastrelax=False`:
```
rank1.pdb
rank2.pdb
...
```

**These are NOT named `pose_{i}.pdb`.** The pipeline MUST rename or re-index them. Options:

1. **In `run_rapidock.py` (inside rapidock-env):** After `main(rapidock_args)` completes, glob `rank*.pdb` and rename to `pose_{i}.pdb` where `i` is 0-indexed by rank order.
2. **In `rapidock_runner.py` (score-env):** After the conda subprocess exits, walk the output directory and rename.

**Recommended: rename in `rapidock_runner.py`** — keeps `run_rapidock.py` minimal and renaming logic in score-env where it can be properly unit-tested.

### Directory Structure RAPiDock Creates

```
{output_dir}/
└── {complex_name}/           # e.g. "complex_0" or whatever --complex_name specifies
    ├── rank1_ref2015.pdb
    ├── rank2_ref2015.pdb
    ...
    └── rankN_ref2015.pdb
    # + ref2015_score.csv if fastrelax=True (we do NOT use fastrelax)
    # + {complex_name}_protein_raw.pdb (copy of receptor)
    # + peptide ESM embedding .pt files
```

**For `run_rapidock.py`:** Pass `--complex_name pose_output` (or the run_id) and `--output_dir {output_dir}/poses_raw/`, then rename in `rapidock_runner.py` to `{output_dir}/poses/pose_{i}.pdb`.

### RAPiDock Invocation Example (as `run_rapidock.py` would call it)

```python
# Inside run_rapidock.py — Python 3.9, executed inside rapidock-env
import sys
import argparse
from pathlib import Path

# Locate inference.py from the installed RAPiDock location
# (passed as --rapidock-dir from rapidock_runner.py)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rapidock-dir", required=True,
                        help="Absolute path to RAPiDock install dir (contains inference.py)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scoring-function", default="confidence")
    args = parser.parse_args()

    # Seed BEFORE any torch/numpy use
    if args.seed is not None:
        import torch, numpy as np, random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Add RAPiDock root to sys.path so its utils/ and dataset/ are importable
    rapidock_dir = str(Path(args.rapidock_dir).resolve())
    if rapidock_dir not in sys.path:
        sys.path.insert(0, rapidock_dir)

    from utils.inference_parsing import get_parser as rd_get_parser
    import inference as rd_inference

    rd_args = rd_get_parser().parse_args([])  # defaults
    rd_args.protein_description = args.receptor
    rd_args.peptide_description = args.peptide
    rd_args.output_dir = args.output_dir
    rd_args.complex_name = "poses_raw"
    rd_args.N = args.n_samples
    rd_args.model_dir = args.model_dir
    rd_args.ckpt = args.ckpt
    rd_args.scoring_function = args.scoring_function
    rd_args.fastrelax = False
    rd_args.save_visualisation = False
    rd_args.config = None
    # ... other required fields from default_inference_args.yaml ...

    rd_inference.main(rd_args)
```

**Note:** RAPiDock's `get_parser()` creates an `ArgumentParser` with `fromfile_prefix_chars='@'` and a `--config` FileType. Calling `parse_args([])` gives defaults; then we override fields directly on the Namespace. This is simpler and safer than re-parsing a YAML.

---

## Environment Blockers

### Blocker 1: PyG cu128 Wheels for PyTorch 2.7.0

**STATUS: RESOLVED — wheels exist** [VERIFIED: https://data.pyg.org/whl/torch-2.7.0+cu128.html, 2026-04-21]

All five required packages are available:

| Package | Version | Python cp39? |
|---------|---------|-------------|
| `pyg_lib` | 0.4.0, 0.5.0 | YES |
| `torch_scatter` | 2.1.2 | YES |
| `torch_sparse` | 0.6.18 | YES |
| `torch_cluster` | 1.6.3 | YES |
| `torch_spline_conv` | 1.2.2 | YES |

**Install command for rapidock-env:**
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

Note: `rapidock-env.yml` already pins `pyg::pyg` via conda, which may install `torch-geometric` 2.x from the pyg channel. The above pip install adds the C++ extension wheels that `torch-geometric` depends on. **Verify after env creation that `import torch_scatter` succeeds inside rapidock-env.**

### Blocker 2: fair-esm 2.0.0 + PyTorch 2.7

**STATUS: LOW CONFIDENCE — unverified at runtime**

Key facts:
- fair-esm 2.0.0 was released November 2022; the repo was archived August 1, 2024 — read-only, no maintenance. [CITED: github.com/facebookresearch/esm]
- RAPiDock uses only ESM2 embeddings (`esm2_t33_650M_UR50D`) via `esm.pretrained.load_model_and_alphabet()`. It does NOT use ESMFold. [VERIFIED: `utils/inference_utils.py` InferenceDataset]
- The "python <= 3.9 required" constraint applies only to ESMFold, not ESM2 models. [CITED: facebookresearch/esm README]
- PyTorch 2.4+ deprecated `torch.cuda.amp.autocast` (FutureWarning only; functionally still works). fair-esm 2.0.0 uses the deprecated API internally. [VERIFIED: PyTorch 2.4 changelog; ASSUMED: fair-esm uses old AMP API — not verified against esm source]
- No breaking changes known for ESM2 inference under PyTorch 2.x — the model loading (`torch.load` + `load_state_dict`) and forward pass APIs are backward compatible. [ASSUMED]

**Expected outcome:** fair-esm 2.0.0 imports successfully under PyTorch 2.7 with at worst FutureWarning spam about `torch.cuda.amp.autocast`. No hard import failure expected. [ASSUMED: LOW confidence — must validate on day-1 of Phase 4 execution]

**Validation command (run inside rapidock-env before writing any sampling code):**
```bash
conda run -n rapidock-env python -c "
import esm
model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
print('ESM2 loaded OK, type:', type(model))
"
```

**Fallback if fair-esm fails:**
- Pin `fair-esm==2.0.0` in `pip:` section of `rapidock-env.yml` (already implicit).
- If `torch.cuda.amp.autocast` is removed in a future PyTorch, monkeypatch it in `run_rapidock.py` before importing esm:
  ```python
  # Python 3.9 compatible monkeypatch
  import torch
  if not hasattr(torch.cuda.amp, 'autocast'):
      torch.cuda.amp.autocast = torch.amp.autocast
  ```
- Last resort: use `esm` (the newer Evolutionary Scale package) but that requires verifying RAPiDock compatibility — don't do this speculatively.

---

## Standard Stack

### Core (score-env, Python 3.11)

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| `subprocess` | stdlib | `Popen` with real-time streaming | — |
| `threading` | stdlib | Stderr daemon thread | — |
| `biopython` | ≥1.83 | PDB parsing in `pose_io.py` | Already in `score-env.yml` [VERIFIED] |
| `numpy` | ≥1.26 | Cα coordinate arrays | Already in `score-env.yml` [VERIFIED] |
| `hashlib` | stdlib | SHA256 for receptor + peptide | — |
| `json` | stdlib | Metadata write | — |
| `importlib.metadata` | stdlib (3.9+) | Read dist-info for commit SHA | — |

### Core (rapidock-env, Python 3.9)

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| `torch` | 2.7.* | GPU inference | `rapidock-env.yml` [VERIFIED] |
| `torch-geometric` | 2.x | Graph neural network | `pyg::pyg` channel [VERIFIED] |
| `fair-esm` | 2.0.0 | ESM2 sequence embeddings | `requirement.txt` [VERIFIED] |
| `MDAnalysis` | ≥2.7 | PDB I/O inside RAPiDock | `rapidock-env.yml` [VERIFIED] |
| `e3nn` | ≥0.5 | Equivariant neural net layers | `rapidock-env.yml` [VERIFIED] |

---

## Architecture Patterns

### System Architecture Diagram

```
score-env (Python 3.11)
┌────────────────────────────────────────────────────────────┐
│  rapidock_runner.py                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Write metadata skeleton (status="running")        │  │
│  │ 2. Build conda run command with absolute paths       │  │
│  │ 3. Popen(cmd, stdout=PIPE, stderr=PIPE)              │  │
│  │    ├── Main thread: readline() loop on stdout        │  │──► logger DEBUG
│  │    └── Daemon thread: readline() loop on stderr      │  │──► logger DEBUG
│  │ 4. proc.wait() → raise RuntimeError if non-zero     │  │
│  │ 5. Rename rank*.pdb → pose_{i}.pdb                  │  │
│  └──────────────────────────────────────────────────────┘  │
│          │ conda run --no-capture-output -n rapidock-env   │
│          │ python /abs/path/run_rapidock.py [args]         │
│          ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ rapidock-env subprocess (Python 3.9)                 │  │
│  │  run_rapidock.py                                     │  │
│  │  ├── torch.manual_seed(seed) if seed given           │  │
│  │  ├── sys.path.insert(0, rapidock_dir)                │  │
│  │  ├── from utils.inference_parsing import get_parser  │  │
│  │  ├── import inference as rd_inference                │  │
│  │  └── rd_inference.main(namespace)                    │  │
│  │       ├── ESM2 embeddings (GPU)                      │  │
│  │       ├── Diffusion sampling × N passes              │  │
│  │       └── Writes: {output_dir}/poses_raw/rank*.pdb  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  pose_io.py                                                │
│  ├── glob poses/pose_*.pdb (after rename)                  │
│  ├── PDBParser.get_structure() per file                    │
│  ├── SEQRES → sequence (fallback: ATOM residue names)      │
│  ├── Select CA atoms → np.ndarray [n_res, 3]              │
│  └── Returns (list[PoseRecord], list[PoseFailure])         │
│                                                            │
│  output/metadata.py                                        │
│  ├── write_metadata_skeleton(config, output_dir)           │
│  │    └── JSON with status="running", timestamp_start      │
│  └── write_metadata_final(config, results, output_dir)     │
│       └── JSON with all D-16 fields, status="complete"     │
└────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
src/hybridock_pep/
├── sampling/
│   ├── __init__.py          # stub (exists)
│   ├── rapidock_runner.py   # score-env: Popen orchestrator
│   ├── run_rapidock.py      # rapidock-env (Python 3.9): calls inference.main()
│   └── pose_io.py           # score-env: PDB → list[PoseRecord]
└── output/
    ├── __init__.py          # stub (exists)
    └── metadata.py          # score-env: writes run_metadata.json

tests/
├── test_sampling.py         # NEW — covers rapidock_runner, pose_io
└── test_output.py           # NEW — covers metadata.py
```

### Pattern 1: Subprocess Streaming with Daemon Thread

**What:** `Popen` with real-time stdout/stderr capture using `threading.Thread`.
**When to use:** Any subprocess that may emit GPU OOM errors on stderr while producing verbose stdout.

```python
# Source: CONTEXT.md D-01, D-02, D-03 (locked decisions)
import subprocess
import threading
import logging

logger = logging.getLogger(__name__)

def _stream_stderr(stream):
    for line in stream:
        logger.debug("[rapidock stderr] %s", line.rstrip())

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
t = threading.Thread(target=_stream_stderr, args=(proc.stderr,), daemon=True)
t.start()

for line in proc.stdout:
    logger.debug("[rapidock stdout] %s", line.rstrip())

proc.wait()
t.join()

if proc.returncode != 0:
    raise RuntimeError(
        f"RAPiDock subprocess exited with code {proc.returncode}"
    )
```

### Pattern 2: Absolute Path Resolution for conda run

**What:** Every path crossing the `conda run` boundary must be pre-resolved to absolute.
**Why:** conda's subprocess working directory is unpredictable. [CITED: CLAUDE.md §7]

```python
# Source: CONTEXT.md D-07 (locked decision)
from pathlib import Path

run_rapidock_abs = str(Path(__file__).parent / "run_rapidock.py").resolve()  # wrong
run_rapidock_abs = str((Path(__file__).parent / "run_rapidock.py").resolve())  # correct
receptor_abs = str(config.receptor_path.resolve())
output_dir_abs = str(config.output_dir.resolve())

cmd = [
    "conda", "run", "--no-capture-output", "-n", "rapidock-env",
    "python", run_rapidock_abs,
    "--peptide", config.peptide_sequence,
    "--receptor", receptor_abs,
    "--output-dir", output_dir_abs,
    "--n-samples", str(config.n_samples),
    "--rapidock-dir", rapidock_dir_abs,
    "--model-dir", model_dir_abs,
    "--ckpt", ckpt_filename,
    "--scoring-function", "confidence",
]
if config.seed is not None:
    cmd += ["--seed", str(config.seed)]
```

### Pattern 3: rapidock_commit_sha Discovery

**What:** Read the commit SHA of the pip-installed RAPiDock package from `direct_url.json`.

```python
# Source: PEP 610 / Python Packaging User Guide [CITED: packaging.python.org/en/latest/specifications/direct-url]
import importlib.metadata
import json

def get_rapidock_commit_sha() -> str:
    """Return the git commit SHA of the installed RAPiDock package.

    Reads from direct_url.json in the .dist-info directory, which pip
    writes when installing from a git URL (PEP 610).

    Returns:
        Commit SHA string, or "unknown" if not resolvable.
    """
    try:
        dist = importlib.metadata.distribution("rapidock")
        direct_url_path = dist._path / "direct_url.json"  # type: ignore[attr-defined]
        data = json.loads(direct_url_path.read_text())
        return data.get("vcs_info", {}).get("commit_id", "unknown")
    except Exception:
        return "unknown"
```

Note: `dist._path` is a private attribute of `importlib.metadata.PathDistribution`. A more portable alternative is scanning `dist.files` for the `direct_url.json` entry. Either is acceptable — `_path` is simpler and works on CPython 3.9+. [ASSUMED: `_path` attribute name is stable — not guaranteed by public API]

Portable fallback:
```python
for f in (dist.files or []):
    if f.name == "direct_url.json":
        data = json.loads(f.read_text())
        return data.get("vcs_info", {}).get("commit_id", "unknown")
```

### Pattern 4: Biopython PDB Parsing for Cα Extraction

**What:** Use `Bio.PDB.PDBParser` to parse RAPiDock output PDB files.
**Why:** Already in `score-env.yml` (biopython ≥1.83). Handles SEQRES, ATOM, HETATM, altlocs, and multi-model PDB files correctly. Manual parsing would need to replicate all these edge cases.

```python
# Source: Biopython PDB module [VERIFIED: score-env.yml includes biopython>=1.83]
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
import numpy as np

def parse_pose(pdb_path: Path, pose_idx: int) -> "PoseRecord | PoseFailure":
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(f"pose_{pose_idx}", str(pdb_path))
    except Exception as e:
        return PoseFailure(pose_idx=pose_idx, stage="parsing",
                           error_msg=f"PDBParser failed: {e}")

    model = structure[0]  # first MODEL
    ca_list = []
    seq_residues = []

    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" not in residue:
                continue
            ca_list.append(residue["CA"].get_vector().get_array())
            try:
                seq_residues.append(three_to_one(residue.get_resname()))
            except KeyError:
                seq_residues.append("X")

    if not ca_list:
        return PoseFailure(pose_idx=pose_idx, stage="parsing",
                           error_msg="No CA atoms found in PDB")

    sequence = "".join(seq_residues)
    ca_coords = np.array(ca_list, dtype=np.float64)  # shape [n_res, 3]
    return PoseRecord(
        pose_idx=pose_idx,
        pdb_path=pdb_path.resolve(),
        sequence=sequence,
        ca_coords=ca_coords,
    )
```

Note: SEQRES extraction from Biopython is done via `SMCRA` chain header records — but `PDBParser` does not expose SEQRES natively in the Structure object. The D-14 fallback (residue names from ATOM records) is what `three_to_one(residue.get_resname())` provides. Full SEQRES parsing would require reading the raw PDB file for `SEQRES` lines separately. **Recommendation: use residue names from ATOM records as the primary source (not SEQRES) since RAPiDock writes all-atom PDB files via MDAnalysis and the ATOM record sequence is the ground truth.** SEQRES may not always be present in MDAnalysis-written PDB files.

### Pattern 5: Metadata Write Pattern

**What:** Write skeleton at run start, overwrite at completion. Compatible with vina.py's `_append_clipped_pose` (which reads and modifies the JSON file).

```python
# Source: CONTEXT.md D-15, D-16 (locked decisions)
import json, hashlib, os, subprocess, time
from pathlib import Path

def write_metadata_skeleton(config: "DockConfig", output_dir: Path) -> None:
    data = {
        "status": "running",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "poses_requested": config.n_samples,
        "seed": config.seed,
        "cli_args": {k: str(v) for k, v in config.model_dump().items()},
        "receptor_sha256": _sha256(config.receptor_path),
        "peptide_sequence_hash": hashlib.sha256(
            config.peptide_sequence.encode()
        ).hexdigest(),
        "git_sha": _git_sha(),
        "rapidock_commit_sha": get_rapidock_commit_sha(),
        "vina_version": _vina_version(),
        "openmm_version": _openmm_version(),
        "cuda_version": _cuda_version(),
    }
    _write_json(output_dir / "run_metadata.json", data)

def write_metadata_final(
    metadata_path: Path, poses_generated: int, status: str = "complete"
) -> None:
    # Read existing (may have clipped_poses from vina.py _append_clipped_pose)
    data = _read_json(metadata_path)
    data["status"] = status
    data["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["poses_generated"] = poses_generated
    _write_json(metadata_path, data)
```

**CRITICAL:** `write_metadata_final` must READ the existing file first (using the same atomic read-modify-write pattern as `_append_clipped_pose` in `vina.py`) to preserve any `clipped_poses` entries that vina.py wrote during scoring. It must NOT blindly overwrite the file.

### Anti-Patterns to Avoid

- **Using `communicate()` instead of `readline()` loop:** Buffers all output until subprocess exits — GPU OOM on stderr is swallowed until the end. Explicitly rejected in D-02.
- **Using `asyncio` subprocess:** Same buffering problem under load; adds complexity for no benefit. Rejected in D-02.
- **Relative paths across conda boundary:** conda run spawns a new shell in an unpredictable cwd. `Path("output").resolve()` called in the subprocess may resolve to a different directory than the caller's cwd. Always pre-resolve in score-env before building the command. [CITED: CLAUDE.md §7]
- **ThreadPoolExecutor for stderr thread:** The task is a single long-running reader, not a pool of short tasks. `threading.Thread` is more explicit, avoids the overhead of a managed pool, and is trivial to join. Use `threading.Thread(daemon=True)`.
- **Blinding overwriting `run_metadata.json` at completion:** Scoring (Phase 3) appends `clipped_poses` to the same file via `_append_clipped_pose`. The final write in `metadata.py` must read-modify-write.
- **Calling `fastrelax=True` in RAPiDock:** Triggers PyRosetta `relax_score` via multiprocessing.Pool. PyRosetta has a non-redistributable license AND the ref2015 scoring function fails on C-terminal cysteine (LISDAELEAIFEADC). Permanently set `fastrelax=False`. [CITED: CLAUDE.md §2.5, §2.6]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDB file parsing | Custom ATOM line parser | `Bio.PDB.PDBParser` | Handles altlocs, multi-model, insertion codes, malformed records |
| Residue → 1-letter AA | `AA_MAP = {"ALA": "A", ...}` dict | `Bio.PDB.Polypeptide.three_to_one` | Handles all 20 standard AAs + non-standard with fallback |
| Atomic JSON write | `Path.write_text(json.dumps(...))` | `os.replace(tmp, path)` | Same atomic pattern as `_append_clipped_pose` in vina.py — prevents corrupt JSON on crash |
| Git SHA discovery | `subprocess.run(["git", "log"])` | `subprocess.run(["git", "rev-parse", "HEAD"])` — 1 line | Simpler; already in repo context |
| RAPiDock SHA discovery | Parsing pip show output | `importlib.metadata` + `direct_url.json` | PEP 610 canonical method; pip always writes this for git installs |
| Conda env detection | Parse `conda info` JSON | `shutil.which("conda")` + assume env name from config | We already know the env name is `rapidock-env` |

---

## Common Pitfalls

### Pitfall 1: RAPiDock Output Filenames Are Not `pose_{i}.pdb`

**What goes wrong:** `pose_io.py` globs `poses/pose_*.pdb` but finds nothing because RAPiDock wrote `rank1_confidence.pdb`.
**Why it happens:** RAPiDock names files by rank and scoring function, not by index. This is hardcoded in `save_predictions()`.
**How to avoid:** In `rapidock_runner.py`, after `proc.wait()` and exit code check, rename files:
```python
raw_poses_dir = output_dir / "poses_raw" / "poses_raw"
poses_dir = output_dir / "poses"
poses_dir.mkdir(exist_ok=True)
pdb_files = sorted(raw_poses_dir.glob("rank*.pdb"),
                   key=lambda p: int(p.stem.split("rank")[1].split("_")[0]))
for i, src in enumerate(pdb_files):
    dst = poses_dir / f"pose_{i}.pdb"
    src.rename(dst)
```
**Warning signs:** Zero PoseRecord results from `pose_io.py` even though RAPiDock exited 0.

### Pitfall 2: RAPiDock Writes Into a Subdirectory Named After `complex_name`

**What goes wrong:** Expecting files in `{output_dir}/poses_raw/*.pdb` but they're actually in `{output_dir}/poses_raw/poses_raw/*.pdb` (or whatever `--complex-name` was).
**Why it happens:** `process_complex()` writes to `f"{args.output_dir}/{name}"` where `name` is `args.complex_name`. So with `--output-dir /tmp/run/raw --complex-name poses_raw`, files land in `/tmp/run/raw/poses_raw/`.
**How to avoid:** Set `--complex-name` to a known value (e.g., `"raw"`) and construct the glob path accordingly:
```python
# In rapidock_runner.py after subprocess exits:
raw_poses_dir = (output_dir / "poses_raw" / "raw").resolve()
```
Or set `--complex-name raw` and look for `{output_dir}/poses_raw/raw/rank*.pdb`.

### Pitfall 3: `--no-capture-output` Required in `conda run`

**What goes wrong:** RAPiDock's stdout/stderr is captured by conda's own buffering and not passed to the Popen pipes in real time.
**Why it happens:** `conda run` by default captures subprocess output to format it. `--no-capture-output` disables this and passes streams directly.
**How to avoid:** Command must always include `conda run --no-capture-output -n rapidock-env ...`. [CITED: REQUIREMENTS.md SAMP-01]
**Warning signs:** `proc.stdout.readline()` returns empty string immediately even while RAPiDock is still running.

### Pitfall 4: RAPiDock Has No `--seed` Flag — Seeds Must Be Set In-Process

**What goes wrong:** `run_rapidock.py` tries to pass `--seed` to `inference.py`'s argument parser and gets `unrecognized arguments: --seed`.
**Why it happens:** `inference.py`'s argparser (in `utils/inference_parsing.py`) has no `--seed` argument. [VERIFIED: live source]
**How to avoid:** `run_rapidock.py`'s own argparser defines `--seed`. The shim calls `torch.manual_seed(args.seed)` before invoking `rd_inference.main()`. The seed is NOT passed to inference.py's args.
**Warning signs:** `argparse` SystemExit with error about unrecognized arguments.

### Pitfall 5: SEQRES Records May Be Missing in MDAnalysis-Written PDB Files

**What goes wrong:** D-14 says "sequence from SEQRES records first". But MDAnalysis (which RAPiDock uses to write PDB files) does not write SEQRES records by default.
**Why it happens:** `raw_pdb.atoms.write(file)` in `save_predictions()` calls `MDAnalysis.Writer`, which writes ATOM/HETATM but not SEQRES.
**How to avoid:** In `pose_io.py`, use ATOM record residue names as the primary (and probably only) sequence source. Do not fail if SEQRES is absent — that is the expected case for RAPiDock output. The D-14 SEQRES path is a nice-to-have fallback, not the primary path.

### Pitfall 6: `metadata.py` Final Write Must Preserve `clipped_poses` Added by `vina.py`

**What goes wrong:** `write_metadata_final()` overwrites `run_metadata.json` with fresh data, erasing `clipped_poses` entries that `scoring/vina.py`'s `_append_clipped_pose()` wrote during the scoring phase (Phase 3).
**Why it happens:** If `write_metadata_final()` calls `json.dumps(fresh_dict)`, it clobbers everything written by other components.
**How to avoid:** Always read-modify-write:
```python
existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
existing.update(new_fields)
_atomic_write(metadata_path, existing)
```
**Warning signs:** `clipped_poses` key missing from final `run_metadata.json` even when poses were clipped during scoring.

### Pitfall 7: `DockConfig.model_dump()` Contains Non-JSON-Serializable Types

**What goes wrong:** `json.dumps(config.model_dump())` raises `TypeError: Object of type Path is not JSON serializable`.
**Why it happens:** `DockConfig` has `receptor_path: Path`, `output_dir: Path`, `scoring: set[Literal[...]]` — none of which are JSON-native.
**How to avoid:** Serialize to strings explicitly:
```python
cli_args = {k: str(v) for k, v in config.model_dump().items()}
```
Or use `config.model_dump(mode="json")` — Pydantic v2 serializes Path as string automatically in JSON mode.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥7.x (score-env) |
| Config file | `pyproject.toml` (existing) |
| Quick run command | `pytest tests/test_sampling.py tests/test_output.py -x` |
| Full suite command | `pytest --cov=hybridock_pep` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SAMP-01 | `rapidock_runner.py` builds correct `conda run` command | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_command_construction -x` | ❌ Wave 0 |
| SAMP-01 | `rapidock_runner.py` raises `RuntimeError` on non-zero exit | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_nonzero_exit_raises -x` | ❌ Wave 0 |
| SAMP-01 | `rapidock_runner.py` warns (not raises) on shortfall | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_shortfall_warns -x` | ❌ Wave 0 |
| SAMP-01 | `rapidock_runner.py` raises `RuntimeError` on zero poses | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_zero_poses_raises -x` | ❌ Wave 0 |
| SAMP-01 | `pose_io.py` parses valid PDB into `PoseRecord` with correct `ca_coords` | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_valid_pdb -x` | ❌ Wave 0 |
| SAMP-01 | `pose_io.py` returns `PoseFailure` on malformed PDB | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_malformed_pdb -x` | ❌ Wave 0 |
| SAMP-01 | `pose_io.py` batch returns `(results, failures)` invariant | unit | `pytest tests/test_sampling.py::TestPoseIO::test_batch_invariant -x` | ❌ Wave 0 |
| SAMP-02 | `metadata.py` skeleton write includes `status="running"` | unit | `pytest tests/test_output.py::TestMetadata::test_skeleton_status -x` | ❌ Wave 0 |
| SAMP-02 | `metadata.py` final write includes all 14 required fields | unit | `pytest tests/test_output.py::TestMetadata::test_final_fields -x` | ❌ Wave 0 |
| SAMP-02 | `metadata.py` final write preserves `clipped_poses` from prior scoring | unit | `pytest tests/test_output.py::TestMetadata::test_preserves_clipped_poses -x` | ❌ Wave 0 |
| SAMP-02 | `get_rapidock_commit_sha()` returns string from `direct_url.json` | unit | `pytest tests/test_output.py::TestMetadata::test_commit_sha_from_direct_url -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_sampling.py tests/test_output.py -x`
- **Per wave merge:** `pytest --cov=hybridock_pep`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_sampling.py` — covers `rapidock_runner.py` and `pose_io.py`
- [ ] `tests/test_output.py` — covers `metadata.py`
- [ ] `tests/fixtures/pose_tiny.pdb` — EXISTS (can reuse for parse tests)

*(No new framework install needed — pytest already configured)*

---

## Security Domain

`security_enforcement` not explicitly set to false — section included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `DockConfig` validators (already built, Phase 1) |
| V6 Cryptography | no | SHA256 used for provenance only, not security |

### Known Threat Patterns for Subprocess Orchestration

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Command injection via peptide sequence / path args | Tampering | Pass as list (not shell=True); `DockConfig` validates peptide to `[A-Z]{1,}` before this phase |
| Path traversal via `--output-dir` | Tampering | DockConfig `output_dir: Path` validated at construction; absolute resolution before use |
| Subprocess hanging indefinitely | Denial of Service | Add `timeout=` to `proc.wait()` or use a watchdog thread — [ASSUMED] Phase 4 v1 does not implement this; flag as v2 item |
| RAPiDock writes outside output_dir | Tampering | RAPiDock creates files only under `args.output_dir` (hardcoded in `save_predictions()`); verified from source |

**Note on `shell=False`:** `subprocess.Popen(cmd_list)` is used throughout (list form, not string). This is the correct pattern — no shell injection possible.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| RAPiDock pins PyTorch 1.11 / CUDA 11.5 | HybriDock-Pep uses PyTorch 2.7 / CUDA 12.8 | 2026-04-21 (STATE.md decision) | Blackwell (CC 12.0) support; requires PyG cu128 wheels |
| `communicate()` for subprocess capture | `readline()` loop + daemon thread for stderr | CONTEXT.md D-01 | Real-time OOM surfacing |
| `torch.cuda.amp.autocast` (old API) | `torch.amp.autocast("cuda", ...)` (PyTorch 2.4+) | PyTorch 2.4 | FutureWarning from fair-esm; no functional break |

**Deprecated/outdated:**
- `rapidock_env.yaml` in RAPiDock repo (pins PyTorch 1.11 / CUDA 11.5): DO NOT USE — incompatible with RTX 5070 (Blackwell). Use `envs/rapidock-env.yml` from HybriDock-Pep instead.
- `--fastrelax` / `--scoring-function ref2015`: triggers PyRosetta relax; fails on C-terminal cysteine. Always set `fastrelax=False` and `scoring_function="confidence"` (or no confidence model — poses are written as `rank{N}.pdb` with no suffix).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | fair-esm 2.0.0 imports cleanly under PyTorch 2.7 (no hard AttributeError, at most FutureWarnings) | Environment Blockers | If wrong: `run_rapidock.py` crashes on import; must monkeypatch `torch.cuda.amp.autocast` or pin an older PyTorch |
| A2 | RAPiDock has no `setup.py`/`pyproject.toml` — pip installs it with the repo root as the module root (so `import inference` works after `sys.path.insert(0, rapidock_dir)`) | RAPiDock API | If wrong: imports fail inside `run_rapidock.py`; need to adjust `sys.path` strategy |
| A3 | MDAnalysis-written PDB files from RAPiDock do NOT include SEQRES records | Common Pitfalls §5 | If wrong: SEQRES-first parsing in D-14 would work as-is (no harm, just unexpected success) |
| A4 | `importlib.metadata.PathDistribution._path` attribute is stable on CPython 3.11 for reading `direct_url.json` | Pattern 3 | If wrong: SHA discovery returns "unknown"; use the `dist.files` iteration fallback instead |
| A5 | CUDA nondeterminism persists even with `torch.manual_seed()` on Blackwell unless `torch.use_deterministic_algorithms(True)` is set | Seed Propagation | If wrong: runs are fully deterministic (better than expected); no harm |
| A6 | RAPiDock's `--scoring-function confidence` mode writes `rank{N}_confidence.pdb`; with no confidence model, it writes `rank{N}.pdb` | RAPiDock API | If wrong: file glob pattern in renaming logic needs adjustment |

---

## Open Questions (RESOLVED)

1. **Does `pip install git+https://github.com/huifengzhao/RAPiDock.git@SHA` register the package as "rapidock" in `importlib.metadata`?**
   - What we know: The repo has no `setup.py` or `pyproject.toml` visible in the root. pip may use a default package name derived from the directory, or may fail to install at all.
   - What's unclear: The exact dist-info package name that `importlib.metadata.distribution("rapidock")` would look up.
   - Recommendation: On day-1 of Phase 4, run `conda run -n rapidock-env pip show rapidock` after installing. If the package name differs, update `get_rapidock_commit_sha()`.
   - **RESOLVED:** Use `importlib.metadata.version("rapidock")` or fall back to parsing `conda run -n rapidock-env pip show rapidock` output. The pip show approach is authoritative and works regardless of whether a `pyproject.toml` was present. If `distribution("rapidock")` raises `PackageNotFoundError`, iterate over all distributions and match by directory name, or use `pip show` subprocess as the fallback.

2. **What `--scoring-function` should `run_rapidock.py` use?**
   - What we know: `"ref2015"` triggers PyRosetta FastRelax (CLAUDE.md §2.5 — must NOT be used). `"confidence"` requires a confidence model checkpoint. With `confidence_model=None`, no re-ranking by confidence occurs.
   - What's unclear: Whether the HybriDock-Pep use case has a confidence model checkpoint available, or if we should run without a scoring function (just `rank{N}.pdb`).
   - Recommendation: Default to no confidence model (`--scoring-function confidence` but with `confidence_model_dir=None`). RAPiDock will write `rank{N}.pdb` files when `confidence is None` (verified from `save_predictions()` code). Document this in the plan.
   - **RESOLVED:** No `--scoring-function` flag is needed in `run_rapidock.py`. RAPiDock uses its internal confidence model; do not pass a scoring function flag. When `confidence_model=None` (default), `save_predictions()` writes `rank{N}.pdb` with no suffix. This is the correct output format for HybriDock-Pep (renamed to `pose_{i}.pdb` by `rapidock_runner.py`).

3. **Does RAPiDock's `InferenceDataset` require the receptor to be a PDB file or can it accept a PDBQT?**
   - What we know: `--protein_description` accepts a PDB path or sequence. RAPiDock calls `get_protein_feature_mda()` which uses MDAnalysis — MDAnalysis can read PDBQT but may behave differently.
   - What's unclear: Whether to pass the original receptor PDB or the prepared receptor PDBQT to RAPiDock.
   - Recommendation: Pass the original `config.receptor_path` (the raw PDB) to RAPiDock, not the PDBQT. RAPiDock generates its own atom features; the PDBQT is for Vina scoring only.
   - **RESOLVED:** Raw PDB is passed directly to RAPiDock per the research body above. No PDBQT conversion is needed for RAPiDock (PDBQT is only for Vina `--score_only` and `--scoring ad4` in Phase 3 scoring). Always pass `config.receptor_path` (the original PDB) as `--protein_description`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| conda | subprocess orchestration | [ASSUMED] ✓ | ≥23.x | None — blocking |
| rapidock-env | Stage 1 inference | Must verify at Phase 4 start | — | None — blocking |
| PyG cu128 wheels | rapidock-env | ✓ confirmed | 2.7.0+cu128 | Source build (not needed) |
| fair-esm 2.0.0 + PyTorch 2.7 | rapidock-env | [ASSUMED] ✓ | 2.0.0 | Monkeypatch torch.cuda.amp |

---

## Sources

### Primary (HIGH confidence)
- `github.com/huifengzhao/RAPiDock` — `inference.py`, `utils/inference_parsing.py`, `utils/sampling.py`, `utils/inference_utils.py`, `requirement.txt`, `default_inference_args.yaml` — all fetched live 2026-04-21
- `data.pyg.org/whl/torch-2.7.0+cu128.html` — PyG wheel availability verified live 2026-04-21
- `src/hybridock_pep/scoring/vina.py` — `_append_clipped_pose` pattern (local codebase)
- `src/hybridock_pep/models.py` — `PoseRecord`, `PoseFailure`, `DockConfig` field definitions (local codebase)
- `envs/score-env.yml`, `envs/rapidock-env.yml` — dependency verification (local codebase)
- `packaging.python.org/en/latest/specifications/direct-url` — PEP 610 `direct_url.json` format

### Secondary (MEDIUM confidence)
- PyTorch 2.4 changelog — `torch.cuda.amp.autocast` deprecation (FutureWarning, not removal)
- facebookresearch/esm README — ESMFold "python <= 3.9" constraint (ESM2 embedding has no such restriction)

### Tertiary (LOW confidence)
- fair-esm 2.0.0 + PyTorch 2.7 import compatibility — no runtime verification; inferred from archived status and backward-compat claim of PyTorch 2.x

---

## Metadata

**Confidence breakdown:**
- RAPiDock API (entry point, args, output naming): HIGH — verified from live GitHub source
- PyG cu128 wheels for PyTorch 2.7.0+cp39: HIGH — verified from data.pyg.org
- fair-esm + PyTorch 2.7 compatibility: LOW — unverified at runtime; must validate day-1
- PDB parsing strategy (Biopython): HIGH — library confirmed in score-env.yml
- Subprocess streaming pattern: HIGH — direct from locked decisions + Python stdlib docs
- Metadata write pattern: HIGH — derives from existing `_append_clipped_pose` in codebase

**Research date:** 2026-04-21
**Valid until:** 2026-05-21 (RAPiDock source is stable; PyG wheels indexed; fair-esm verdict needed before coding)

---

## RESEARCH COMPLETE
