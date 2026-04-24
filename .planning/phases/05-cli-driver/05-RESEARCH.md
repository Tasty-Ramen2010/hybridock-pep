# Phase 5: CLI & Driver - Research

**Researched:** 2026-04-23
**Domain:** Python CLI (argparse), pipeline orchestration (driver.py), module wiring
**Confidence:** HIGH — all findings derived from reading existing source directly

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `--input-poses <dir>` fully wired in Phase 5. When present, driver.py skips `run_sampling()`, calls `parse_poses(<dir>)` directly, and proceeds to Stage 2 (prep + scoring). Mutually exclusive with Stage 1 — driver raises argparse error if ambiguous.
- **D-02:** driver.py wires what exists and stubs what doesn't. After scoring, logs `"Clustering and output: Phase 6/7 not yet implemented"` at INFO level and returns `list[ScoredPose]`. No exception raised — clean exit.
- **D-03:** All four subcommands get real arg definitions. `calibrate` dispatches to `scripts/calibrate_alpha.py` logic. `prep` dispatches to `prep/receptor.py:prepare_receptor()`. `benchmark` raises `NotImplementedError("benchmark: Phase 8 scope")`. Every flag has a help string with units.
- **D-04:** DockConfig Pydantic validators are the primary validation gate. CLI builds a `DockConfig` from parsed args as the first thing in `_run_dock()`. Invalid input raises `ValidationError`, caught and formatted as `parser.error(str(e))` → exit code 2.
- **D-05:** `logging` module throughout. `main()` configures `basicConfig` based on `-v` count (already in stub). Every subprocess call logs the full command at DEBUG level. Driver logs stage transitions at INFO.

### Claude's Discretion

- How `calibrate_alpha.py` is invoked from the `calibrate` subcommand (import as module vs. subprocess) — choose based on what's cleanest given the existing script structure.
- Exact arg name for the grid box center — use `--site` per CLAUDE.md §5.
- Exit code conventions beyond code 2 (argparse default).

### Deferred Ideas (OUT OF SCOPE)

- `--skip-sampling` flag (v2 scope, OPT-02). `--input-poses` covers the Mac use case for Phase 5.
- MM-GBSA `--refine-topk` execution: flag is defined on `dock` for Phase 5, but actual MM-GBSA dispatch is v2 scope (OPT-01). Phase 5 validates the flag and stores in DockConfig; driver.py notes it's unimplemented.
- Parallel scoring across poses: v1 uses sequential scoring; parallelism is a v2 optimization.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLI-01 | User can run `dock`, `calibrate`, `benchmark`, `prep` subcommands from a single `hybridock-pep` entry point | cli.py stub exists with all 4 subcommand names; Phase 5 fills in arg definitions and dispatch functions |
| CLI-02 | All inputs validated and errors reported before any subprocess is spawned | DockConfig with Pydantic validators already handles peptide, receptor_path, box_size, n_samples; ValidationError caught and formatted as parser.error() → exit code 2 |
| CLI-03 | User can pass `--seed N` to make a run deterministic (modulo CUDA nondeterminism, flagged in metadata) | DockConfig.seed: int | None already implemented; run_sampling() propagates seed to --seed arg; metadata.py writes seed field in skeleton |
</phase_requirements>

---

## Summary

Phase 5 is a pure wiring phase — every module it needs to call is already implemented and tested. The task is: (1) expand the cli.py stub into real subcommand arg definitions, (2) write driver.py as the orchestrator for the dock subcommand, and (3) wire calibrate/prep/benchmark dispatch.

All four subcommand bodies are stubs that raise `parser.error(...)` today. The parser structure (4 subcommands, `-v` verbosity, entry point in pyproject.toml) is already in place and must not be rebuilt. The planner should treat cli.py as an expand-in-place task, not a rewrite.

The single largest integration decision is how `calibrate_alpha.py` is invoked. The script's `main()` accepts an `argparse.Namespace` and is designed to be called as a function (it has `if args is None: args = parse_args()` at the top of `main()`). The cleanest path is to import `calibrate_alpha.main` and pass a hand-constructed `Namespace` from the CLI args — this avoids a subprocess, keeps logging unified, and lets pytest mock it.

**Primary recommendation:** Import calibrate_alpha as a module. Build driver.py as a single `run_dock(config, input_poses_dir)` function that calls the existing modules in sequence and returns `list[ScoredPose]`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Arg parsing, validation | CLI (cli.py) | — | DockConfig is the single validation gate; no validation logic in driver.py |
| Stage 1 orchestration (RAPiDock) | Driver (driver.py) | rapidock_runner.py | driver.py calls run_sampling(); runner owns subprocess mechanics |
| Stage 2 orchestration (prep + scoring) | Driver (driver.py) | prep/, scoring/ | driver.py calls prepare_receptor, prepare_ligand_batch, score_vina_batch, score_ad4_batch, apply_hybrid_score in sequence |
| `--input-poses` bypass | Driver (driver.py) | cli.py flag | cli.py defines the flag; driver.py checks it and branches before run_sampling() |
| Metadata provenance | Driver (driver.py) | output/metadata.py | driver.py calls write_metadata_skeleton() before Stage 1 and finalize_metadata() after parsing |
| calibrate dispatch | CLI (_run_calibrate()) | calibrate_alpha.main() | CLI imports and calls; no subprocess |
| prep dispatch | CLI (_run_prep()) | prep/receptor.py | CLI builds a minimal DockConfig and calls prepare_receptor() |
| benchmark stub | CLI (_run_benchmark()) | — | Raises NotImplementedError; no driver involvement |

---

## Standard Stack

### Core (all already in pyproject.toml dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| argparse | stdlib | Subcommand CLI parsing | CLAUDE.md §4 mandate; already used in stub and calibrate_alpha.py |
| pydantic ≥ 2.0 | ≥2.0 | DockConfig validation | Already the project standard; frozen model; field_validators already implemented |
| logging | stdlib | All output | CLAUDE.md §4 mandate; no print() anywhere |
| pathlib.Path | stdlib | All file paths | Project standard; DockConfig.receptor_path, output_dir are Path objects |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sys | stdlib | sys.argv, sys.exit | CLI entry point cleanup |
| dataclasses | stdlib | ScoredPose, PoseRecord, PoseFailure | Already the model layer |

### No new dependencies
Phase 5 adds zero new packages. Everything it needs is already imported and tested in Phases 2–4.

---

## Architecture Patterns

### System Architecture Diagram

```
CLI args
    │
    ▼
cli.py:main()
    │
    ├─ _build_parser()  ← expand existing stub with real arg defs
    │
    ├─ args.command == "dock"
    │       │
    │       ▼
    │   DockConfig(**vars(args))  ← ValidationError → parser.error() → exit 2
    │       │
    │       ▼
    │   driver.run_dock(config, input_poses_dir=args.input_poses)
    │       │
    │       ├─ [Stage 0] write_metadata_skeleton(config, metadata_path)
    │       │
    │       ├─ [Stage 1] if input_poses_dir:
    │       │       parse_poses(input_poses_dir)
    │       │   else:
    │       │       run_sampling(config) → poses/
    │       │       parse_poses(config.output_dir / "poses")
    │       │
    │       ├─ [Stage 2a] prepare_receptor(config) → receptor.pdbqt
    │       │             generate_ad4_maps(config, receptor_pdbqt) → maps/
    │       │
    │       ├─ [Stage 2b] prepare_ligand_batch(pdb_paths, pdbqt_dir)
    │       │             → (pdbqt_paths, prep_failures)
    │       │
    │       ├─ [Stage 2c] build ScoredPose list from PoseRecord + pdbqt_path
    │       │
    │       ├─ [Stage 2d] score_vina_batch(poses, config, receptor_pdbqt)
    │       │             score_ad4_batch(poses, maps_dir)
    │       │             apply_hybrid_score(pose, alpha, beta, n_residues) per pose
    │       │
    │       ├─ [Stage 3 stub] logger.info("Clustering and output: Phase 6/7 not yet implemented")
    │       │
    │       ├─ finalize_metadata(metadata_path, poses_generated=len(records))
    │       │
    │       └─ return list[ScoredPose]
    │
    ├─ args.command == "calibrate"
    │       │
    │       ▼
    │   _run_calibrate(args)  → import calibrate_alpha; calibrate_alpha.main(namespace)
    │
    ├─ args.command == "prep"
    │       │
    │       ▼
    │   _run_prep(args)  → prepare_receptor(config)
    │
    └─ args.command == "benchmark"
            │
            ▼
        raise NotImplementedError("benchmark: Phase 8 scope")
```

### Recommended Project Structure (changes from Phase 5)
```
src/hybridock_pep/
├── cli.py          ← EXPAND in place (do not rebuild parser from scratch)
└── driver.py       ← NEW file; imported by cli.py for dock subcommand
```

### Pattern 1: DockConfig as validation gate (D-04)
**What:** Build DockConfig immediately from parsed args; catch ValidationError and re-raise as parser.error().
**When to use:** Always first in `_run_dock()`; also in `_run_prep()` for receptor path validation.

```python
# Source: models.py field_validators + CONTEXT.md D-04
def _run_dock(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from pydantic import ValidationError
    try:
        config = DockConfig(
            peptide_sequence=args.peptide,
            receptor_path=Path(args.receptor),
            site_coords=tuple(args.site),
            box_size=args.box,
            n_samples=args.n_samples,
            seed=args.seed,
            scoring=set(args.scoring.split(",")),
            output_dir=Path(args.output_dir),
        )
    except ValidationError as exc:
        parser.error(str(exc))
```

### Pattern 2: Collect-all-failures with failure threshold check (D-02)
**What:** After each batch operation, log failures at WARNING, continue if enough poses survive.
**When to use:** After prepare_ligand_batch and after scoring.

```python
# Source: ligand.py, vina.py — established collect-all-failures pattern
pdbqt_paths, prep_failures = prepare_ligand_batch(pdb_paths, pdbqt_dir)
if prep_failures:
    logger.warning("%d poses failed ligand prep", len(prep_failures))
if not pdbqt_paths:
    raise RuntimeError("All poses failed ligand prep — cannot continue.")
```

### Pattern 3: ScoredPose construction from PoseRecord + pdbqt_path
**What:** ScoredPose is a dataclass that extends PoseRecord. To construct it, pass all PoseRecord fields plus the optional ScoredPose fields.
**Critical detail:** ScoredPose is a `@dataclass` inheriting from PoseRecord, also a `@dataclass`. Python dataclass inheritance requires the child to list all parent fields that have no defaults before child fields with defaults. The existing definition in models.py has all PoseRecord fields required (no defaults) and all ScoredPose fields defaulted — so construction is:

```python
# Source: models.py ScoredPose definition
from hybridock_pep.models import ScoredPose

scored_pose = ScoredPose(
    pose_idx=record.pose_idx,
    pdb_path=record.pdb_path,
    sequence=record.sequence,
    ca_coords=record.ca_coords,
    pdbqt_path=pdbqt_path,   # set immediately after prep
    # vina_score, ad4_score, etc. default to None
)
```

### Pattern 4: apply_hybrid_score requires both scores pre-set
**What:** `apply_hybrid_score()` asserts `vina_score is not None` and `ad4_score is not None`. Driver must call vina scoring, then ad4 scoring, then entropy — in that order.

```python
# Source: scoring/entropy.py apply_hybrid_score()
from hybridock_pep.scoring.entropy import apply_hybrid_score, load_calibration

cal = load_calibration(Path(args.calibration))  # validated on load
alpha, beta = cal["alpha"], cal["beta"]
n_residues = len(config.peptide_sequence)

for pose in scored_poses:
    apply_hybrid_score(pose, alpha=alpha, beta=beta, n_residues=n_residues)
```

### Pattern 5: calibrate subcommand dispatch — import as module, not subprocess
**What:** `calibrate_alpha.main()` accepts an optional `argparse.Namespace`. Pass a hand-constructed Namespace from the CLI args.
**Why not subprocess:** Script is in `scripts/`, not on PATH by default. Subprocess would require knowing the absolute path. Direct import is cleaner, keeps logging unified, allows pytest mocking.

```python
# Source: scripts/calibrate_alpha.py main() signature
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import calibrate_alpha

def _run_calibrate(args: argparse.Namespace) -> None:
    ns = argparse.Namespace(
        training_csv=args.training_csv,
        scores_json=args.scores_json,
        output=args.output,
        verbose=args.verbose > 0,
    )
    calibrate_alpha.main(ns)
```

**Alternative:** Add `scripts/` to the package (or import via `importlib`). Either works; direct sys.path insert is simplest given the existing script structure.

### Pattern 6: --input-poses mutual exclusion
**What:** `--input-poses` and `--n-samples > 0` together is ambiguous. Detect and error before DockConfig construction.
**When to use:** At the top of `_run_dock()`, before building DockConfig.

```python
# Source: CONTEXT.md D-01
if args.input_poses and args.n_samples > 0:
    parser.error(
        "--input-poses and --n-samples are mutually exclusive. "
        "Use --input-poses to skip Stage 1 (macOS), or --n-samples to run RAPiDock."
    )
```

### Pattern 7: prep subcommand — receptor-only DockConfig
**What:** `prep` only needs receptor_path and output_dir to call prepare_receptor(). Build a minimal DockConfig with dummy values for required fields that aren't used.
**Issue:** DockConfig requires site_coords, box_size, peptide_sequence, and output_dir. For `prep`, these aren't meaningful. Options:
1. Accept them as optional args to `prep` subcommand with dummy defaults.
2. Relax DockConfig for prep (contradicts D-04 — DockConfig is the single gate).
3. Call prepare_receptor() directly without DockConfig (bypasses the pattern).

**Recommendation:** Option 3 — `_run_prep()` calls `prepare_receptor(config)` where config is a minimal DockConfig with dummy coordinates. This avoids introducing dummy defaults that are confusing to users. The `prep` subcommand only needs `--receptor` and `--output-dir`; other DockConfig fields can use safe defaults (sequence "A", site_coords (0,0,0), box_size 20).

### Anti-Patterns to Avoid

- **Rebuilding the parser from scratch:** cli.py stub has the parser with 4 subcommands. Expand `_build_parser()` in place; do not replace it.
- **Validation logic in driver.py:** All validation is in DockConfig. driver.py never validates inputs — it trusts the config it receives.
- **print() anywhere:** All output is via `logging`. Even the final summary of scored poses should be logger.info().
- **Relative paths to subprocess:** All paths passed across conda boundary must be `.resolve()`d. Already established in Phase 4 (run_sampling enforces this).
- **Calling score_ad4_batch before score_vina_batch:** apply_hybrid_score asserts both scores are set. Order matters: Vina → AD4 → entropy.
- **Raising on partial prep/scoring failures:** Collect-all-failures pattern. Log failures at WARNING; raise only if zero poses survive.
- **Python 3.10+ syntax in run_rapidock.py:** That file runs in rapidock-env (Python 3.9). Phase 5 does not touch run_rapidock.py, but any shim added near it must stay 3.9-compatible.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Input validation | Custom validation functions in CLI | `DockConfig(...)` then catch `ValidationError` | Validators already implemented and tested in test_models.py |
| Grid maps generation | Inline autogrid4 call | `prep.grids.generate_ad4_maps(config, receptor_pdbqt)` | Already implemented with PrepError diagnostics |
| Receptor preparation | Inline pdbfixer + prepare_receptor4 | `prep.receptor.prepare_receptor(config)` | Already implemented with filter + fixer + hard abort |
| Calibration fitting | New scipy optimization | `scoring.entropy.fit_calibration()` + `calibrate_alpha.main()` | Already implemented with L-BFGS-B bounds |
| Metadata writing | Manual JSON dict construction | `output.metadata.write_metadata_skeleton()` + `finalize_metadata()` | Already implemented with atomic write and read-modify-write |

---

## Verified Interface Signatures

All signatures verified by reading source directly. [VERIFIED: source read]

### run_sampling
```python
# src/hybridock_pep/sampling/rapidock_runner.py
def run_sampling(config: DockConfig) -> list[Path]:
    # Returns: list of absolute Paths to pose_*.pdb files under config.output_dir/poses/
    # Raises: RuntimeError if exit != 0 or zero poses produced
```

### parse_poses
```python
# src/hybridock_pep/sampling/pose_io.py
def parse_poses(poses_dir: Path) -> tuple[list[PoseRecord], list[PoseFailure]]:
    # Scans for pose_*.pdb in poses_dir
    # Returns: (records, failures) — never raises
```

### prepare_receptor
```python
# src/hybridock_pep/prep/receptor.py
def prepare_receptor(config: DockConfig) -> Path:
    # Returns: Path to output_dir/receptor.pdbqt
    # Raises: PrepError if prepare_receptor4.py exits non-zero
    # Raises: FileNotFoundError if prepare_receptor4.py not on PATH
```

### generate_ad4_maps
```python
# src/hybridock_pep/prep/grids.py
def generate_ad4_maps(config: DockConfig, receptor_pdbqt: Path) -> Path:
    # Returns: Path to output_dir/maps/ directory
    # Raises: PrepError if autogrid4 exits non-zero or receptor.HD.map missing
```

### prepare_ligand_batch
```python
# src/hybridock_pep/prep/ligand.py
def prepare_ligand_batch(
    pdb_paths: list[Path],      # pose PDB files (not PoseRecord objects — raw Paths)
    output_dir: Path,
    *,
    max_workers: int | None = None,
) -> tuple[list[Path], list[PoseFailure]]:
    # Returns: (pdbqt_paths, failures) — never raises on per-pose errors
```

**Critical:** `prepare_ligand_batch` takes `list[Path]` (PDB paths), NOT `list[PoseRecord]`. Driver must extract `record.pdb_path` from each PoseRecord to build the input list.

### score_vina_batch
```python
# src/hybridock_pep/scoring/vina.py
def score_vina_batch(
    poses: list[ScoredPose],
    config: DockConfig,
    receptor_pdbqt: Path,
    *,
    verbosity: int = 0,
    metadata_path: Path | None = None,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    # Returns: (scored, failures) — per-pose exceptions caught
    # Raises: propagates Vina instance creation / receptor loading errors
```

**Note:** Takes `list[ScoredPose]` not `list[PoseRecord]`. Driver must construct ScoredPose objects (Pattern 3) before calling this.

### score_ad4_batch
```python
# src/hybridock_pep/scoring/ad4.py
def score_ad4_batch(
    poses: list[ScoredPose],
    maps_dir: Path,
    *,
    verbosity: int = 0,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    # Returns: (scored, failures) — per-pose exceptions caught
    # Raises: FileNotFoundError if receptor.HD.map missing (check before loop)
    # Raises: propagates Vina(sf_name='ad4') / load_maps() errors
```

**Note:** `maps_dir` is the directory (e.g., `config.output_dir / "maps"`). The map_prefix used internally is `str(maps_dir / "receptor")`.

### apply_hybrid_score
```python
# src/hybridock_pep/scoring/entropy.py
def apply_hybrid_score(
    pose: ScoredPose,
    *,
    alpha: float,
    beta: float,
    n_residues: int,
) -> None:
    # Mutates pose in place: sets entropy_correction and hybrid_score
    # Asserts: pose.vina_score is not None AND pose.ad4_score is not None
```

### load_calibration
```python
# src/hybridock_pep/scoring/entropy.py
def load_calibration(path: Path) -> dict:
    # Returns: dict with keys 'alpha', 'beta', plus calibration metadata
    # Raises: FileNotFoundError if path does not exist
    # Raises: ValueError if alpha not in [0.2, 1.2] or beta not in [0.0, 0.5]
```

### write_metadata_skeleton / finalize_metadata
```python
# src/hybridock_pep/output/metadata.py
def write_metadata_skeleton(config: DockConfig, metadata_path: Path) -> None:
    # Writes status="running" skeleton before Stage 1

def finalize_metadata(
    metadata_path: Path,
    poses_generated: int,
    cuda_version: str | None = None,
    status: str = "complete",
) -> None:
    # Read-modify-write; preserves clipped_poses entries
```

---

## dock Subcommand Flag Reference

Exact flag names from CLAUDE.md §5 and CONTEXT.md. [VERIFIED: CLAUDE.md §5 read]

| Flag | Type | Required | Default | Help |
|------|------|----------|---------|------|
| `--peptide` | str | yes | — | Peptide amino acid sequence (single-letter codes) |
| `--receptor` | Path | yes | — | Path to receptor PDB file |
| `--site` | 3 floats | yes | — | Grid box center coordinates in Angstroms (x y z) |
| `--box` | float | yes | — | Grid box edge length in Angstroms |
| `--n-samples` | int | no | 100 | Number of RAPiDock inference passes |
| `--scoring` | str | no | "vina,ad4" | Comma-separated scoring backends (vina, ad4) |
| `--refine-topk` | int | no | None | Top-K poses for MM-GBSA refinement (v2; validated but not dispatched) |
| `--output-dir` | Path | yes | — | Directory for run outputs |
| `--seed` | int | no | None | Random seed for deterministic sampling |
| `--input-poses` | Path | no | None | Directory of pre-generated pose PDBs (skips Stage 1; required on macOS) |
| `--calibration` | Path | no | data/calibration.json | Path to calibration.json for entropy correction |

**argparse `--site` nargs:** Use `nargs=3, type=float, metavar=("X", "Y", "Z")`. Driver converts `args.site` to `tuple(args.site)` for DockConfig.

---

## calibrate Subcommand Flag Reference

From `scripts/calibrate_alpha.py:parse_args()` — flags must match so the Namespace can be constructed directly. [VERIFIED: calibrate_alpha.py read]

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--training-csv` | Path | data/training_complexes.csv | Training CSV with pdb_id, peptide_sequence, experimental_pkd |
| `--scores-json` | Path | None | JSON mapping pdb_id → {vina_score, ad4_score} (required) |
| `--output` | Path | data/calibration.json | Output calibration.json path |

---

## prep Subcommand Flag Reference

| Flag | Type | Required | Help |
|------|------|----------|------|
| `--receptor` | Path | yes | Receptor PDB to prepare |
| `--output-dir` | Path | yes | Directory to write receptor.pdbqt |

---

## benchmark Subcommand Flag Reference

From CONTEXT.md D-03 — real args defined but NotImplementedError raised immediately.

| Flag | Type | Required | Help |
|------|------|----------|------|
| `--test-csv` | Path | yes | Test complexes CSV (pdb_id, peptide_sequence, experimental_pkd) |
| `--baselines` | str | no | Comma-separated baseline scorers (e.g., vina,adcp,rapidock) |
| `--report` | Path | no | Path to write benchmark report (Markdown) |

---

## Common Pitfalls

### Pitfall 1: ScoredPose construction — dataclass inheritance field order
**What goes wrong:** Calling `ScoredPose(pose_idx=..., pdb_path=..., sequence=..., ca_coords=...)` passes all required fields but misses `pdbqt_path`. The field defaults to `None` which is valid, but vina.py will then hit `if pose.pdbqt_path is None: raise ValueError(...)` during scoring.
**Why it happens:** ScoredPose inherits from PoseRecord. pdbqt_path is an optional child field but must be set after prep.
**How to avoid:** Immediately after `prepare_ligand_batch`, build a dict mapping `pdb_path → pdbqt_path` and construct ScoredPose with `pdbqt_path=pdbqt_path` set. Only build ScoredPose objects for poses that succeeded prep.
**Warning signs:** `ValueError: Pose N has pdbqt_path=None` in scoring logs.

### Pitfall 2: score_ad4_batch before generate_ad4_maps
**What goes wrong:** `score_ad4_batch` raises `FileNotFoundError("AD4 HD map not found")` if maps haven't been generated. Driver must call `generate_ad4_maps` before either scoring batch.
**Why it happens:** AD4 scoring requires pre-computed autogrid4 maps; there is no lazy generation.
**How to avoid:** Stage 2a (generate_ad4_maps) must complete before Stage 2d (score_ad4_batch).

### Pitfall 3: apply_hybrid_score assertion failure — wrong call order
**What goes wrong:** `AssertionError: vina_score must be set before apply_hybrid_score` if driver calls entropy before vina.
**Why it happens:** `apply_hybrid_score` asserts both scores are not None. If `"ad4"` not in config.scoring, ad4_score stays None and assertion fires.
**How to avoid:** Only call `apply_hybrid_score` when both scoring backends have run. If scoring is vina-only, skip entropy correction or handle ad4_score=None explicitly. For v1, CLAUDE.md §5 shows `--scoring vina,ad4` as the default; both always run.

### Pitfall 4: calibrate_alpha.main() reconfigures basicConfig
**What goes wrong:** `calibrate_alpha.main()` calls `logging.basicConfig(...)` internally. If the CLI has already called `logging.basicConfig()` in `main()`, the second call is a no-op (Python logging design). This means calibrate's log format may differ or be suppressed.
**Why it happens:** calibrate_alpha.py was designed as a standalone script that configures its own logging.
**How to avoid:** In `_run_calibrate()`, pass a hand-constructed `Namespace` that includes `verbose` derived from the CLI's verbosity count. The second `basicConfig()` call will be silent (no-op) which is acceptable — the root logger is already configured.

### Pitfall 5: --input-poses with run_sampling return value
**What goes wrong:** `run_sampling()` returns `list[Path]` (the renamed pose files). When `--input-poses` is used, `run_sampling()` is not called — driver must instead call `parse_poses(input_poses_dir)` directly, skipping the list[Path] step.
**Why it happens:** `parse_poses()` takes a directory, not a list of paths. The driver path for `--input-poses` is: `input_poses_dir → parse_poses(input_poses_dir)` (no run_sampling, no list[Path] needed).
**How to avoid:** Two branches in driver.py: Stage 1 branch calls `run_sampling()` then `parse_poses(config.output_dir / "poses")`; bypass branch calls `parse_poses(input_poses_dir)` directly.

### Pitfall 6: Failure counting — prep failures use pose_idx from ligand.py, not PoseRecord.pose_idx
**What goes wrong:** `prepare_ligand_batch` assigns `pose_idx` based on the enumeration index of `pdb_paths`, not the `PoseRecord.pose_idx`. If driver passes only surviving-parse PoseRecords' pdb_paths (not the full range 0..N), the indices in PoseFailure records from prep will be offset.
**Why it happens:** `args_list = [(idx, pdb_path, output_dir) for idx, pdb_path in enumerate(pdb_paths)]` — idx is position in the passed list, not the embedded pose number.
**How to avoid:** Build the pdb_paths list from records in sorted pose_idx order, or use a wrapper that passes `record.pose_idx` explicitly. The simplest fix: extract `[(record.pose_idx, record.pdb_path, output_dir) for record in records]` and call `_prepare_single_ligand` directly, or document that failure pose_idx in prep is relative to the input list.

---

## Driver.py Execution Order (Complete)

1. `write_metadata_skeleton(config, metadata_path)` — before anything else
2. If `input_poses_dir` is set:
   - `records, failures = parse_poses(input_poses_dir)`
3. Else:
   - `pose_paths = run_sampling(config)`
   - `records, failures = parse_poses(config.output_dir / "poses")`
4. Log parsing failure count
5. `receptor_pdbqt = prepare_receptor(config)` — always runs (no caching)
6. `maps_dir = generate_ad4_maps(config, receptor_pdbqt)` — always runs (AD4 always in v1)
7. `pdb_paths = [r.pdb_path for r in records]`
8. `pdbqt_paths, prep_failures = prepare_ligand_batch(pdb_paths, config.output_dir / "pdbqt")`
9. Build `pdbqt_map: dict[Path, Path]` = {pdb_path: pdbqt_path}
10. Build `scored_poses: list[ScoredPose]` — only for poses where pdbqt_map[pdb_path] exists
11. `scored_vina, vina_failures = score_vina_batch(scored_poses, config, receptor_pdbqt, metadata_path=metadata_path)`
12. `scored_ad4, ad4_failures = score_ad4_batch(scored_vina, maps_dir)`
13. `cal = load_calibration(calibration_path)`; extract `alpha`, `beta`
14. For each pose in scored_ad4: `apply_hybrid_score(pose, alpha=alpha, beta=beta, n_residues=len(config.peptide_sequence))`
15. `logger.info("Clustering and output: Phase 6/7 not yet implemented")`
16. `finalize_metadata(metadata_path, poses_generated=len(records))`
17. Return `scored_ad4`

---

## calibrate_alpha.py Integration Detail

The script's `main()` accepts `args: argparse.Namespace | None`. When called with a pre-built Namespace, it skips `parse_args()` and uses the provided Namespace directly. The Namespace fields it reads are:
- `args.scores_json` (Path | None) — must not be None or ValueError raised
- `args.training_csv` (Path)
- `args.output` (Path)
- `args.verbose` (bool)

The `calibrate` subcommand in Phase 5 per D-03 should accept `--training-csv`, `--pdbs-dir`, and `--output`. However, `calibrate_alpha.py` accepts `--scores-json` (pre-computed), not `--pdbs-dir`. The discrepancy: D-03 says "dispatches to calibrate_alpha.py logic," but the CONTEXT.md note says "Live Vina/AD4 scoring is wired in Phase 5 via hybridock-pep calibrate." This means Phase 5 should wire the full live-scoring calibrate path, not just the pre-computed scores path.

**Recommendation (Claude's discretion):** For Phase 5, the `calibrate` subcommand accepts `--training-csv`, `--pdbs-dir`, and `--output`. Since live scoring for calibration requires running the full pipeline on training complexes (a significant scope), and calibrate_alpha.py only handles pre-computed scores, the cleanest Phase 5 implementation is:
- Accept `--scores-json` on the CLI (matching calibrate_alpha.py's existing interface)
- Dispatch directly to `calibrate_alpha.main(namespace)`
- Add a note in help that live scoring via `--pdbs-dir` is Phase 8 scope

This avoids implementing a mini-pipeline inside the calibrate subcommand that duplicates the dock subcommand logic.

---

## Project Constraints (from CLAUDE.md)

- **Python 3.11** for all in-repo code in score-env. Type hints everywhere with `from __future__ import annotations` at top of every module.
- **No bare `except:`** — catch specific exceptions. The existing codebase uses `except Exception as e:  # noqa: BLE001` with explicit justification comments where broad catching is required.
- **No `print()`** — `logging` module only.
- **CLI flag names are public interface** — do not rename `--peptide`, `--receptor`, `--site`, `--box`, `--n-samples`, `--scoring`, `--refine-topk`, `--output-dir` from CLAUDE.md §5.
- **Validate inputs before spawning subprocesses** — DockConfig construction (which calls receptor_path.exists()) must happen before any subprocess. Failing 30 minutes in is unacceptable UX.
- **All paths crossing conda boundary must be `.resolve()`d** — established in Phase 4; driver.py passes config (which already has absolute paths from DockConfig validators) to run_sampling, which resolves internally.
- **mypy strict mode** — all new code must pass mypy. Use `from __future__ import annotations` and explicit type annotations everywhere.
- **Ruff linting, Black formatting**, line length 100.
- **Docstrings in Google style** with at least `Args`, `Returns`, `Raises` sections.
- **Conventional Commits** (`feat:`, `fix:`, etc.); one logical change per commit.
- **No ADFRsuite/AutoDock4 binaries committed** — non-redistributable licenses.

---

## Environment Availability

Step 2.6: SKIPPED for this phase — Phase 5 adds no new external tool dependencies. All tools it orchestrates (conda, prepare_receptor4.py, autogrid4, vina) were already dependencies from Phases 2–4.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥ 8.0 + pytest-cov ≥ 5.0 |
| Config file | pyproject.toml (no pytest.ini; pytest finds tests/ by convention) |
| Quick run command | `pytest tests/test_cli.py -x` |
| Full suite command | `pytest --cov=hybridock_pep` |

### Existing Test Files (verified)
- `tests/test_models.py` — DockConfig, PoseRecord, ScoredPose, PoseFailure
- `tests/test_prep.py` — prepare_receptor, prepare_ligand_batch, generate_ad4_maps
- `tests/test_scoring.py` — score_vina_batch, score_ad4_batch, apply_hybrid_score, load_calibration
- `tests/test_sampling.py` — run_sampling, parse_poses
- `tests/test_output.py` — write_metadata_skeleton, finalize_metadata

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLI-01 | `hybridock-pep dock/calibrate/benchmark/prep` subcommands are dispatched | unit | `pytest tests/test_cli.py::TestSubcommands -x` | ❌ Wave 0 |
| CLI-01 | `dock` with all required flags produces a DockConfig | unit | `pytest tests/test_cli.py::TestDockSubcommand -x` | ❌ Wave 0 |
| CLI-01 | `calibrate` dispatches to calibrate_alpha.main | unit | `pytest tests/test_cli.py::TestCalibrateSubcommand -x` | ❌ Wave 0 |
| CLI-01 | `prep` dispatches to prepare_receptor | unit | `pytest tests/test_cli.py::TestPrepSubcommand -x` | ❌ Wave 0 |
| CLI-01 | `benchmark` raises NotImplementedError | unit | `pytest tests/test_cli.py::TestBenchmarkSubcommand -x` | ❌ Wave 0 |
| CLI-02 | Bad peptide sequence exits with code 2 before subprocess | unit | `pytest tests/test_cli.py::TestValidation -x` | ❌ Wave 0 |
| CLI-02 | Missing receptor path exits with code 2 before subprocess | unit | `pytest tests/test_cli.py::TestValidation -x` | ❌ Wave 0 |
| CLI-02 | Negative box_size exits with code 2 | unit | `pytest tests/test_cli.py::TestValidation -x` | ❌ Wave 0 |
| CLI-03 | `--seed N` stored in DockConfig.seed and appears in metadata | unit | `pytest tests/test_cli.py::TestSeed -x` | ❌ Wave 0 |
| CLI-01 | `--input-poses` bypasses run_sampling, calls parse_poses | unit | `pytest tests/test_driver.py::TestInputPosesBypass -x` | ❌ Wave 0 |
| CLI-02 | `--input-poses` + `--n-samples > 0` raises parser error | unit | `pytest tests/test_cli.py::TestMutualExclusion -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_cli.py tests/test_driver.py -x`
- **Per wave merge:** `pytest --cov=hybridock_pep`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_cli.py` — covers CLI-01, CLI-02, CLI-03
- [ ] `tests/test_driver.py` — covers driver.py orchestration, --input-poses bypass, failure handling

*(All existing test files cover Phases 1–4 modules; no test file exists yet for cli.py or driver.py)*

---

## Security Domain

`security_enforcement` not explicitly set to false in config.json — treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | CLI tool; no auth layer |
| V3 Session Management | no | Stateless pipeline runs |
| V4 Access Control | no | Single-user local tool |
| V5 Input Validation | yes | DockConfig Pydantic validators (peptide chars, receptor path exists, box_size > 0) |
| V6 Cryptography | no | SHA256 for provenance checksums only — stdlib hashlib |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `--receptor` or `--output-dir` | Tampering | DockConfig resolves paths; subprocess calls use absolute paths only |
| Shell injection via peptide sequence | Tampering | peptide_sequence is passed as a positional argument to subprocess, not interpolated into a shell string; subprocess.Popen called with list form (not shell=True) |
| Untrusted calibration.json | Tampering | load_calibration() validates alpha and beta bounds; JSON parsed with stdlib json |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `calibrate` subcommand in Phase 5 should accept `--scores-json` (not `--pdbs-dir`) to match calibrate_alpha.py's existing interface | calibrate_alpha.py Integration Detail | If user expects `--pdbs-dir` live-scoring flow, calibrate subcommand won't do what they expect; but this is explicitly "Claude's Discretion" per CONTEXT.md |

**All other claims verified by reading source files directly.**

---

## Open Questions

1. **Missing driver.py — no existing file to expand from**
   - What we know: driver.py is listed in CLAUDE.md §3 architecture and CONTEXT.md as a new file. No stub exists.
   - What's unclear: Nothing — the interface is fully specified.
   - Recommendation: Create from scratch; it's a pure wiring module.

2. **`--calibration` flag on dock subcommand**
   - What we know: driver.py calls `load_calibration(path)` to get alpha/beta. The path must come from somewhere.
   - What's unclear: CLAUDE.md §5 does not list `--calibration` as a dock flag, but the scoring flow requires it.
   - Recommendation: Add `--calibration` flag with default `data/calibration.json` so the common case works without specifying it, matching the calibrate_alpha.py default output path.

---

## Sources

### Primary (HIGH confidence)
- `src/hybridock_pep/cli.py` — existing stub read directly
- `src/hybridock_pep/models.py` — DockConfig, ScoredPose, PoseRecord, PoseFailure signatures read directly
- `src/hybridock_pep/sampling/rapidock_runner.py` — run_sampling() signature and behavior read directly
- `src/hybridock_pep/sampling/pose_io.py` — parse_poses() signature read directly
- `src/hybridock_pep/prep/receptor.py` — prepare_receptor() signature read directly
- `src/hybridock_pep/prep/ligand.py` — prepare_ligand_batch() signature and pdb_paths input type verified
- `src/hybridock_pep/prep/grids.py` — generate_ad4_maps() signature read directly
- `src/hybridock_pep/scoring/vina.py` — score_vina_batch() signature and ScoredPose input requirement verified
- `src/hybridock_pep/scoring/ad4.py` — score_ad4_batch() signature and maps_dir parameter verified
- `src/hybridock_pep/scoring/entropy.py` — apply_hybrid_score() assertion requirements verified; load_calibration() dict schema verified
- `src/hybridock_pep/output/metadata.py` — write_metadata_skeleton() and finalize_metadata() signatures read directly
- `scripts/calibrate_alpha.py` — main() signature, Namespace fields, and import-as-module pattern verified
- `.planning/phases/05-cli-driver/05-CONTEXT.md` — locked decisions D-01 through D-05
- `CLAUDE.md` §3, §4, §5, §7 — flag names, conventions, constraints
- `pyproject.toml` — entry point, dependencies, test framework

### Secondary (MEDIUM confidence)
- None required — all claims verified from source

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Interface signatures: HIGH — read directly from source
- Driver execution order: HIGH — derived from module contracts (assert requirements, input types)
- calibrate dispatch pattern: MEDIUM — "Claude's Discretion" per CONTEXT.md; import-as-module recommended but not locked
- Test strategy: HIGH — existing test patterns in test_sampling.py and test_prep.py confirm mock-based approach works

**Research date:** 2026-04-23
**Valid until:** 2026-05-23 (stable — all modules are implemented and their interfaces won't change)
