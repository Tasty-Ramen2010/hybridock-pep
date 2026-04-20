---
phase: 01-foundation
plan: "02"
subsystem: package-scaffold
tags: [pydantic, dataclasses, cli, models, packaging]
dependency_graph:
  requires: [01-01]
  provides: [hybridock_pep package, DockConfig, PoseRecord, ScoredPose, PoseFailure, hybridock-pep CLI entry point]
  affects: [all phases — downstream phases import from hybridock_pep]
tech_stack:
  added: [pydantic>=2.0, setuptools>=68, argparse]
  patterns: [Pydantic v2 BaseModel with field_validator/model_validator, @dataclass inheritance, PEP 621 pyproject.toml, PEP 561 py.typed]
key_files:
  created:
    - pyproject.toml
    - src/hybridock_pep/__init__.py
    - src/hybridock_pep/py.typed
    - src/hybridock_pep/models.py
    - src/hybridock_pep/cli.py
    - src/hybridock_pep/prep/__init__.py
    - src/hybridock_pep/sampling/__init__.py
    - src/hybridock_pep/scoring/__init__.py
    - src/hybridock_pep/analysis/__init__.py
    - src/hybridock_pep/output/__init__.py
    - tests/__init__.py
    - tests/test_models.py
    - .gitignore
  modified: []
decisions:
  - "DockConfig frozen=True (immutable after construction) — config passed across subprocess boundary should never mutate"
  - "run_id auto-generated via @model_validator(mode='before') so it resolves before field-level validation fires"
  - "ScoredPose extends PoseRecord via @dataclass inheritance — all parent fields required, all child fields defaulted to None/False"
  - "py.typed is empty file (PEP 561 marker only)"
  - "Subpackage __init__ stubs are empty — downstream phases own the content"
  - "Added .gitignore as deviation Rule 2 — missing gitignore would leave egg-info/pycache untracked after pip install"
metrics:
  duration: "4 min"
  completed: "2026-04-20"
  tasks_completed: 3
  files_created: 13
---

# Phase 1 Plan 2: Package Scaffold, Core Models, and CLI Stub Summary

**One-liner:** Pydantic v2 DockConfig + three @dataclass poses wired into a pip-installable package with four-subcommand argparse CLI stub.

## What Was Built

### Task 1 — pyproject.toml + directory skeleton (commit d6f1ffb)

- `pyproject.toml`: PEP 621 metadata using setuptools>=68 + wheel. No `readme = "README.md"` (Phase 8 scope). Optional dep groups: `scoring` (meeko, vina), `mmgbsa` (openmm), `dev` (pytest, ruff, black, mypy). Entry point: `hybridock-pep = "hybridock_pep.cli:main"`. Ruff/black line-length 100. mypy strict.
- `src/hybridock_pep/__init__.py`: re-exports all four public types from models.py with `__all__`.
- `src/hybridock_pep/py.typed`: empty PEP 561 marker.
- Five empty subpackage `__init__.py` stubs: prep, sampling, scoring, analysis, output.
- `tests/__init__.py`: empty.

### Task 2 — models.py (commit d98f0f5)

DockConfig field inventory as committed:

| Field | Type | Default | Validation |
|-------|------|---------|-----------|
| peptide_sequence | str | required | uppercase coercion; chars in ACDEFGHIKLMNPQRSTVWY only |
| receptor_path | Path | required | must exist on disk |
| site_coords | tuple[float, float, float] | required | none |
| box_size | float | required | must be > 0 |
| n_samples | int | 100 | must be > 0 |
| seed | int \| None | None | none |
| scoring | set[Literal["vina", "ad4"]] | {"vina", "ad4"} | none |
| output_dir | Path | required | none |
| run_id | str | auto-generated | timestamp_sha1[:8] of seed string |
| verbosity | int | 0 | none |

PoseRecord fields: `pose_idx: int`, `pdb_path: Path`, `sequence: str`, `ca_coords: np.ndarray`

ScoredPose fields (inherits PoseRecord): `vina_score`, `ad4_score`, `entropy_correction`, `hybrid_score`, `cluster_id`, `pdbqt_path` (all defaulted to None), `is_ad4_anomaly: bool = False`, `is_clipped: bool = False`

PoseFailure fields: `pose_idx: int`, `stage: Literal["parsing","prep","scoring","clustering"]`, `error_msg: str`

### Task 3 — cli.py + tests/test_models.py + install verification (commit 848a39c)

- `cli.py`: argparse with `dock`, `calibrate`, `benchmark`, `prep` subcommands. Subcommand bodies are Phase 1 stubs (raise via `parser.error()`). No print statements; logging module throughout.
- `tests/test_models.py`: 8 tests across 4 classes. All pass.
- `pip install -e .` succeeds. `hybridock-pep --help` lists all four subcommands.

### Bonus — .gitignore (commit 6c3474e)

Added .gitignore covering Python bytecode, egg-info, dist/build, venv, pytest cache, .DS_Store, and runs/. (See Deviations.)

## pytest Output

```
8 passed in 0.17s

tests/test_models.py::TestDockConfig::test_valid_construction PASSED
tests/test_models.py::TestDockConfig::test_uppercases_sequence PASSED
tests/test_models.py::TestDockConfig::test_invalid_peptide_sequence PASSED
tests/test_models.py::TestDockConfig::test_missing_receptor PASSED
tests/test_models.py::TestDockConfig::test_nonpositive_box_size PASSED
tests/test_models.py::TestPoseRecord::test_construction PASSED
tests/test_models.py::TestScoredPose::test_is_pose_record PASSED
tests/test_models.py::TestPoseFailure::test_construction PASSED
```

## Deviations from Plan

### Auto-added: .gitignore

**Rule:** Rule 2 (missing critical functionality)
**Found during:** Task 3 post-commit check
**Issue:** `pip install -e .` generates `src/hybridock_pep.egg-info/` and `__pycache__/` directories. Without a .gitignore these appear as untracked files in every subsequent `git status` and risk being accidentally committed.
**Fix:** Created `.gitignore` covering standard Python build artifacts, macOS .DS_Store, and runs/ output directory.
**Files modified:** `.gitignore` (new file)
**Commit:** 6c3474e

## Known Stubs

- `src/hybridock_pep/cli.py`: all four subcommand bodies call `parser.error()` — intentional Phase 1 stub. Phase 5 (`dock`), Phase 3 (`calibrate`), Phase 8 (`benchmark`), Phase 2 (`prep`) own the implementations.
- `src/hybridock_pep/prep/__init__.py` through `output/__init__.py`: all empty — intentional, later phases populate these subpackages.

## Threat Flags

None. No network endpoints, auth paths, file access patterns beyond local Path validation, or schema changes introduced.

## Self-Check: PASSED

Files confirmed:
- pyproject.toml: FOUND
- src/hybridock_pep/models.py: FOUND
- src/hybridock_pep/cli.py: FOUND
- src/hybridock_pep/__init__.py: FOUND
- tests/test_models.py: FOUND

Commits confirmed:
- d6f1ffb: FOUND (chore: pyproject.toml + skeleton)
- d98f0f5: FOUND (feat: core models)
- 848a39c: FOUND (feat: CLI stub + tests)
- 6c3474e: FOUND (chore: .gitignore)
