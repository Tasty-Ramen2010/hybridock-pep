# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 01-foundation
**Areas discussed:** Dataclass field design

---

## Dataclass Field Design

### Gray area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Dataclass field design | DockConfig, PoseRecord, ScoredPose, PoseFailure field contracts | ✓ |
| Env pinning strictness | Exact pins vs. lower-bound constraints in YML files | |
| Smoke test scope | macOS ARM CUDA skip behavior, rapidock-env verification | |
| pyproject.toml structure | Optional dependency groups, iGEM metadata | |

---

## DockConfig — type choice

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen dataclass | Immutable after construction, no dependencies | |
| Mutable dataclass | Can be updated mid-run | |
| Pydantic BaseModel | Validates inputs on construction (peptide chars, path existence, coord ranges) | ✓ |

**User's choice:** Pydantic BaseModel
**Notes:** Explicit choice — validation on construction is the right trade-off for a scientific CLI where bad inputs 30 minutes into a GPU run is unacceptable UX.

---

## DockConfig — output_dir + run_id

| Option | Description | Selected |
|--------|-------------|----------|
| Include output_dir + run_id | Self-contained config object | ✓ |
| Inputs only, output_dir separate | Cleaner separation, two objects | |

**User's choice:** Include output_dir + run_id (recommended)

---

## PoseRecord/ScoredPose relationship

| Option | Description | Selected |
|--------|-------------|----------|
| Inheritance: ScoredPose(PoseRecord) | ScoredPose IS a PoseRecord | ✓ |
| Composition: ScoredPose has a PoseRecord field | More explicit, more verbose | |

**User's choice:** Inheritance (recommended)

---

## PoseRecord Cα coordinate storage

| Option | Description | Selected |
|--------|-------------|----------|
| Store ca_coords: np.ndarray at parse time | O(1) access for clustering | ✓ |
| Load lazily from pdb_path | Lightweight dataclass, repeated I/O during clustering | |

**User's choice:** Store in PoseRecord (recommended)

---

## PoseFailure stage granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Stage enum + error string | pose_idx, stage Literal, error_msg: str | ✓ |
| Stage + full exception info | Adds exception_type and traceback | |

**User's choice:** Stage enum + error string (recommended)

---

## DockConfig scoring field representation

| Option | Description | Selected |
|--------|-------------|----------|
| set[Literal['vina','ad4']] | Type-safe, Pydantic-validated, extensible | ✓ |
| Separate bool flags | use_vina, use_ad4 | |

**User's choice:** Set of Literals (recommended)

---

## Claude's Discretion

- Exact package pins in YML beyond PyTorch 2.7 + CUDA 12.8
- macOS ARM smoke test CUDA skip/warn behavior
- pyproject.toml optional dependency groups
- run_id generation format
- Whether DockConfig uses frozen=True Pydantic config

## Deferred Ideas

None.
