# Phase 1: Foundation - Context

**Gathered:** 2026-04-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Bootstrap the entire project: both conda environments installable and verified, the Python package scaffold in place in score-env, and four core dataclasses defining the interfaces every later module plugs into. Also includes `scripts/smoke_test.sh` for dependency validation.

What this phase does NOT include: any actual scoring, sampling, CLI subcommand logic, or analysis. Those start in Phase 2+.

</domain>

<decisions>
## Implementation Decisions

### DockConfig — Pydantic BaseModel

- **D-01:** `DockConfig` is a **Pydantic BaseModel** (not a plain dataclass). Pydantic validates inputs on construction — catching bad peptide sequences, missing receptor paths, and out-of-range coordinates before anything is spawned. This is the first line of defence for CLI-02.
- **D-02:** `DockConfig` includes `output_dir: Path` and `run_id: str` (e.g., timestamp + seed hash). Config is self-contained — the driver can write `run_metadata.json` directly from the config object without needing extra context passed in.
- **D-03:** `scoring: set[Literal['vina', 'ad4']]` — type-safe set, default `{'vina', 'ad4'}`. Validated by Pydantic on construction. Check presence with `'ad4' in config.scoring`.
- **D-04:** Full field inventory for planner to spec out:
  - `peptide_sequence: str` — validated for standard amino acid characters
  - `receptor_path: Path` — validated to exist
  - `site_coords: tuple[float, float, float]` — binding site center
  - `box_size: float` — grid box edge length (Å)
  - `n_samples: int = 100`
  - `seed: int | None = None`
  - `scoring: set[Literal['vina', 'ad4']] = {'vina', 'ad4'}`
  - `output_dir: Path`
  - `run_id: str` — generated at construction (timestamp + seed hash)
  - `verbosity: int = 0`

### PoseRecord — parsed pose

- **D-05:** `PoseRecord` stores `ca_coords: np.ndarray` parsed once at PDB load time (shape `[n_residues, 3]`). No lazy loading — clustering (Phase 6) gets O(1) access without re-reading 100 PDB files.
- **D-06:** Field inventory:
  - `pose_idx: int`
  - `pdb_path: Path`
  - `sequence: str`
  - `ca_coords: np.ndarray`

### ScoredPose — inherits PoseRecord

- **D-07:** `ScoredPose(PoseRecord)` — inheritance, not composition. A ScoredPose IS a PoseRecord everywhere PoseRecord is expected. Parse → score is an upgrade, not a wrap.
- **D-08:** Additional fields beyond PoseRecord:
  - `vina_score: float | None = None`
  - `ad4_score: float | None = None`
  - `entropy_correction: float | None = None`
  - `hybrid_score: float | None = None`
  - `cluster_id: int | None = None`
  - `pdbqt_path: Path | None = None`
  - `is_ad4_anomaly: bool = False` — True if AD4 score is positive (flagged per SCORE-02)
  - `is_clipped: bool = False` — True if any atoms were outside grid bounds (logged per SCORE-01)

### PoseFailure — error record

- **D-09:** `PoseFailure` captures: `pose_idx: int`, `stage: Literal['parsing', 'prep', 'scoring', 'clustering']`, `error_msg: str`. No full traceback stored in the dataclass — sufficient for `run_metadata.json` diagnostics without code smell.

### Env pinning (locked from STATE.md)

- **D-10:** `rapidock-env`: Python 3.9, PyTorch 2.7, CUDA 12.8 (first native sm_120 — Blackwell RTX 5070). PyG, MDAnalysis, E3NN, RDKit, PyRosetta.
- **D-11:** `score-env`: Python 3.11, Vina 1.2.5+, OpenMM 8.1+, scikit-learn, Meeko, ADFRsuite binaries on PATH.
- **D-12:** Planner decides exact pin strictness (lower bounds preferred for portability, exact for reproducibility) — Claude's discretion.

### Smoke test (Claude's discretion)

- **D-13:** `scripts/smoke_test.sh` checks: CUDA compute capability ≥ (12, 0), ADFRsuite `prepare_receptor4.py` on PATH, Vina version ≥ 1.2.5. On macOS ARM (no CUDA), CUDA check should warn but not fail hard — cross-platform is a success criterion (CLAUDE.md §8). Exact skip/warn logic is Claude's discretion.

### Claude's Discretion

- Exact package versions in YML files beyond PyTorch 2.7 + CUDA 12.8 (lower bounds vs exact pins)
- macOS ARM smoke test behavior (skip CUDA check vs warn-only vs fail with message)
- pyproject.toml optional dependency groups (e.g., `[mmgbsa]`, `[dev]`)
- `run_id` generation format (timestamp + seed hash is the intent; exact format is Claude's call)
- Whether `DockConfig` uses `model_config = ConfigDict(frozen=True)` or stays mutable — either is fine for Phase 1

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Primary spec
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5, §11, §12, §16 — load-bearing sections; §5.6–5.7 explicitly reject Vina recompilation; §16.1 documents the ref2015 cysteine workaround
- `CLAUDE.md` §2 (non-negotiable constraints), §3 (architecture + target repo layout), §4 (dev conventions), §7 (pre-coding playbook)

### Environment constraints
- `CLAUDE.md` §2.2 — RTX 5070 is Blackwell CC 12.0, CUDA 12.4+ / PyTorch 2.3+ required (STATE.md pins to 2.7 + CUDA 12.8)
- `CLAUDE.md` §2.3 — PULCHRA must be v3.04 exactly (Bioconda ships 3.06 with a side-chain bug)
- `CLAUDE.md` §2.4 — Two separate envs; driver in score-env orchestrates rapidock-env via subprocess + conda run

### Requirements
- `.planning/REQUIREMENTS.md` — TEST-01 is the Phase 1 requirement; full traceability table included

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — `src/` is empty. Phase 1 creates the scaffold from scratch.

### Established Patterns
- None yet — Phase 1 establishes the patterns all later phases follow.

### Integration Points
- `src/hybridock_pep/__init__.py` must export `DockConfig`, `PoseRecord`, `ScoredPose`, `PoseFailure` at the package level (success criterion 4)
- `pyproject.toml` wires `hybridock-pep` CLI entry point to `hybridock_pep.cli:main` (success criterion 2)
- `scripts/smoke_test.sh` must be executable and source-agnostic (called from `bash scripts/smoke_test.sh`)

</code_context>

<specifics>
## Specific Ideas

- Pydantic was explicitly chosen over plain `@dataclass` for `DockConfig` — use Pydantic v2 (`pydantic>=2.0`) with standard validators (`@field_validator`). Don't downgrade to v1 style.
- `ScoredPose` inherits `PoseRecord` — since numpy arrays in dataclasses/Pydantic need care, use `model_config = ConfigDict(arbitrary_types_allowed=True)` if Pydantic models are used for PoseRecord/ScoredPose too, or use plain `@dataclass` for PoseRecord/ScoredPose and Pydantic only for DockConfig. Planner should decide this cleanly.
- Contact-zone Cα RMSD (not full-peptide) is the clustering metric — `ca_coords` stores ALL residue Cα; contact zone masking happens in the clustering module (Phase 6), not in PoseRecord.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-04-19*
