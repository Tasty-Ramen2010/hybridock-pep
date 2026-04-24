# Phase 5: CLI & Driver - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the full two-stage docking pipeline into a single `hybridock-pep` entry point. Phase 5 delivers:
- All four subcommands (`dock`, `calibrate`, `benchmark`, `prep`) with real arg definitions and help strings on every flag (with units)
- Pre-run input validation that fires in <1s before any subprocess launches
- `driver.py` orchestrating Stage 1 (RAPiDock via conda run) + Stage 2 (prep + scoring) in sequence
- `--input-poses` bypass so macOS users can skip Stage 1 and run scoring on pre-generated poses
- `calibrate` dispatches to existing calibrate_alpha.py logic; `prep` dispatches to prep/receptor.py; `benchmark` has real args + NotImplementedError (Phase 8 scope)

Clustering (Phase 6) and output file writing (Phase 7) are out of scope — driver.py stubs the handoff point cleanly.

</domain>

<decisions>
## Implementation Decisions

### --input-poses bypass (D-01)
- **D-01:** `hybridock-pep dock --input-poses <dir>` is **fully wired** in Phase 5. When present, driver.py skips the `run_sampling()` call (no conda run), calls `parse_poses(<dir>)` directly, and proceeds to Stage 2 (prep + scoring). This enables macOS users to run the full scoring pipeline on poses generated elsewhere.
- Flag goes on the `dock` subparser with help: `"Directory of pre-generated pose PDBs (skips RAPiDock Stage 1). Required on macOS."` No default — mutually exclusive with running Stage 1 (driver raises argparse error if both `--input-poses` and `--n-samples` >0 are ambiguous).

### Driver Stage 2 scope (D-02)
- **D-02:** `driver.py` wires what exists and stubs what doesn't:
  - Stage 1: `run_sampling(config)` → `list[Path]` (or reads from `--input-poses`)
  - Parse: `parse_poses(poses_dir)` → `(list[PoseRecord], list[PoseFailure])`
  - Prep: `prepare_ligand_batch(records, ...)` → `(list[PreparedPose], list[PoseFailure])`
  - Score: vina + ad4 + entropy per pose → `list[ScoredPose]`
  - Handoff: after scoring, driver logs `"Clustering and output: Phase 6/7 not yet implemented"` at INFO level and returns `list[ScoredPose]`. No exception raised — clean exit.
- Phase 6 will add a `cluster_poses()` call between scoring and return. Phase 7 will add `write_output()`. The stub structure makes those plug-ins trivial.

### calibrate / benchmark / prep subcommand depth (D-03)
- **D-03:** All three get real arg definitions with help strings (with units where applicable):
  - `calibrate`: `--training-csv`, `--pdbs-dir`, `--output` → dispatches to `scripts/calibrate_alpha.py` logic (already written as standalone script; Phase 5 wires it as a callable function or subprocess call)
  - `prep`: `--receptor <path>` → dispatches to `prep/receptor.py:prepare_receptor()`; outputs PDBQT to same dir
  - `benchmark`: `--test-csv`, `--baselines`, `--report` → raises `NotImplementedError("benchmark: Phase 8 scope")` with clear message
- All subcommands: every flag has a help string with units (e.g., `"Grid box edge length in Angstroms"`)

### Validation layer (D-04, carried from DockConfig)
- **D-04:** DockConfig Pydantic validators are the primary validation gate. The CLI builds a `DockConfig` from parsed args as the first thing in `_run_dock()`. Any invalid input (bad sequence, missing receptor, out-of-range coords) raises a Pydantic `ValidationError` which the CLI catches and formats as `argparse.error(str(e))` → exits with code 2 in <1s, before any subprocess.
- No duplicate validation logic in argparse or driver.py — DockConfig is the single source of truth.

### Logging and output (D-05, carried from prior phases)
- **D-05:** `logging` module throughout. `main()` configures `basicConfig` based on `-v` count (already done in stub). Every subprocess call logs the full command at DEBUG level. Driver logs stage transitions at INFO.

### Claude's Discretion
- How `calibrate_alpha.py` is invoked from the `calibrate` subcommand (import as module vs. subprocess) — Claude chooses based on what's cleanest given the existing script structure
- Exact arg name for the grid box center (`--site` vs `--site-center`) — use `--site` per CLAUDE.md example
- Exit code conventions beyond code 2 (argparse default) — Claude chooses consistent codes

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Primary spec
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5, §11, §12, §16 — authoritative source; wins over CLAUDE.md on conflicts

### CLI & architecture
- `CLAUDE.md` §3 (architecture diagram), §4 (dev conventions), §5 (common commands — exact flag names) — CLI flag names in §5 are the reference
- `CLAUDE.md` §7 ("Before refactoring the CLI") — flag rename rules and validation requirements

### Existing interfaces to wire
- `src/hybridock_pep/models.py` — DockConfig fields (peptide_sequence, receptor_path, site_coords, box_size, n_samples, seed, scoring, output_dir, run_id, verbosity)
- `src/hybridock_pep/sampling/rapidock_runner.py` — `run_sampling(config: DockConfig) -> list[Path]`
- `src/hybridock_pep/sampling/pose_io.py` — `parse_poses(poses_dir: Path) -> tuple[list[PoseRecord], list[PoseFailure]]`
- `src/hybridock_pep/prep/ligand.py` — `prepare_ligand_batch(...)` collect-all-failures pattern
- `src/hybridock_pep/prep/receptor.py` — `prepare_receptor(...)` for `prep` subcommand dispatch
- `src/hybridock_pep/scoring/vina.py`, `ad4.py`, `entropy.py` — per-pose scorers
- `src/hybridock_pep/output/metadata.py` — `write_metadata_skeleton()`, `finalize_metadata()`
- `scripts/calibrate_alpha.py` — existing calibration logic; `calibrate` subcommand dispatches here

### Phase 4 decisions (sampling architecture)
- `.planning/phases/04-sampling-integration/04-CONTEXT.md` — subprocess pattern, absolute paths, two-write metadata

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cli.py`: argparse stub with 4 subcommands already defined — Phase 5 fills in arg definitions and dispatch, doesn't rebuild the parser
- `DockConfig` (models.py): Pydantic frozen model with field validators for peptide, receptor path, coords — primary validation layer, no duplication needed
- `write_metadata_skeleton()` / `finalize_metadata()` (output/metadata.py): already handles the two-write provenance pattern; driver calls these at Stage 1 start and end
- `run_sampling()` (sampling/rapidock_runner.py): full implementation, returns `list[Path]`
- `parse_poses()` (sampling/pose_io.py): full implementation, returns `(list[PoseRecord], list[PoseFailure])`
- `calibrate_alpha.py` (scripts/): standalone script with full logic — needs to be made callable from CLI

### Established Patterns
- Collect-all-failures: `(results, failures)` tuple — all batch operations follow this; driver.py should log failure counts rather than abort on partial failures
- Absolute paths: all paths passed to subprocesses must be `.resolve()`d — established in Phase 4
- Lazy imports: Phase 3/4 pattern for score-env optional deps (vina, openmm)

### Integration Points
- `pyproject.toml`: `[project.scripts]` entry point `hybridock-pep = "hybridock_pep.cli:main"` — already wired from Phase 1
- `driver.py` is a new file; `cli.py` imports and calls it for the `dock` subcommand

</code_context>

<specifics>
## Specific Ideas

- CLAUDE.md §5 gives the exact flag surface for `hybridock-pep dock`: `--peptide`, `--receptor`, `--site` (3 floats), `--box`, `--n-samples`, `--scoring` (comma-separated), `--refine-topk`, `--output-dir`. Use these exact names.
- `--input-poses` is explicitly required for macOS support per CLAUDE.md §8.

</specifics>

<deferred>
## Deferred Ideas

- `--skip-sampling` flag: noted in STATE.md as v2 scope (OPT-02). `--input-poses` covers the Mac use case for Phase 5.
- MM-GBSA `--refine-topk` execution: flag is defined on `dock` for Phase 5, but actual MM-GBSA dispatch is v2 scope (OPT-01). Phase 5 validates the flag and stores in DockConfig; driver.py notes it's unimplemented.
- Parallel scoring across poses: v1 uses sequential scoring; parallelism is a v2 optimization.

</deferred>

---

*Phase: 05-cli-driver*
*Context gathered: 2026-04-23*
