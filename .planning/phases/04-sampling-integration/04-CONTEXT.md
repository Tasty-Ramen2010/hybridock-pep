# Phase 4: Sampling Integration - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the RAPiDock subprocess wrapper, PDB pose parser, and provenance metadata writer. Delivers:
- `src/hybridock_pep/sampling/rapidock_runner.py` — score-env driver that calls rapidock-env via `conda run`
- `src/hybridock_pep/sampling/run_rapidock.py` — thin Python 3.9 wrapper script executed *inside* rapidock-env; imports and calls RAPiDock's Python API
- `src/hybridock_pep/sampling/pose_io.py` — parses RAPiDock PDB output into `list[PoseRecord]`
- `src/hybridock_pep/output/metadata.py` — writes `run_metadata.json` with full provenance

Phase ends when 100 poses are generated, parsed, and metadata written. Scoring (Phase 3), clustering (Phase 6), and CLI orchestration (Phase 5) are NOT in scope.

</domain>

<decisions>
## Implementation Decisions

### Subprocess streaming
- **D-01:** `rapidock_runner.py` uses `subprocess.Popen` with a `readline()` loop on stdout. stderr is captured on a **separate daemon thread** running its own `readline()` loop. Both emit to `logger` in real time.
  - Rationale: GPU OOM errors appear on stderr — a dedicated thread ensures they surface immediately even if stdout is quiet. Avoids stdout/stderr interleaving confusion in logs.
- **D-02:** Do NOT use `asyncio` subprocess or `communicate()` — both buffer output and fail the real-time OOM requirement.
- **D-03:** Every line read from stdout/stderr is logged at DEBUG level. Non-zero returncode after Popen.wait() raises a `RuntimeError` with the exit code.

### RAPiDock invocation interface
- **D-04:** `rapidock_runner.py` (score-env, Python 3.11) calls `conda run --no-capture-output -n rapidock-env python {abs_path_to_run_rapidock}` with args passed as CLI flags.
- **D-05:** `run_rapidock.py` lives at `src/hybridock_pep/sampling/run_rapidock.py`. It is executed *inside* rapidock-env (Python 3.9). Its absolute path is resolved in score-env via `Path(__file__).resolve()` and passed to `conda run`. This avoids any PATH ambiguity across the conda boundary.
- **D-06:** **CRITICAL — `run_rapidock.py` must be strictly Python 3.9 compatible.** No `match`/`case`, no `X | Y` type unions, no walrus operator in comprehensions, no `TypeAlias`. Researcher must inspect the RAPiDock repo to identify the actual callable entry point (class instantiation, function call, or top-level script) **before writing any code**. Do not assume `python -m rapidock` or a CLI entry point exists.
- **D-07:** All file paths passed across the `conda run` boundary (receptor PDBQT, output_dir, run_rapidock.py itself) MUST be absolute paths resolved via `Path(...).resolve()`. Relative paths will break silently inside conda's subprocess working directory.
- **D-08:** Seed is passed as a CLI arg to `run_rapidock.py` (e.g. `--seed N`). If `DockConfig.seed` is None, no seed flag is passed and non-determinism is noted in `run_metadata.json`.

### Pose count tolerance
- **D-09:** If RAPiDock produces fewer poses than `DockConfig.n_samples`, the pipeline **warns and continues** — it does NOT abort. Log a `WARNING` with the shortfall count.
- **D-10:** `run_metadata.json` records both `poses_requested: int` and `poses_generated: int` so downstream tools can detect shortfall without re-counting files.
- **D-11:** Zero poses generated → raise `RuntimeError` (this is always a hard failure, not a shortfall).

### PDB parsing (pose_io.py)
- **D-12:** `pose_io.py` parses all `poses/pose_*.pdb` files into `list[PoseRecord]`. Malformed or truncated PDB files → `PoseFailure(pose_idx, stage="parsing", error_msg=...)`. Batch never raises; returns `(list[PoseRecord], list[PoseFailure])` consistent with prep and scoring patterns.
- **D-13:** Cα coordinates extracted at parse time and stored in `PoseRecord.ca_coords: np.ndarray` (shape `[n_residues, 3]`). No lazy loading — clustering (Phase 6) needs O(1) access without re-reading 100 PDB files.
- **D-14:** Sequence extracted from SEQRES records if present; falls back to residue names from ATOM records. If neither is parseable, `PoseFailure` is emitted.

### Metadata (output/metadata.py)
- **D-15:** `run_metadata.json` is written **twice**: a skeleton at sampling start (status: "running"), then overwritten at completion (status: "complete"). Crash during sampling leaves a partial metadata file with status "running" — diagnosable without re-running.
- **D-16:** Required fields in `run_metadata.json` (from SAMP-02):
  - `git_sha` — current repo HEAD SHA
  - `rapidock_commit_sha` — SHA of the installed RAPiDock package (read from pip show or the git log inside the env)
  - `cli_args` — full dict of DockConfig fields
  - `seed` — integer or null
  - `vina_version` — from `vina --version`
  - `openmm_version` — from openmm.__version__
  - `cuda_version` — from torch.version.cuda (queried inside rapidock-env)
  - `receptor_sha256` — SHA256 of the input receptor PDB
  - `peptide_sequence_hash` — SHA256 of the peptide sequence string
  - `timestamp_start` — ISO 8601
  - `timestamp_end` — ISO 8601 (added on completion)
  - `poses_requested` — DockConfig.n_samples
  - `poses_generated` — actual count from pose_io.py
  - `status` — "running" | "complete" | "failed"
- **D-17:** `metadata.py` lives in `src/hybridock_pep/output/` (the `output/` module stub already exists). This is Phase 4's contribution to the output module.

### Claude's Discretion
- Exact argparse flag names in `run_rapidock.py` (must pass peptide, receptor, output_dir, seed, n_samples — names are Claude's call)
- Whether to use `threading.Thread` or `concurrent.futures.ThreadPoolExecutor` for the stderr daemon thread
- Exact PDB ATOM record parsing logic (Biopython PDBParser vs manual ATOM line parsing — pick whichever is lighter; Biopython is already a likely dep via MDAnalysis in rapidock-env, but pose_io runs in score-env)
- rapidock_commit_sha discovery strategy (pip show rapidock, or subprocess `git -C $(python -c "import rapidock; ...") log -1 --format=%H`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Technical specification
- `docs/HybriDock-Pep_Technical_Specification.pdf` §11, §12 — Sampling pipeline architecture, subprocess orchestration design
- `docs/HybriDock-Pep_Technical_Specification.pdf` §16 — Known workarounds (ref2015 cysteine, PULCHRA version)

### Requirements
- `.planning/REQUIREMENTS.md` SAMP-01 — `conda run --no-capture-output`, real-time OOM surfacing, exact 100-pose output
- `.planning/REQUIREMENTS.md` SAMP-02 — seed propagation into RAPiDock, full provenance metadata fields

### Project constraints
- `CLAUDE.md` §2.4 — Two envs; driver in score-env calls rapidock-env via subprocess + conda run
- `CLAUDE.md` §4 — **CRITICAL:** `run_rapidock.py` is executed by rapidock-env (Python 3.9). No Python 3.10+ syntax.
- `CLAUDE.md` §7 (Before touching the RAPiDock subprocess wrapper) — verify CUDA/PyTorch combo, preserve seed propagation, no GPU parallelism, absolute paths only
- `CLAUDE.md` §2.2 — RTX 5070 is Blackwell CC 12.0; PyTorch 2.7 + CUDA 12.8 confirmed in STATE.md

### Prior phase context
- `.planning/phases/01-foundation/01-CONTEXT.md` D-05–D-06 — PoseRecord fields: `pose_idx`, `pdb_path`, `sequence`, `ca_coords`; `PoseFailure` fields: `pose_idx`, `stage`, `error_msg`
- `.planning/phases/01-foundation/01-CONTEXT.md` D-10 — rapidock-env: Python 3.9, PyTorch 2.7, CUDA 12.8
- `src/hybridock_pep/models.py` — authoritative PoseRecord and PoseFailure definitions

### Blockers to check at Phase 4 start (from STATE.md)
- fair-esm 2.0.0 import against PyTorch 2.7 is unverified — validate before writing any sampling code
- PyG cu128 prebuilt wheels for PyTorch 2.7.0 may not exist — have source build fallback ready

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/hybridock_pep/models.py` `PoseRecord` — fields already defined; pose_io.py fills them, do not add new fields without checking models.py first
- `src/hybridock_pep/models.py` `PoseFailure(stage="parsing")` — use for per-pose parse failures; same pattern as prep and scoring
- `src/hybridock_pep/prep/ligand.py` `prepare_ligand_batch()` — reference implementation for collect-all-failures batch with (results, failures) return; pose_io.py follows same pattern
- `src/hybridock_pep/sampling/__init__.py` — stub exists, empty
- `src/hybridock_pep/output/__init__.py` — stub exists, empty

### Established Patterns
- `from __future__ import annotations` first line of every module (score-env modules only; run_rapidock.py in rapidock-env should also include it for Python 3.9 forward-compat)
- `logger = logging.getLogger(__name__)` + log full command string before every subprocess call
- No bare `except:` — catch specific exceptions, reraise with context
- Google-style docstrings with Args, Returns, Raises
- Type hints everywhere; mypy strict mode (score-env modules)

### Integration Points
- `rapidock_runner.py` → reads `DockConfig` (site_coords, n_samples, seed, output_dir, receptor_path, peptide_sequence, run_id); writes poses to `output_dir/poses/pose_{i}.pdb`
- `pose_io.py` → reads `output_dir/poses/pose_*.pdb`; returns `list[PoseRecord]` for Phase 5 driver and Phase 6 clustering
- `output/metadata.py` → reads `DockConfig` + sampling results; writes `output_dir/run_metadata.json`
- Phase 5 driver will call `run_sampling()` → `parse_poses()` → `write_metadata()` in sequence

</code_context>

<specifics>
## Specific Ideas

- `run_rapidock.py` is a thin shim: accepts `--peptide`, `--receptor`, `--output-dir`, `--n-samples`, `--seed` via argparse, imports RAPiDock, calls its API, writes pose PDBs to the output dir. The researcher must read the RAPiDock source to determine the exact API call.
- Stderr daemon thread pattern: `t = threading.Thread(target=_stream_stderr, args=(proc.stderr,), daemon=True); t.start()` then readline loop on proc.stdout in main thread; `t.join()` after `proc.wait()`.
- Metadata skeleton written before `conda run` is launched — so even if the conda subprocess hangs indefinitely, the file exists with `status: "running"` and `timestamp_start`.

</specifics>

<deferred>
## Deferred Ideas

- Metadata write: No incremental-per-pose metadata — two writes (start skeleton + end update) is sufficient for v1.
- Configurable `min_poses` threshold in DockConfig — deferred to v2; warn-and-continue is the v1 behavior.
- GPU parallelism across multiple GPUs — explicitly out of scope per CLAUDE.md §7 (one GPU, sequential inference).

</deferred>

---

*Phase: 04-sampling-integration*
*Context gathered: 2026-04-21*
