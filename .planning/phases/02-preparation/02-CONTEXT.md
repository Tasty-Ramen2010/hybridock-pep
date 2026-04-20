# Phase 2: Preparation Pipeline - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Automate receptor and per-pose ligand PDBQT preparation, and generate AD4 affinity maps via autogrid4. This phase ends when `hybridock-pep prep` produces valid PDBQT files for both receptor and 100 pose PDBs, and `receptor.HD.map` is verified to exist. Vina/AD4 scoring and RAPiDock sampling are NOT in scope.

</domain>

<decisions>
## Implementation Decisions

### Receptor prep workflow
- **D-01:** pdbfixer applies all three fixes in sequence: add missing atoms, add missing residues, add hydrogens at pH 7.4. No selective fixing — always the full set.
- **D-02:** Receptor PDBQT is always regenerated on each run — no caching, no mtime checks, no skip-if-exists logic.
- **D-03:** If `prepare_receptor4.py` exits non-zero, raise an exception immediately with the full stderr captured. Hard abort — no fallback, no retry.

### autogrid4 / GPF generation
- **D-04:** The `.gpf` (grid parameter file) is generated programmatically from `DockConfig` fields (`site_coords`, `box_size`, receptor PDBQT path). No template file on disk — constructed in Python and written to `output_dir/maps/receptor.gpf`.
- **D-05:** After autogrid4 completes, check for `output_dir/maps/receptor.HD.map`. If missing, raise `PrepError` with message: `"receptor.HD.map not found after autogrid4 — AD4 scoring will fail. Check your atom types in the GPF."` Hard abort.
- **D-06:** AD4 atom types in the GPF: `C A N O S H HD e d` — the full peptide set. Covers cysteine sulfur (S) required by LISDAELEAIFEADC and the HD polar-hydrogen type needed for the HD map guard.
- **D-07:** All autogrid4 outputs (`.map`, `.gpf`, `.glg`) are written to `output_dir/maps/` subdirectory. Created by the prep code if absent.

### Claude's Discretion
- Ligand batch parallelism: executor type (ProcessPoolExecutor vs ThreadPoolExecutor), degree of parallelism, per-pose error handling strategy (collect all failures vs fail-fast).
- Test fixture strategy: minimal toy PDB content is fine; deterministic small files only.
- Exact GPF grid spacing value (standard 0.375 Å unless spec says otherwise).

</decisions>

<specifics>
## Specific Ideas

- The HD map guard error message is verbatim from discussion: `"receptor.HD.map not found after autogrid4 — AD4 scoring will fail. Check your atom types in the GPF."`
- pdbfixer is already in `score-env.yml` (version ≥1.9) — no new dependency needed.
- Meeko is already in `score-env.yml` (version ≥0.5) — ligand prep uses it directly.
- `PrepError` should be a custom exception class defined in `prep/` (e.g., `prep/errors.py` or at top of the relevant module).

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Technical specification
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5 — Non-negotiable constraints including charge handling, environment split, PULCHRA version
- `docs/HybriDock-Pep_Technical_Specification.pdf` §16.1 — Cysteine/ref2015 workaround (relevant to LISDAELEAIFEADC prep)

### Requirements
- `.planning/REQUIREMENTS.md` PREP-01 — Receptor PDBQT via pdbfixer + prepare_receptor4.py, no manual steps
- `.planning/REQUIREMENTS.md` PREP-02 — Meeko batch ligand PDBQT, parallelized, no file left behind
- `.planning/REQUIREMENTS.md` PREP-03 — autogrid4 map generation + HD map existence guard

### Phase 1 context (locked interfaces)
- `.planning/phases/01-foundation/01-CONTEXT.md` — DockConfig field inventory; `receptor_path`, `site_coords`, `box_size`, `output_dir` are the inputs to prep

### Project constraints
- `CLAUDE.md` §2.1 — Vina does NOT use partial charges; AD4 does (via `vina --scoring ad4`). Prep must produce Gasteiger-charged PDBQT for ligands.
- `CLAUDE.md` §2.6 — ADFRsuite is non-redistributable; link to download in INSTALL.md, never bundle.
- `CLAUDE.md` §2.4 — All prep code runs in `score-env` (Python 3.11). `rapidock_runner.py` is the only 3.9-target file.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/hybridock_pep/models.py` `DockConfig` — frozen Pydantic v2 model; fields `receptor_path` (validated Path), `site_coords` (tuple[float,float,float]), `box_size` (float), `output_dir` (Path), `run_id` (str UUID) are the inputs the prep modules receive.
- `src/hybridock_pep/models.py` `PoseRecord` — output type for pose I/O; prep produces the PDBQT files that later populate these records.

### Established Patterns
- Subprocess calls use `logging` at INFO level with the full command before execution (see CLAUDE.md §4).
- Exceptions: no bare `except:`, catch specific types, reraise with context.
- `from __future__ import annotations` at top of every module.
- Type hints everywhere; mypy strict mode on CI.

### Integration Points
- `prep/receptor.py` → called by `driver.py` Stage 2 init, takes `DockConfig`, writes `output_dir/receptor.pdbqt`
- `prep/ligand.py` → called per-pose batch, takes list of PDB paths + output dir, writes `*.pdbqt` alongside each PDB
- `prep/grids.py` → called after receptor prep, takes `DockConfig` + receptor PDBQT path, writes to `output_dir/maps/`, raises `PrepError` on HD map missing
- `cli.py` `prep` subcommand → entry point wiring (stub already exists)

</code_context>

<deferred>
## Deferred Ideas

- Optional `--skip-prep` / `--reuse-pdbqt` flag to skip regeneration when poses already exist — deferred to Phase 5 (CLI & Driver).
- MM-GBSA OpenMM minimization before Vina scoring (CLAUDE.md §2.5 workaround) — deferred to Phase 3 (Scoring Core).
- PULCHRA v3.04 rebuild/pin — relevant only if ADCP output is used as input; not in Phase 2 scope.

</deferred>

---

*Phase: 02-preparation*
*Context gathered: 2026-04-20*
