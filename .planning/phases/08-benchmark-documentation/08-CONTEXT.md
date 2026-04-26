# Phase 8: Benchmark & Documentation - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 delivers the final v1 release artifacts: the `scripts/benchmark.py` harness against 10
held-out peptide-protein complexes, `README.md`, `docs/architecture.md`, `docs/tutorial.ipynb`
(pre-run with outputs), and a committed license audit at `docs/licenses.txt`. This phase does NOT
add new pipeline features — it validates and documents what Phases 1–7 built.

Requirements: TEST-03, DOCS-01, DOCS-02, DOCS-03, DOCS-04

</domain>

<decisions>
## Implementation Decisions

### Benchmark Harness (TEST-03)

- **D-01:** `scripts/benchmark.py` runs the **full pipeline per complex**: downloads PDB, preps
  receptor, runs `hybridock-pep dock` (full Stage 1+2 via CLI), scores all poses, clusters, and
  reports best ΔG. It is **not** score-only — Stage 1 (RAPiDock) is included in the benchmark
  invocation on the RTX 5070 machine.

- **D-02:** The benchmark is **not gated in CI**. ADFRsuite is not on PATH in the dev environment.
  The harness code is tested structurally (imports, argument parsing, output format), but benchmark
  execution requires score-env + ADFRsuite + GPU. Execution happens on the RTX 5070 machine and
  results are committed manually.

- **D-03:** `scripts/benchmark.py` produces two output files:
  - `benchmark_report.md` — formatted Markdown with per-complex table (complex, HybriDock ΔG,
    Vina-alone ΔG, experimental pKd, RMSD if available), Pearson r summary, pass/fail for
    r ≥ 0.55 and +0.10 improvement over Vina-alone.
  - `benchmark_results.csv` — raw numbers (pdb_id, hybrid_score, vina_score, pearson_r,
    n_poses, runtime_s).
  Both files are written to `--output-dir` (default `runs/benchmark/`). Committed results go in
  `data/benchmark_results/` (pre-run, < 1 MB total).

- **D-04:** Vina-alone baseline computed by `benchmark.py` itself: run `hybridock-pep dock` with
  `--scoring vina` only (no AD4, no entropy correction) and record `vina_score` of best pose.
  This gives a controlled comparison from the same PDB/pose set.

### 10-Complex Benchmark Dataset (data/test_complexes.csv)

- **D-05:** The 10 test complexes are **fully held-out** — none overlap with `training_complexes.csv`
  (2OY2, 1YCR, 3LNJ). The researcher agent must curate the list from literature.

- **D-06:** Selection criteria for the 10 complexes:
  - Peptide-protein crystal structure in PDB with resolution ≤ 2.5 Å
  - Experimental Kd (or IC50 convertible to Kd) published and verifiable
  - Peptide length 8–20 residues (within RAPiDock's training distribution)
  - Diverse targets: not all MDM2/p53 family — include BclXL, XIAP, calmodulin, PDZ
    domains, MHC, or similar well-characterized peptide binders
  - pKd range 5.0–10.0 (avoids edge cases outside the model's sensitivity range)

- **D-07:** `data/test_complexes.csv` schema mirrors `training_complexes.csv`:
  `pdb_id,peptide_sequence,experimental_pkd`. The researcher agent populates this file with
  verified entries. No placeholder or synthetic data.

### Tutorial Notebook (DOCS-04)

- **D-08:** `docs/tutorial.ipynb` ships **pre-run with outputs committed** — plots, CSV previews,
  and printed scores are all saved in the notebook JSON. Fresh clone shows results without any
  install step. The pre-run is on MDM2/p53 (PDB 2OY2 / peptide ETFSDLWKLLPE), reusing
  `tests/fixtures/mdm2_p53/` fixture poses to avoid GPU dependency.

- **D-09:** Tutorial section structure:
  1. **Introduction** — what HybriDock-Pep does and why (2-3 paragraphs)
  2. **Installation** — `conda env create` + `pip install -e .` commands (code cell, no live output)
  3. **Receptor preparation** — `hybridock-pep prep` on MDM2 receptor (code + pre-run output)
  4. **Docking run** — `hybridock-pep dock` with fixture poses via `--input-poses`
     (code + pre-run output showing stage progress)
  5. **Reading output** — `pandas.read_csv('ranked_poses.csv')` table display, convergence plot
     image embed, ΔG summary
  6. **Interpreting results** — explain hybrid score vs Vina-alone, cluster quality,
     what ΔG < −3 kcal/mol means for MDM2/p53

- **D-10:** The notebook cells use `--input-poses tests/fixtures/mdm2_p53/` to bypass Stage 1.
  A markdown note explains that production runs use `--n-samples 100` with Stage 1 on a GPU.

### README.md (DOCS-01)

- **D-11:** `README.md` is a **comprehensive user guide** — not a quick-start card. Sections:
  1. Project overview (2-3 sentences: hybrid ML + physics rescoring, iGEM context)
  2. Architecture overview (one-paragraph summary + link to `docs/architecture.md`)
  3. Prerequisites (CUDA, conda, ADFRsuite — with link to `INSTALL.md` for full setup)
  4. Quick install (the two `conda env create` + `pip install -e .` commands)
  5. CLI reference — all four subcommands: `dock`, `calibrate`, `benchmark`, `prep`
     - Each with the canonical example command from CLAUDE.md §5
     - All flags listed with units and defaults
  6. Expected output files — what each run produces and where
  7. Running tests — `pytest` / `pytest -m slow` / `pytest --cov`
  8. Troubleshooting — top 5 failure modes (CUDA version, ADFRsuite not on PATH, PULCHRA
     version, pdbfixer not in base env, score-env vs rapidock-env confusion)
  9. License (MIT) and citation block

- **D-12:** `INSTALL.md` (152 lines, already exists) is reviewed and updated for completeness —
  not rewritten. Add: ADFRsuite exact download link, PULCHRA v3.04 build instructions, conda env
  activation order, smoke test invocation.

### Architecture Documentation (DOCS-02)

- **D-13:** `docs/architecture.md` uses **ASCII art + prose** in the same style as CLAUDE.md §3.
  Content:
  1. Top-level pipeline diagram (expand the CLAUDE.md diagram with file paths and env labels)
  2. Module breakdown table: module → responsibility → key functions → data in/out
  3. Subprocess orchestration section: how `driver.py` calls `conda run -n rapidock-env`,
     what crosses the boundary (file paths, return codes), why absolute paths are mandatory
  4. Data model section: `DockConfig`, `PoseRecord`, `ScoredPose`, `ClusterResult` — one
     paragraph each with field summary
  5. Config/calibration flow: how `calibration.json` is loaded and `alpha`/`beta` applied

### License Audit (DOCS-03)

- **D-14:** `docs/licenses.txt` is committed. It contains pip-licenses output for both environments:
  - score-env: `pip-licenses --format=plain-vertical --with-urls`
  - rapidock-env: same command
  Both sections are separated by a header line in the file. Any GPL/LGPL/AGPL dependency
  must be flagged as a blocker (iGEM OSI-license requirement from CLAUDE.md §2.6).
  The file is generated on the RTX machine (where both envs exist) and committed manually.

### Claude's Discretion

- README.md flag table format: Markdown code blocks for CLI examples, inline table for flag
  reference (flag | type | default | description).
- Tutorial notebook kernel: `python3` (score-env). Metadata should specify score-env as the
  recommended kernel.
- If any benchmark complex fails PDB download (network issue), `benchmark.py` logs a warning
  and skips it — does not abort the whole run. Final report notes skipped complexes.
- `docs/tutorial.ipynb` plot cells embed plots as base64 PNG in cell output (default Jupyter
  behavior when pre-run) — no external image files needed.

</decisions>

<canonical_refs>
## Canonical References

Downstream agents MUST read these before planning or implementing.

- `docs/HybriDock-Pep_Technical_Specification.pdf` — §14 (benchmark complex list and accuracy
  targets), §15 (iGEM wiki documentation rubric). Read before writing benchmark.py or docs.
- `CLAUDE.md` — §2.6 (license constraints: no copyleft), §3 (architecture diagram to expand),
  §4 (dev conventions), §5 (canonical CLI examples for README), §8 (success criteria: Pearson r
  ≥ 0.55, best-of-top-25 Cα RMSD ≤ 2.0 Å).
- `INSTALL.md` — existing 152-line install guide. Extend, don't rewrite.
- `data/training_complexes.csv` — 3 complexes to EXCLUDE from test_complexes.csv.
- `.planning/phases/07-output-integration/07-CONTEXT.md` — output schema for ranked_poses.csv
  and driver return type (scored_poses, cluster_result) used in tutorial notebook narrative.
- `tests/fixtures/mdm2_p53/` — fixture poses used in tutorial notebook via --input-poses.
- `calibration.json` — committed calibration file referenced in tutorial and benchmark.

</canonical_refs>

<deferred_ideas>
## Deferred Ideas (out of Phase 8 scope)

- MM-GBSA refinement section in tutorial (OPT-01 is v2 scope)
- Interactive benchmark dashboard / HTML report (nice to have, not v1)
- Automated benchmark CI job (needs score-env on CI runner — deferred to v2)

</deferred_ideas>
