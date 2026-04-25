# Phase 7: Output & Integration - Context

**Gathered:** 2026-04-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 delivers the final output layer: `ranked_poses.csv`, `best_pose.pdb`, a ΔG summary
line to stdout, and the MDM2/p53 integration test (TEST-02). It adds Stage 4 to `driver.py`
that calls the new `output/csv_writer.py` functions and completes the end-to-end pipeline.
Phase 7 does NOT regenerate `cluster_summary.csv` — that is Phase 6's output, read as input here.

</domain>

<decisions>
## Implementation Decisions

### ranked_poses.csv Composition

- **D-01:** `ranked_poses.csv` contains the **top-10 individual poses** sorted by `hybrid_score`
  ascending (most negative = best binding first). This is 10 individual pose rows, not 1 per cluster.
- **D-02:** Columns (in order): `rank`, `hybrid_score`, `vina_score`, `ad4_score`,
  `entropy_correction`, `delta_g`, `cluster_id`, `pose_filename`, `is_ad4_anomaly`, `is_clipped`.
- **D-03:** `is_ad4_anomaly` and `is_clipped` flags are included as boolean columns (True/False).
  They are already on `ScoredPose` — free to include and useful for users auditing suspect scores.

### ΔG Reporting

- **D-04:** `delta_g` is a **dedicated column** in `ranked_poses.csv`, distinct in name from
  `hybrid_score` but equal in value (`delta_g = hybrid_score`). Explicit labeling serves scientific
  readers who expect ΔG notation in output tables.
- **D-05:** At run completion, `driver.py` prints a single summary line to stdout (via `logger.info`
  at INFO level, so it appears in default output):
  ```
  Best pose: ΔG = -5.3 kcal/mol (cluster 0, pose_042.pdb)
  ```
  The values are: best cluster's `mean_hybrid_score` rounded to 1 decimal, cluster ID (0-indexed,
  sorted by mean score), and the best pose filename.

### best_pose.pdb Selection

- **D-06:** "Top-ranked cluster" = cluster with the **lowest `mean_hybrid_score`** (most negative).
  Clusters are sorted by mean score at write time — cluster IDs assigned by k-means are not
  necessarily ordered by quality.
- **D-07:** `write_best_pose_pdb()` uses the **in-memory `ClusterResult`** (not re-reading
  `cluster_summary.csv`). It takes the top cluster's `best_pose_idx`, reads
  `config.output_dir / "poses" / f"pose_{best_pose_idx:03d}.pdb"`, and copies it to
  `config.output_dir / "best_pose.pdb"`.

### Stage 4 Driver Wiring

- **D-08:** Phase 7 adds a **full Stage 4** to `driver.py` that:
  1. Calls `write_ranked_csv(scored_poses, config)` → writes `ranked_poses.csv`
  2. Calls `write_best_pose_pdb(cluster_result, scored_poses, config)` → writes `best_pose.pdb`
  3. Prints the ΔG summary line (D-05)
  4. Returns `(scored_poses, cluster_result)` — richer than the current `list[ScoredPose]` return
- **D-09:** `csv_writer.py` lives in `src/hybridock_pep/output/` alongside `metadata.py`.
  It exports `write_ranked_csv` and `write_best_pose_pdb`. Wired via `output/__init__.py`.

### MDM2/p53 Integration Test (TEST-02)

- **D-10:** `tests/test_e2e.py` contains the MDM2/p53 integration test tagged `pytest -m slow`.
  The test is **Stage 2–4 only**: fixture poses → prep → score → cluster → output. Stage 1
  (RAPiDock GPU) is skipped entirely.
- **D-11:** **25 pre-generated MDM2/p53 fixture poses** are checked into
  `tests/fixtures/mdm2_p53/` (PDB files, ~50KB each ≈ 1.25MB total). These are real poses from
  PDB 2OY2 (MDM2) / peptide `ETFSDLWKLLPE`. They must produce `corrected ΔG < −3 kcal/mol`
  on the best pose to pass TEST-02.
- **D-12:** The integration test asserts:
  1. `ranked_poses.csv` exists and has ≤10 rows with all required columns
  2. `best_pose.pdb` exists and is non-empty
  3. Best pose `hybrid_score < −3.0` kcal/mol (TEST-02 threshold)
  4. `run_metadata.json` exists with `status: "complete"`

### Claude's Discretion

- Column order in CSV: rank column is implicit (row 1 = rank 1), but include explicit `rank`
  column for readability.
- Float precision in CSV: 4 decimal places for all score columns.
- `write_ranked_csv` should use Python's `csv` stdlib (no pandas dependency) — consistent with
  score-env keeping dependencies minimal.
- Fixture PDB generation: synthesize minimal valid PDB files for the 25 poses, or use real
  truncated structures from PDB 2OY2. Claude decides based on what produces scores in the
  expected range for the threshold test.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Technical Specification
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5, §11, §12 — Pipeline architecture,
  scoring definitions, output requirements, and integration test baseline.

### Phase Contracts
- `.planning/ROADMAP.md` §Phase 7 — Success criteria OUT-01, OUT-02, OUT-03, TEST-02
- `.planning/REQUIREMENTS.md` OUT-01, OUT-02, OUT-03, TEST-02 — Requirement definitions
- `.planning/phases/06-analysis-plots/06-CONTEXT.md` D-09, D-10 — ClusterResult structure
  (`best_pose_idx` per cluster); cluster_summary.csv owned by Phase 6

### Existing Code
- `src/hybridock_pep/models.py` — `ScoredPose` (all score fields + flags), `DockConfig`
  (`output_dir`, `peptide_sequence`)
- `src/hybridock_pep/analysis/clustering.py` — `ClusterResult` dataclass with
  `per_cluster_stats: list[dict]` (each dict: `cluster_id`, `n_poses`, `mean_hybrid`,
  `std_hybrid`, `ci95_lower`, `ci95_upper`, `best_pose_idx`)
- `src/hybridock_pep/driver.py` — Stage 3 ends at `finalize_metadata()`; Stage 4 stub belongs here
- `src/hybridock_pep/output/metadata.py` — pattern for atomic JSON writes; follow same
  style for CSV writes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ScoredPose` fields `vina_score`, `ad4_score`, `entropy_correction`, `hybrid_score`,
  `cluster_id`, `pdb_path`, `is_ad4_anomaly`, `is_clipped` — all needed columns are directly
  on `ScoredPose`; no recomputation needed.
- `_write_json_atomic()` in `metadata.py` — pattern to follow: write to `.tmp`, then `os.replace`.
  Apply equivalent pattern to CSV (write to `.tmp`, rename).
- `PoseFailure(stage="clustering")` defined in `models.py` — if Stage 4 fails, use same failure
  pattern as prior stages.

### Established Patterns
- Stdlib-only I/O (no pandas): `csv.DictWriter` for CSV, `shutil.copy2` for PDB copy.
- `logging` not `print`; ΔG summary line at INFO level so it appears in default output.
- `output_dir.mkdir(parents=True, exist_ok=True)` before writing any file.
- Atomic writes via `.tmp` intermediate to avoid partial files on crash.

### Integration Points
- `driver.py` line ~160 (after `finalize_metadata()`): insert Stage 4 call block
- `output/__init__.py`: add `write_ranked_csv`, `write_best_pose_pdb` to exports
- `tests/test_e2e.py`: new file; imports `run_dock` or calls through driver
- `tests/fixtures/mdm2_p53/`: new directory for 25 fixture PDB poses

</code_context>

<specifics>
## Specific Ideas

- The `delta_g` column value is identical to `hybrid_score` — the separation is purely for
  scientific labeling clarity. Do not compute a different value.
- The stdout summary line format `Best pose: ΔG = -5.3 kcal/mol (cluster 0, pose_042.pdb)`
  must appear via `logger.info` (not print) so it respects verbosity flags and goes to the
  log file.
- The integration test (TEST-02) requires `corrected ΔG < −3 kcal/mol` on MDM2/p53 fixture
  poses. The fixture poses must be chosen/synthesized so that real scoring produces values in
  this range — synthetic fixtures with random coordinates will NOT satisfy the threshold.
  Use truncated real poses from PDB 2OY2 if possible.

</specifics>

<deferred>
## Deferred Ideas

- MM-GBSA top-K post-processing (`--refine-topk N` flag) — out of Phase 7 scope; already
  tracked as Phase 8 or future work per CLAUDE.md.
- `run_metadata.json` enrichment with output file paths — could be added but Phase 7 scope
  is limited to the 4 success criteria.
- Full pipeline integration test (Stage 1–4 with GPU) — deferred; TEST-02 uses fixture poses
  for portability. A GPU integration test can be added separately if needed.

</deferred>

---

*Phase: 07-output-integration*
*Context gathered: 2026-04-25*
