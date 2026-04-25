# Phase 6: Analysis & Plots - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6 delivers the analysis layer that transforms a scored `list[ScoredPose]` into clustered
binding modes with ensemble statistics and diagnostic plots. Three modules are created in
`analysis/`: `clustering.py`, `statistics.py`, and `plotting.py`. Phase 6 owns writing three
output files: `cluster_summary.csv`, `convergence_plot.png`, and `silhouette_plot.png`.
Phase 7 reads `cluster_summary.csv` to produce `ranked_poses.csv` and `best_pose.pdb`.

</domain>

<decisions>
## Implementation Decisions

### Contact-Zone Residue Definition

- **D-01:** Contact-zone residues are identified by **distance from receptor**: any peptide residue
  whose Cα is within **6 Å** of any receptor Cα is "in contact". Receptor Cα coordinates are
  parsed from `config.receptor_path` (raw PDB, not PDBQT) at clustering time using Biopython.
- **D-02:** **Fallback to full-peptide Cα RMSD** when a pose has fewer than 3 contact-zone
  residues. The pose stays in the RMSD matrix and clustering — it is not discarded as a failure.
  This avoids biasing statistics when some poses have poor docking geometry.

### Convergence Series Definition

- **D-03:** Convergence plot uses **score-sorted order** (ascending by `hybrid_score`, most
  negative first). The running mean ± σ shows ranking stability — how quickly the top-N score
  distribution stabilizes as more poses are included.
- **D-04:** X-axis is N (1 to `len(scored_poses)`). Y-axis is running mean ± σ of
  `hybrid_score` over the top-N poses. This tests whether the best-binder signal stabilizes
  rather than whether the sampling itself converged (arrival-order convergence is out of scope
  for v1).

### Cluster Count (k) Selection

- **D-05:** Final k is **silhouette-optimal**: compute silhouette score for k = 2..k_max,
  select `k = argmax(silhouette_scores)`. Fully automatic and reproducible.
- **D-06:** k range upper bound is **adaptive**: `k_max = min(15, n_poses // 5)`. For
  n_poses=100: k_max=15. For n_poses=50: k_max=10. Lower bound is always 2. If k_max < 2
  (fewer than 10 poses), fall back to k=2 without silhouette search.
- **D-07:** `silhouette_plot.png` shows the full silhouette score curve across the k range,
  making the auto-selected k transparent and auditable.

### `cluster_poses()` API and Output Ownership

- **D-08:** `cluster_poses(scored_poses: list[ScoredPose], config: DockConfig) -> ClusterResult`
  **mutates `ScoredPose.cluster_id` in-place** on every pose (consistent with
  `apply_hybrid_score()` pattern from Phase 3). The mutation is the primary side effect.
- **D-09:** `ClusterResult` is a new `@dataclass` defined in `analysis/clustering.py` (not
  `models.py`). `models.py` defines input contracts; `ClusterResult` is analysis output.
  Minimum fields: `k_optimal: int`, `silhouette_score: float`,
  `per_cluster_stats: list[dict]` (each dict: cluster_id, n_poses, mean_hybrid, std_hybrid,
  ci95_lower, ci95_upper, best_pose_idx).
- **D-10:** **Phase 6 owns all three output files.** `cluster_poses()` writes:
  - `{config.output_dir}/cluster_summary.csv` — per-cluster stats from `ClusterResult`
  - `{config.output_dir}/convergence_plot.png` — running mean ± σ (score-sorted)
  - `{config.output_dir}/silhouette_plot.png` — silhouette scores across k range
  Phase 7 reads `cluster_summary.csv` as input; it does not re-generate it.
- **D-11:** Driver integration: `cluster_poses()` replaces the Stage 3 stub in `driver.py`
  (lines 147-148). The driver calls it after scoring and before `finalize_metadata()`. No
  structural changes to driver.py beyond replacing the stub lines.

### Claude's Discretion

- Matplotlib backend: `Agg` (non-interactive, headless-safe). Set via
  `matplotlib.use("Agg")` at module import.
- Figure size and DPI: 8×5 inches at 150 DPI (fast, readable for iGEM wiki).
- Agglomerative clustering linkage: `average` (as specified in ROADMAP SC-1).
- RMSD matrix computed with `sklearn.metrics.pairwise_distances` with precomputed metric
  (as specified in ROADMAP SC-1).
- `cluster_summary.csv` column order: `cluster_id, n_poses, mean_hybrid_score,
  std_hybrid_score, ci95_lower, ci95_upper, best_pose_idx`.
- 95% CI uses `scipy.stats.t.interval` (t-distribution, appropriate for n < 30 clusters).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Technical Specification
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5, §11, §12, §16 — Pipeline architecture,
  scoring definitions, clustering requirements, and known failure modes.

### Phase Contracts
- `.planning/ROADMAP.md` §Phase 6 — Success criteria ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05
- `.planning/phases/05-cli-driver/05-CONTEXT.md` D-02 — driver.py Stage 3 stub that Phase 6 replaces

### Existing Code
- `src/hybridock_pep/models.py` — `ScoredPose` (has `ca_coords`, `cluster_id`, `hybrid_score`),
  `DockConfig` (has `receptor_path`, `output_dir`, `peptide_sequence`)
- `src/hybridock_pep/driver.py` lines 147-148 — Stage 3 stub to replace with `cluster_poses()` call
- `src/hybridock_pep/scoring/entropy.py` — `apply_hybrid_score()` as pattern for in-place mutation
- `src/hybridock_pep/analysis/__init__.py` — exists, empty; add `cluster_poses` export here

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ScoredPose.ca_coords: np.ndarray` shape `[n_residues, 3]` — already stored at parse time (D-13
  from Phase 4), no re-read from disk needed at clustering time.
- `PoseFailure(stage="clustering")` — already defined in `models.py`; use for clustering failures
  consistent with prep/scoring failure pattern.
- `apply_hybrid_score(pose, ...)` in `scoring/entropy.py` — establishes the in-place mutation
  pattern that `cluster_poses()` follows.
- Biopython already a dependency (used in `pose_io.py`) — can use it to parse receptor Cα coords
  without adding a new dependency.

### Established Patterns
- Batch-failure pattern: functions return `(results, failures)` tuple, never raise on per-item
  errors. `cluster_poses()` can follow this: if receptor Cα parsing fails, fall back gracefully.
- Lazy imports at module top with `try/except ImportError` (see `scoring/vina.py`) — follow for
  `sklearn` and `scipy` so tests can run in base env.
- `logging` not `print` everywhere; subprocess calls log full commands.

### Integration Points
- `driver.py` Stage 3 stub (lines 147-148): replace with `cluster_result = cluster_poses(scored_poses, config)`
- `analysis/__init__.py`: export `cluster_poses` so driver imports cleanly from `hybridock_pep.analysis`
- `tests/test_clustering.py`: already referenced in ROADMAP SC-5 — create this file in Phase 6

</code_context>

<specifics>
## Specific Ideas

- The convergence plot (score-sorted, running mean ± σ) tests ranking stability, not sampling
  convergence. This is the explicit user preference — document clearly in the plot title and
  in-code docstring so future developers don't "fix" it to arrival order.
- `k_max = min(15, n_poses // 5)` is the adaptive upper bound for silhouette search. For the
  standard 100-pose run this yields k_max=15, which is generous without being absurd for a
  15-mer peptide targeting a single binding site.

</specifics>

<deferred>
## Deferred Ideas

- VIZ-01: Cluster dendrogram plot — explicitly deferred to v2 (see STATE.md deferred items).
- Arrival-order convergence (tests whether sampling converged) — deferred to v2 if needed;
  score-sorted convergence (ranking stability) is what v1 plots.
- `--n-clusters` CLI flag — not added in v1; silhouette auto-selection makes it unnecessary.
  Can be added in v2 if users need manual override.

</deferred>

---

*Phase: 06-analysis-plots*
*Context gathered: 2026-04-24*
