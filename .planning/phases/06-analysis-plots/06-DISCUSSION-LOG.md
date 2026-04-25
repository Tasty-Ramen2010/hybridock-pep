# Phase 6: Analysis & Plots - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 06-analysis-plots
**Areas discussed:** Contact-zone definition, Convergence series, Cluster count k, cluster_poses() API

---

## Contact-zone definition

| Option | Description | Selected |
|--------|-------------|----------|
| Distance from receptor | Residues with Cα within XÅ of any receptor Cα | ✓ |
| Terminal exclusion | Drop first/last N residues | |
| All residues | Full-peptide RMSD (already ruled out by STATE.md) | |

**User's choice:** Distance from receptor — 6 Å threshold

| Option | Description | Selected |
|--------|-------------|----------|
| 6 Å | Standard contact threshold in SBDD | ✓ |
| 8 Å | Wider, second-shell residues included | |
| You decide | Defer to Claude | |

**User's choice:** 6 Å

| Option | Description | Selected |
|--------|-------------|----------|
| Fall back to full-peptide RMSD | Keep pose in analysis with all residues | ✓ |
| Treat as PoseFailure(stage='clustering') | Exclude from RMSD matrix | |
| Fixed minimum window | Use middle 5 residues | |

**User's choice:** Fall back to full-peptide RMSD when fewer than 3 contact residues

---

## Convergence series definition

| Option | Description | Selected |
|--------|-------------|----------|
| Pose arrival order | Tests sampling convergence (pose_idx 0..N) | |
| Score-sorted order | Tests ranking stability (sorted by hybrid_score) | ✓ |
| You decide | Defer to Claude | |

**User's choice:** Score-sorted order

| Option | Description | Selected |
|--------|-------------|----------|
| Ascending — best scores first | Most negative hybrid_score first | ✓ |
| Descending — worst first | Starts from worst scorer | |

**User's choice:** Ascending — best scores first

**Notes:** The convergence plot tests ranking stability, not sampling convergence. Running mean ± σ of top-N scorers as N expands from 1 to len(scored_poses).

---

## Cluster count (k) selection

| Option | Description | Selected |
|--------|-------------|----------|
| Silhouette-optimal k | argmax silhouette over k range | ✓ |
| Fixed k=5 | Simple, silhouette is diagnostic only | |
| New --n-clusters CLI flag | User-supplied k | |

**User's choice:** Silhouette-optimal k

| Option | Description | Selected |
|--------|-------------|----------|
| k=2..10 | Fixed wide range | |
| k=2..min(15, n_poses÷5) | Adaptive upper bound | ✓ |
| k=2..5 | Conservative narrow range | |

**User's choice:** k=2..min(15, n_poses÷5) — adaptive upper bound

---

## cluster_poses() API & ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Mutate in-place + return ClusterResult | Matches apply_hybrid_score() pattern | ✓ |
| Return annotated list only | Loses cluster metadata | |
| Return (list[ScoredPose], dict) tuple | Less typed | |

**User's choice:** Mutate in-place + return ClusterResult

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 6 owns all 3 outputs | Writes CSV + 2 plots directly | ✓ |
| Phase 6 returns data only | Driver/Phase 7 writes | |
| Split: Phase 6 plots, Phase 7 CSVs | Contradicts ROADMAP SC-2 | |

**User's choice:** Phase 6 owns cluster_summary.csv, convergence_plot.png, silhouette_plot.png

| Option | Description | Selected |
|--------|-------------|----------|
| analysis/clustering.py | Close to producing code, models.py stays focused | ✓ |
| models.py | All dataclasses in one place | |

**User's choice:** ClusterResult defined in analysis/clustering.py

---

## Claude's Discretion

- Matplotlib backend: Agg (headless)
- Figure size/DPI: 8×5 inches, 150 DPI
- Agglomerative linkage: average (per ROADMAP SC-1)
- RMSD matrix: sklearn pairwise_distances with precomputed metric (per ROADMAP SC-1)
- cluster_summary.csv column order: cluster_id, n_poses, mean_hybrid_score, std_hybrid_score, ci95_lower, ci95_upper, best_pose_idx

## Deferred Ideas

- VIZ-01: Dendrogram plot — v2
- Arrival-order convergence — v2
- --n-clusters CLI flag — v2
