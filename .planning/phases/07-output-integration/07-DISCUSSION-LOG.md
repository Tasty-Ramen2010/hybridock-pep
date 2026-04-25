# Phase 7: Output & Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-25
**Phase:** 07-output-integration
**Areas discussed:** ranked_poses.csv rows, ΔG reporting, MDM2/p53 test approach, Stage 4 driver wiring

---

## ranked_poses.csv rows

| Option | Description | Selected |
|--------|-------------|----------|
| Top-10 by hybrid_score | Sort all scored poses by hybrid_score ascending, take top 10. Cluster IDs still appear as a column. | ✓ |
| 1 best per cluster (top 10 clusters) | 1 representative per cluster sorted by mean_hybrid_score. Shows cluster-level diversity. | |
| All poses, sorted by score | Write every scored pose. Downstream users filter themselves. | |

**User's choice:** Top-10 by hybrid_score
**Notes:** None

---

## CSV anomaly flags

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — include both flags | is_ad4_anomaly and is_clipped as columns. Already on ScoredPose — free to include. | ✓ |
| No — scores only | Keep CSV minimal. Anomalies are logged; users who care check the log. | |

**User's choice:** Yes — include both flags
**Notes:** None

---

## ΔG source

| Option | Description | Selected |
|--------|-------------|----------|
| hybrid_score of best pose = ΔG | hybrid_score IS the ΔG estimate. No extra column — just label it clearly. | |
| Separate delta_g column (kcal/mol) | Dedicated delta_g column; value identical to hybrid_score but explicitly labeled for scientific readers. | ✓ |

**User's choice:** Separate delta_g column
**Notes:** Same value as hybrid_score, distinct column name for scientific notation clarity.

---

## ΔG stdout format

| Option | Description | Selected |
|--------|-------------|----------|
| Single summary line | e.g. 'Best pose: ΔG = -5.3 kcal/mol (cluster 0, pose_042.pdb)'. Concise, machine-parseable. | ✓ |
| Multi-line summary | Cluster count, best pose, ΔG, and run time on separate lines. More human-readable. | |

**User's choice:** Single summary line
**Notes:** None

---

## MDM2/p53 test approach

| Option | Description | Selected |
|--------|-------------|----------|
| Fixture poses + real scoring | ~25 pre-generated fixture PDBs; test runs Stage 2–4 only. No network, no GPU. | ✓ |
| Full pipeline run | Spawn RAPiDock (Stage 1), all 4 stages. Requires GPU + rapidock-env. | |
| Mocked scores, output only | Pre-inject ScoredPose objects with known scores. Tests only output writing, not scoring path. | |

**User's choice:** Fixture poses + real scoring
**Notes:** None

---

## MDM2/p53 fixture count

| Option | Description | Selected |
|--------|-------------|----------|
| ~10 poses | Small, ~500KB total. Enough for clustering. Fast. | |
| ~25 poses | More realistic cluster distribution. ~1.25MB total. Still checkable. | ✓ |

**User's choice:** ~25 poses
**Notes:** None

---

## Stage 4 driver wiring

| Option | Description | Selected |
|--------|-------------|----------|
| Full Stage 4 in driver.py | Phase 7 adds Stage 4: write_ranked_csv(), write_best_pose_pdb(), print ΔG line. Pipeline complete. | ✓ |
| csv_writer.py only, no driver change | Write csv_writer.py; driver.py still returns list[ScoredPose]. Wiring deferred to Phase 8. | |

**User's choice:** Full Stage 4 in driver.py
**Notes:** None

---

## best_pose.pdb identification

| Option | Description | Selected |
|--------|-------------|----------|
| From ClusterResult (in-memory) | Sort clusters by mean_hybrid_score, take top cluster's best_pose_idx. No extra file I/O. | ✓ |
| Re-read cluster_summary.csv | Parse CSV to find best cluster. Works without ClusterResult in scope. | |

**User's choice:** From ClusterResult (in-memory)
**Notes:** None

---

## Claude's Discretion

- CSV stdlib (no pandas): `csv.DictWriter`
- Float precision: 4 decimal places for score columns
- Atomic CSV writes via `.tmp` intermediate (same pattern as metadata.py)
- Fixture PDB generation approach (real truncated 2OY2 poses vs. synthetic)

## Deferred Ideas

- MM-GBSA post-processing (`--refine-topk N`) — future phase
- run_metadata.json enrichment with output file paths — out of scope for Phase 7
- Full GPU integration test (Stage 1–4) — deferred; fixture-based TEST-02 sufficient for v1
