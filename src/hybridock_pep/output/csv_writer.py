"""Ranked CSV and best-pose PDB writers for HybriDock-Pep output (OUT-01, OUT-02, OUT-03)."""
from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from hybridock_pep.models import DockConfig, ScoredPose
from hybridock_pep.analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)

FIELDNAMES: list[str] = [
    "rank",
    "hybrid_score",
    "vina_score",
    "ad4_score",
    "entropy_correction",
    "delta_g",
    "mmgbsa_dg",
    "cluster_id",
    "pose_filename",
    "n_contact_residues",
    "is_ad4_anomaly",
    "is_clipped",
    "is_clashed",
]


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Atomically write rows as CSV to path using a .tmp intermediate file.

    Args:
        path: Destination path for the CSV file.
        rows: List of row dicts. Keys must be a superset of fieldnames.
        fieldnames: Column order for the CSV header and rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_ranked_csv(scored_poses: list[ScoredPose], config: DockConfig) -> Path:
    """Write top-10 poses ranked by hybrid_score to ranked_poses.csv.

    Sorts all scored_poses by hybrid_score ascending (most negative = best first),
    takes the top 10, formats floats to 4 decimal places, and writes atomically.
    delta_g is identical to hybrid_score per D-04 (same number, scientific label).

    Args:
        scored_poses: All scored poses from the pipeline. Sorted internally.
        config: Run configuration. output_dir is the write destination.

    Returns:
        Absolute path to the written ranked_poses.csv.
    """
    sorted_poses = sorted(
        scored_poses,
        key=lambda p: (p.hybrid_score if p.hybrid_score is not None else float("inf")),
    )
    top10 = sorted_poses[:10]

    rows: list[dict[str, Any]] = []
    for rank, pose in enumerate(top10, start=1):
        hs = pose.hybrid_score if pose.hybrid_score is not None else float("nan")
        rows.append(
            {
                "rank": rank,
                "hybrid_score": f"{hs:.4f}",
                "vina_score": f"{pose.vina_score:.4f}" if pose.vina_score is not None else "",
                "ad4_score": f"{pose.ad4_score:.4f}" if pose.ad4_score is not None else "",
                "entropy_correction": (
                    f"{pose.entropy_correction:.4f}"
                    if pose.entropy_correction is not None
                    else ""
                ),
                "delta_g": f"{hs:.4f}",  # D-04: same value as hybrid_score
                "mmgbsa_dg": (
                    f"{pose.mmgbsa_dg:.4f}" if pose.mmgbsa_dg is not None else ""
                ),
                "cluster_id": pose.cluster_id if pose.cluster_id is not None else "",
                "pose_filename": pose.pdb_path.name,
                "n_contact_residues": (
                    pose.n_contact_residues if pose.n_contact_residues is not None else ""
                ),
                "is_ad4_anomaly": str(pose.is_ad4_anomaly),
                "is_clipped": str(pose.is_clipped),
                "is_clashed": str(pose.is_clashed),
            }
        )

    output_path = config.output_dir / "ranked_poses.csv"
    _write_csv_atomic(output_path, rows, FIELDNAMES)
    logger.info("Wrote ranked_poses.csv (%d poses) to %s", len(rows), output_path)
    return output_path


def write_best_pose_pdb(
    cluster_result: ClusterResult,
    config: DockConfig,
    scored_poses: list[ScoredPose],
) -> Path:
    """Copy the best cluster centroid PDB to best_pose.pdb.

    Selects the cluster with the lowest mean_hybrid_score (most negative = best),
    looks up the source pdb_path directly from scored_poses (works for both
    RAPiDock-generated poses in output_dir/poses/ and --input-poses bypass paths).

    Args:
        cluster_result: Completed clustering result with per_cluster_stats populated.
        config: Run configuration. output_dir is the write destination.
        scored_poses: All scored poses; used to resolve pdb_path by pose_idx.

    Returns:
        Absolute path to the written best_pose.pdb.

    Raises:
        ValueError: If per_cluster_stats is empty or best pose_idx not in scored_poses.
        FileNotFoundError: If the source PDB does not exist.
    """
    if not cluster_result.per_cluster_stats:
        raise ValueError("cluster_result.per_cluster_stats is empty — cannot select best pose")

    # Prefer MM-GBSA winner if any poses were refined; fall back to best cluster centroid.
    mmgbsa_poses = [p for p in scored_poses if p.mmgbsa_dg is not None]
    if mmgbsa_poses:
        best_mmgbsa = min(mmgbsa_poses, key=lambda p: p.mmgbsa_dg)  # type: ignore[arg-type]
        best_pose_idx = best_mmgbsa.pose_idx
        logger.info(
            "Best pose selected by MM-GBSA: pose %d ΔG = %.2f kcal/mol",
            best_pose_idx, best_mmgbsa.mmgbsa_dg,
        )
    else:
        best_cluster = min(
            cluster_result.per_cluster_stats,
            key=lambda s: s["mean_hybrid_score"],
        )
        best_pose_idx = best_cluster["best_pose_idx"]

    pose_by_idx = {p.pose_idx: p for p in scored_poses}
    if best_pose_idx not in pose_by_idx:
        raise ValueError(
            f"best_pose_idx={best_pose_idx} not found in scored_poses "
            f"(available: {sorted(pose_by_idx)})"
        )
    src = pose_by_idx[best_pose_idx].pdb_path
    dest = config.output_dir / "best_pose.pdb"

    if not src.exists():
        raise FileNotFoundError(f"Source pose PDB not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    selected_pose = pose_by_idx[best_pose_idx]
    score_val = selected_pose.mmgbsa_dg if selected_pose.mmgbsa_dg is not None else selected_pose.hybrid_score
    logger.info(
        "Best pose: ΔG = %.1f kcal/mol (cluster %s, %s)",
        score_val if score_val is not None else float("nan"),
        selected_pose.cluster_id,
        src.name,
    )
    return dest
