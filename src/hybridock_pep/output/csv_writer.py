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
    "ml_pose_score",
    "bsa_fit_score",
    "bsa",
    "n_clash",
    "hybrid_score",
    "vina_score",
    "ad4_score",
    "entropy_correction",
    "delta_g",
    "mmgbsa_dg",
    "ensemble_dg",
    "pooled_affinity_dg",
    "cluster_id",
    "pose_filename",
    "n_contact_residues",
    "is_ad4_anomaly",
    "is_clipped",
    "is_clashed",
]


def _rank_key(pose: ScoredPose) -> float:
    """Pose ranking key (ascending = best first).

    Priority: ML pose ranker (predicted native RMSD, ≈2× BSA-fit within-complex τ;
    E96) → BSA-fit (buried surface + clash penalty) → hybrid_score. The ML score and
    BSA-fit are STRUCTURAL rankers only; neither alters the affinity number. Falls
    through so ranking never crashes when an upstream ranker is unavailable.
    """
    if pose.ml_pose_score is not None:
        return pose.ml_pose_score
    if pose.bsa_fit_score is not None:
        return pose.bsa_fit_score
    return pose.hybrid_score if pose.hybrid_score is not None else float("inf")


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
    """Write all scored poses ranked by hybrid_score to ranked_poses.csv.

    Sorts all scored_poses by hybrid_score ascending (most negative = best first),
    formats floats to 4 decimal places, and writes atomically.
    delta_g is identical to hybrid_score per D-04 (same number, scientific label).

    All poses are written (no truncation) so users can inspect the full
    score distribution and clustering results.

    Args:
        scored_poses: All scored poses from the pipeline. Sorted internally.
        config: Run configuration. output_dir is the write destination.

    Returns:
        Absolute path to the written ranked_poses.csv.
    """
    sorted_poses = sorted(scored_poses, key=_rank_key)

    rows: list[dict[str, Any]] = []
    for rank, pose in enumerate(sorted_poses, start=1):
        hs = pose.hybrid_score if pose.hybrid_score is not None else float("nan")
        rows.append(
            {
                "rank": rank,
                "ml_pose_score": (
                    f"{pose.ml_pose_score:.3f}" if pose.ml_pose_score is not None else ""
                ),
                "bsa_fit_score": (
                    f"{pose.bsa_fit_score:.4f}" if pose.bsa_fit_score is not None else ""
                ),
                "bsa": f"{pose.bsa:.1f}" if pose.bsa is not None else "",
                "n_clash": f"{pose.n_clash:.0f}" if pose.n_clash is not None else "",
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
                "ensemble_dg": (
                    f"{pose.ensemble_dg:.4f}" if pose.ensemble_dg is not None else ""
                ),
                "pooled_affinity_dg": (
                    f"{pose.pooled_affinity_dg:.4f}" if pose.pooled_affinity_dg is not None else ""
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

    # Pose ranker priority: ML ranker (predicted native RMSD, ≈2× BSA-fit τ; E96) →
    # BSA-fit → MM-GBSA winner → best cluster centroid. All STRUCTURAL selectors; none
    # changes the affinity number. Each falls through when unavailable.
    ml_poses = [p for p in scored_poses if p.ml_pose_score is not None]
    bsa_poses = [p for p in scored_poses if p.bsa_fit_score is not None]
    mmgbsa_poses = [p for p in scored_poses if p.mmgbsa_dg is not None]
    if ml_poses:
        best = min(ml_poses, key=lambda p: p.ml_pose_score)  # type: ignore[arg-type]
        best_pose_idx = best.pose_idx
        logger.info(
            "Best pose selected by ML ranker: pose %d (predicted native RMSD = %.2f Å)",
            best_pose_idx, best.ml_pose_score,
        )
    elif bsa_poses:
        best = min(bsa_poses, key=lambda p: p.bsa_fit_score)  # type: ignore[arg-type]
        best_pose_idx = best.pose_idx
        logger.info(
            "Best pose selected by BSA-fit: pose %d (fit=%.3f, BSA=%.0f Å², clash=%.0f)",
            best_pose_idx, best.bsa_fit_score, best.bsa or float("nan"), best.n_clash or 0,
        )
    elif mmgbsa_poses:
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
    selected = pose_by_idx[best_pose_idx]
    dest = config.output_dir / "best_pose.pdb"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the Vina-optimized geometry when one exists. Vina's clash-relief
    # `v.optimize()` is more aggressive than the OpenMM minimization (it can
    # move atoms beyond the 0.5 Å displacement cap), so the PDBQT it writes is
    # the geometry that produced the reported Vina score. Visualizing the
    # original poses_minimized/ PDB shows the *pre*-Vina-optimize geometry,
    # which can look clashed even when Vina settled to a clean negative score.
    pdbqt_path = selected.pdbqt_path
    if pdbqt_path is not None and pdbqt_path.exists():
        # Convert PDBQT → PDB via openbabel (already required for the pipeline).
        try:
            import subprocess as _sp
            _sp.run(
                ["obabel", str(pdbqt_path), "-O", str(dest)],
                check=True, capture_output=True, timeout=30,
            )
            logger.debug(
                "best_pose.pdb written from Vina-optimized PDBQT %s",
                pdbqt_path,
            )
        except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired) as exc:
            logger.warning(
                "PDBQT→PDB conversion failed (%s); falling back to "
                "pre-optimize pose PDB. The output geometry will reflect "
                "Stage 1.5 minimization only, not Vina clash relief.",
                exc,
            )
            shutil.copy2(selected.pdb_path, dest)
    else:
        # Legacy path / no PDBQT available — copy the pose PDB as-is.
        if not selected.pdb_path.exists():
            raise FileNotFoundError(
                f"Source pose PDB not found: {selected.pdb_path}"
            )
        shutil.copy2(selected.pdb_path, dest)
    src = selected.pdb_path
    selected_pose = pose_by_idx[best_pose_idx]
    score_val = selected_pose.mmgbsa_dg if selected_pose.mmgbsa_dg is not None else selected_pose.hybrid_score
    logger.info(
        "Best pose: ΔG = %.1f kcal/mol (cluster %s, %s)",
        score_val if score_val is not None else float("nan"),
        selected_pose.cluster_id,
        src.name,
    )
    return dest
