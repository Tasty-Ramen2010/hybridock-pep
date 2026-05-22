from __future__ import annotations

"""Ensemble statistics for clustered peptide poses.

Computes per-cluster mean, standard deviation (ddof=1), 95% confidence
interval (t-distribution) and best_pose_idx, then writes
cluster_summary.csv with a fixed column order.

Implements ANAL-02 from the Phase 6 analysis plan.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from hybridock_pep.models import ScoredPose

try:
    from scipy.stats import t as t_dist
except ImportError:
    t_dist = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _ci95(values: list[float]) -> tuple[float, float]:
    """Compute a 95% confidence interval using the t-distribution.

    For n == 1 the interval degenerates to (value, value).
    For n >= 2 uses ``scipy.stats.t.interval`` with df=n-1 and
    scale equal to the standard error of the mean (std/sqrt(n)).

    Args:
        values: Non-empty list of numeric score values.

    Returns:
        Tuple of (ci95_lower, ci95_upper) bounds.

    Raises:
        ValueError: If values is empty.
    """
    n = len(values)
    if n == 0:
        raise ValueError("values must not be empty")
    if n == 1:
        return (float(values[0]), float(values[0]))

    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))  # standard error of mean, NOT std

    if t_dist is None:
        logger.warning(
            "scipy not available; falling back to mean ± 1.96*SEM for 95% CI"
        )
        return (mean - 1.96 * sem, mean + 1.96 * sem)

    # sem=0 when all values are identical (std=0); t.interval multiplies
    # ±inf * 0 which produces NaN in scipy. Return degenerate interval instead.
    if sem == 0.0:
        return (mean, mean)

    lo, hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return (float(lo), float(hi))


def compute_cluster_stats(
    scored_poses: list[ScoredPose],
) -> list[dict[str, Any]]:
    """Compute per-cluster ensemble statistics from clustered scored poses.

    Expects that ``ScoredPose.cluster_id`` is already populated by
    ``cluster_poses()`` before this function is called.

    Args:
        scored_poses: List of ScoredPose objects with ``cluster_id`` and
            ``hybrid_score`` populated (both non-None at this stage).

    Returns:
        List of dicts, one per cluster, sorted by cluster_id ascending.
        Each dict has exactly these 7 keys:

        - ``cluster_id`` (int): cluster index (0-based)
        - ``n_poses`` (int): number of poses in the cluster
        - ``mean_hybrid_score`` (float): arithmetic mean of hybrid scores
        - ``std_hybrid_score`` (float): sample std (ddof=1); 0.0 for n=1
        - ``ci95_lower`` (float): lower 95% CI bound
        - ``ci95_upper`` (float): upper 95% CI bound
        - ``best_pose_idx`` (int): pose_idx of the pose with the lowest
          hybrid_score in the cluster
    """
    clusters: dict[int, list[ScoredPose]] = {}
    for pose in scored_poses:
        cid = pose.cluster_id
        if cid is None:
            logger.warning("Pose %d has no cluster_id; skipping", pose.pose_idx)
            continue
        clusters.setdefault(cid, []).append(pose)

    stats: list[dict[str, Any]] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        scores = [p.hybrid_score for p in members if p.hybrid_score is not None]

        n = len(scores)
        if scores:
            arr = np.asarray(scores, dtype=np.float64)
            mean_hybrid = float(arr.mean())
            std_hybrid = float(arr.std(ddof=1)) if n > 1 else 0.0
            ci_lo, ci_hi = _ci95(scores)
            best_pose = min(members, key=lambda p: p.hybrid_score if p.hybrid_score is not None else float("inf"))
        else:
            mean_hybrid = float("nan")
            std_hybrid = float("nan")
            ci_lo, ci_hi = float("nan"), float("nan")
            best_pose = members[0]

        logger.debug(
            "Cluster %d: n=%d, mean=%.3f, sil N/A", cid, len(members), mean_hybrid
        )

        stats.append(
            {
                "cluster_id": cid,
                "n_poses": len(members),
                "mean_hybrid_score": mean_hybrid,
                "std_hybrid_score": std_hybrid,
                "ci95_lower": ci_lo,
                "ci95_upper": ci_hi,
                "best_pose_idx": best_pose.pose_idx,
            }
        )

    logger.debug("compute_cluster_stats: %d clusters processed", len(stats))
    return stats


def write_cluster_summary_csv(
    stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write per-cluster statistics to a CSV file.

    Creates parent directories if they do not exist. Columns are written
    in a fixed order regardless of dict key order.

    Args:
        stats: List of cluster stat dicts from ``compute_cluster_stats()``.
        output_path: Destination path for the CSV file.

    Raises:
        ValueError: If stats is empty (nothing to write).
    """
    if not stats:
        raise ValueError("stats must not be empty; nothing to write")

    FIELDNAMES = [
        "cluster_id",
        "n_poses",
        "mean_hybrid_score",
        "std_hybrid_score",
        "ci95_lower",
        "ci95_upper",
        "best_pose_idx",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stats)

    logger.info("Wrote cluster_summary.csv to %s", output_path)
