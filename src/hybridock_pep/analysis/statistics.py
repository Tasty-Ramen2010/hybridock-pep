from __future__ import annotations

"""Ensemble statistics for clustered peptide poses.

Stub module — full implementation in Phase 6 Plan 06-03.
Provides the function signatures consumed by clustering.py so that
cluster_poses() can import and call them without error.

compute_cluster_stats() and write_cluster_summary_csv() are the two
public functions called by cluster_poses().  _ci95() is the private
helper exposed in tests.
"""

import csv
import logging
import math
from pathlib import Path
from typing import Any

from hybridock_pep.models import ScoredPose

_log = logging.getLogger(__name__)


def _ci95(scores: list[float]) -> tuple[float, float]:
    """Compute a 95% confidence interval for a list of scores.

    Uses the t-distribution for n >= 2 and degenerates to (value, value)
    for n == 1.

    Args:
        scores: List of numeric score values.

    Returns:
        Tuple of (lower, upper) 95% CI bounds.
    """
    n = len(scores)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (scores[0], scores[0])

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n)

    # t critical value for 95% CI, two-tailed.
    # Use scipy if available, fall back to a conservative approximation.
    try:
        from scipy.stats import t as t_dist  # type: ignore[import-untyped]

        t_crit = float(t_dist.ppf(0.975, df=n - 1))
    except ImportError:
        # Conservative fallback: use z = 1.96 (valid for large n; slightly
        # anti-conservative for small n, but avoids a hard scipy dependency
        # in environments where scipy is absent).
        t_crit = 1.96

    margin = t_crit * stderr
    return (mean - margin, mean + margin)


def compute_cluster_stats(
    scored_poses: list[ScoredPose],
) -> list[dict[str, Any]]:
    """Compute per-cluster ensemble statistics from clustered scored poses.

    Expects that ScoredPose.cluster_id is already populated (set by
    cluster_poses() before this function is called).

    Args:
        scored_poses: List of ScoredPose objects with cluster_id and
            hybrid_score populated.

    Returns:
        List of dicts, one per cluster, sorted by cluster_id, each
        containing keys: cluster_id, n_poses, mean_hybrid_score,
        std_hybrid_score, ci95_lower, ci95_upper, best_pose_idx.
    """
    # Group by cluster_id
    clusters: dict[int, list[ScoredPose]] = {}
    for pose in scored_poses:
        cid = pose.cluster_id
        if cid is None:
            _log.warning("Pose %d has no cluster_id; skipping", pose.pose_idx)
            continue
        clusters.setdefault(cid, []).append(pose)

    stats: list[dict[str, Any]] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        scores = [
            p.hybrid_score for p in members if p.hybrid_score is not None
        ]

        if scores:
            mean_score = sum(scores) / len(scores)
            variance = (
                sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
                if len(scores) > 1
                else 0.0
            )
            std_score = math.sqrt(variance)
            ci_lo, ci_hi = _ci95(scores)
            # Best pose = lowest (most negative) hybrid_score
            best_pose = min(members, key=lambda p: p.hybrid_score or float("inf"))
        else:
            mean_score = float("nan")
            std_score = float("nan")
            ci_lo, ci_hi = float("nan"), float("nan")
            best_pose = members[0]

        stats.append(
            {
                "cluster_id": cid,
                "n_poses": len(members),
                "mean_hybrid_score": mean_score,
                "std_hybrid_score": std_score,
                "ci95_lower": ci_lo,
                "ci95_upper": ci_hi,
                "best_pose_idx": best_pose.pose_idx,
            }
        )

    _log.debug("compute_cluster_stats: %d clusters processed", len(stats))
    return stats


def write_cluster_summary_csv(
    stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write per-cluster statistics to a CSV file.

    Creates parent directories if they do not exist.

    Args:
        stats: List of cluster stat dicts from compute_cluster_stats().
        output_path: Destination path for the CSV file.

    Raises:
        ValueError: If stats is empty (nothing to write).
    """
    if not stats:
        raise ValueError("stats must not be empty; nothing to write")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cluster_id",
        "n_poses",
        "mean_hybrid_score",
        "std_hybrid_score",
        "ci95_lower",
        "ci95_upper",
        "best_pose_idx",
    ]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stats)

    _log.info("Cluster summary CSV written to %s (%d rows)", output_path, len(stats))
