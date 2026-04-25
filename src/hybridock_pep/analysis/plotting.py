from __future__ import annotations

"""Matplotlib plotting functions for HybriDock-Pep analysis output.

Stub module — full implementation in Phase 6 Plan 06-04.
Provides the function signatures consumed by clustering.py so that
cluster_poses() can import and call them without error.

plot_convergence() and plot_silhouette() are the two public functions
called by cluster_poses().
"""

import logging
from pathlib import Path
from typing import Any

from hybridock_pep.models import ScoredPose

_log = logging.getLogger(__name__)


def plot_convergence(
    scored_poses: list[ScoredPose],
    output_path: Path,
) -> None:
    """Plot score convergence over sampled poses and save to a PNG file.

    Shows cumulative best hybrid_score as a function of pose index, which
    visualises whether the sampling run has converged (i.e., top score
    is no longer improving).

    Args:
        scored_poses: List of ScoredPose objects sorted by pose_idx.
            hybrid_score may be None for failed poses (skipped in plot).
        output_path: Destination path for the PNG. Parent directories are
            created if they do not exist.

    Note:
        When matplotlib is not installed, a zero-byte placeholder file is
        written and a warning is logged. Full plotting requires score-env
        with matplotlib installed.
    """
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning(
            "matplotlib not available; writing placeholder for %s", output_path
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = [p.pose_idx for p in scored_poses if p.hybrid_score is not None]
    scores = [p.hybrid_score for p in scored_poses if p.hybrid_score is not None]

    # Cumulative best (running minimum)
    best_so_far: list[float] = []
    current_best = float("inf")
    for s in scores:
        current_best = min(current_best, s)  # type: ignore[type-var]
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(indices, scores, color="steelblue", alpha=0.5, linewidth=0.8, label="Hybrid score")
    ax.plot(indices, best_so_far, color="tomato", linewidth=1.5, label="Cumulative best")
    ax.set_xlabel("Pose index")
    ax.set_ylabel("Hybrid score (kcal/mol)")
    ax.set_title("Score convergence over sampling")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    _log.info("Convergence plot written to %s", output_path)


def plot_silhouette(
    sil_scores: dict[int, float],
    *,
    k_optimal: int,
    output_path: Path,
) -> None:
    """Plot silhouette scores vs k and highlight the selected k_optimal.

    Args:
        sil_scores: Dict mapping k → silhouette score for each k tried.
            May be empty (n < 10 shortcut — no silhouette loop was run).
        k_optimal: The k selected by argmax silhouette; highlighted in red.
        output_path: Destination path for the PNG. Parent directories are
            created if they do not exist.

    Note:
        When matplotlib is not installed, a zero-byte placeholder file is
        written and a warning is logged.
    """
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning(
            "matplotlib not available; writing placeholder for %s", output_path
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    if sil_scores:
        ks = sorted(sil_scores)
        vals = [sil_scores[k] for k in ks]
        ax.bar(ks, vals, color="steelblue", alpha=0.7, label="Silhouette score")
        if k_optimal in sil_scores:
            ax.bar(
                [k_optimal],
                [sil_scores[k_optimal]],
                color="tomato",
                label=f"Selected k={k_optimal}",
            )
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette k-search")
        ax.legend(loc="upper right")
    else:
        # n < 10: no silhouette search; show placeholder
        ax.text(
            0.5,
            0.5,
            f"k fixed at {k_optimal}\n(n < 10; no silhouette search)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Silhouette k-search (skipped)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    _log.info("Silhouette plot written to %s", output_path)
