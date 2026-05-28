from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # MUST be before any import of matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

from hybridock_pep.models import ScoredPose

logger = logging.getLogger(__name__)


def plot_convergence(
    scored_poses: list[ScoredPose],
    output_path: Path,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 150,
) -> None:
    """Generate convergence plot showing running mean ± σ of hybrid score (score-sorted).

    Sorts poses by hybrid_score ascending (most negative first) and plots running
    mean ± σ to visualize ranking stability. This tests how quickly the top-N
    score distribution stabilizes — NOT arrival-order sampling convergence.

    Args:
        scored_poses: Poses with populated hybrid_score. Order does not matter;
            poses are sorted internally by hybrid_score ascending.
        output_path: Absolute path to write the PNG file.
        figsize: Matplotlib figure size in inches. Default (8, 5).
        dpi: Output resolution. Default 150.

    Raises:
        ValueError: If scored_poses is empty.
    """
    if not scored_poses:
        raise ValueError("scored_poses must not be empty")

    scores = sorted(p.hybrid_score for p in scored_poses)  # ascending
    n = len(scores)
    ns = np.arange(1, n + 1)
    running_mean = np.array([np.mean(scores[:i]) for i in ns])
    running_std = np.array([np.std(scores[:i], ddof=0) for i in ns])  # population σ

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ns, running_mean, color="steelblue", label="Running mean")
    ax.fill_between(
        ns,
        running_mean - running_std,
        running_mean + running_std,
        alpha=0.3,
        color="steelblue",
        label="±σ",
    )
    ax.set_xlabel("Top-N poses (score-sorted)")
    ax.set_ylabel("Hybrid score (kcal/mol)")
    ax.set_title("Score-sorted convergence: ranking stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)  # CRITICAL: release memory; avoids ResourceWarning in tests
    logger.info("Wrote convergence plot to %s", output_path)


def plot_silhouette(
    sil_scores: dict[int, float],
    k_optimal: int,
    output_path: Path,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 150,
) -> None:
    """Generate silhouette score plot across k range with k_optimal annotated.

    Args:
        sil_scores: Dict mapping k → silhouette_score for k = 2..k_max.
            Empty dict (k_max < 2 fallback) produces a minimal placeholder plot.
        k_optimal: The k value selected by argmax. Annotated with a vertical dashed line.
        output_path: Absolute path to write the PNG file.
        figsize: Matplotlib figure size in inches. Default (8, 5).
        dpi: Output resolution. Default 150.

    Raises:
        ValueError: If k_optimal < 2.
    """
    if k_optimal < 2:
        raise ValueError(f"k_optimal must be >= 2, got {k_optimal}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if sil_scores:
        ks = sorted(sil_scores.keys())
        scores = [sil_scores[k] for k in ks]
        ax.bar(ks, scores, color="steelblue", alpha=0.7, label="Silhouette score")
        ax.axvline(x=k_optimal, color="red", linestyle="--", label=f"k_optimal={k_optimal}")
        ax.set_xticks(ks)
    else:
        # k_max < 2 fallback: single bar at k_optimal
        ax.bar([k_optimal], [float("nan")], color="steelblue", alpha=0.7)
        ax.text(
            0.5,
            0.5,
            f"k_optimal={k_optimal} (fallback, n<10)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs. number of clusters")
    if any(isinstance(artist.get_label(), str)
           and artist.get_label() and not artist.get_label().startswith("_")
           for artist in ax.get_children()
           if hasattr(artist, "get_label")):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)  # CRITICAL: release memory; avoids ResourceWarning in tests
    logger.info("Wrote silhouette plot to %s", output_path)
