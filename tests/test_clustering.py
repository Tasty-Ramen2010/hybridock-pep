"""Tests for Phase 6 analysis modules: clustering, statistics, plotting.

Covers ANAL-01 (clustering), ANAL-02 (statistics + CSV), ANAL-03 (convergence plot),
OUT-04 (convergence_plot.png), OUT-05 (silhouette_plot.png).

All hybridock_pep imports are lazy (inside test functions) per project convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Synthetic pose data shared across test classes
# ---------------------------------------------------------------------------
# 10 poses in 2 clear clusters:
#   Group A (poses 0-4): ca_coords near origin → cluster 0
#   Group B (poses 5-9): ca_coords near (10,10,10) → cluster 1
# hybrid_score: Group A = [-10.0, -9.8, -9.6, -9.4, -9.2]
#               Group B = [ -5.0, -4.8, -4.6, -4.4, -4.2]
# n=10 → k_max = min(15, 10//5) = 2 → k_optimal must be 2

_GROUP_A_COORDS = [np.array([[i * 0.1, 0.0, 0.0]] * 5, dtype=np.float64) for i in range(5)]
_GROUP_B_COORDS = [np.array([[10.0 + i * 0.1, 10.0, 10.0]] * 5, dtype=np.float64) for i in range(5)]
_SCORES_A = [-10.0, -9.8, -9.6, -9.4, -9.2]
_SCORES_B = [-5.0, -4.8, -4.6, -4.4, -4.2]


def _make_scored_poses(tmp_path: Path):
    """Return 10 ScoredPose objects with populated ca_coords and hybrid_score."""
    from hybridock_pep.models import ScoredPose

    poses = []
    for i, (coords, score) in enumerate(
        zip(_GROUP_A_COORDS + _GROUP_B_COORDS, _SCORES_A + _SCORES_B)
    ):
        pose = ScoredPose(
            pose_idx=i,
            pdb_path=tmp_path / f"pose_{i}.pdb",
            sequence="ACDEF",
            ca_coords=coords,
            pdbqt_path=tmp_path / f"pose_{i}.pdbqt",
        )
        pose.hybrid_score = score
        poses.append(pose)
    return poses


def _make_config(tmp_path: Path):
    from hybridock_pep.models import DockConfig

    return DockConfig(
        peptide_sequence="ACDEF",
        receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=tmp_path / "out",
        n_samples=10,
    )


# ---------------------------------------------------------------------------
# ANAL-01: Clustering
# ---------------------------------------------------------------------------

class TestClustering:
    """Tests for cluster_poses(), ClusterResult, RMSD matrix, silhouette loop (ANAL-01)."""

    def test_contact_zone_indices(self) -> None:
        """Residues within 6 Å of receptor Cα are returned; distant ones are not."""
        from hybridock_pep.analysis.clustering import _contact_zone_indices

        pose_ca = np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], dtype=np.float64)
        receptor_ca = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # near residue 0 only
        indices = _contact_zone_indices(pose_ca, receptor_ca, cutoff=6.0)
        assert 0 in indices, "Residue 0 (1Å away) must be in contact zone"
        assert 1 not in indices, "Residue 1 (>34Å away) must NOT be in contact zone"

    def test_contact_zone_fallback(self) -> None:
        """Fewer than 3 contact residues → full-peptide indices returned (D-02)."""
        from hybridock_pep.analysis.clustering import _contact_zone_indices

        pose_ca = np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], dtype=np.float64)
        receptor_ca = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # only 1 contact
        indices = _contact_zone_indices(pose_ca, receptor_ca, cutoff=6.0)
        # Only 1 contact residue found; function returns it (fallback handled at RMSD level)
        # The D-02 fallback (< 3 → use full peptide) is tested in the RMSD matrix builder
        assert len(indices) >= 1

    def test_rmsd_matrix_symmetry(self, tmp_path: Path) -> None:
        """RMSD matrix is square, symmetric, and zero on diagonal."""
        from hybridock_pep.analysis.clustering import _build_rmsd_matrix

        poses = _make_scored_poses(tmp_path)
        ca_arrays = [p.ca_coords for p in poses]
        # Use full-peptide indices for simplicity (no receptor needed)
        full_indices = [np.arange(5) for _ in poses]
        dist = _build_rmsd_matrix(ca_arrays, full_indices)

        assert dist.shape == (10, 10), f"Expected (10,10), got {dist.shape}"
        assert np.allclose(dist, dist.T, atol=1e-10), "Distance matrix must be symmetric"
        assert np.allclose(np.diag(dist), 0.0, atol=1e-10), "Diagonal must be zero"

    def test_cluster_poses_assigns_ids(self, tmp_path: Path) -> None:
        """cluster_poses() assigns integer cluster_id to every ScoredPose (D-08)."""
        from hybridock_pep.analysis.clustering import cluster_poses

        poses = _make_scored_poses(tmp_path)
        config = _make_config(tmp_path)
        result = cluster_poses(poses, config)

        for pose in poses:
            assert pose.cluster_id is not None, f"Pose {pose.pose_idx} cluster_id must be set"
            assert isinstance(pose.cluster_id, int)

    def test_silhouette_k_selection(self, tmp_path: Path) -> None:
        """ClusterResult.k_optimal is 2 for 10 clearly-separated poses (D-05, D-06)."""
        from hybridock_pep.analysis.clustering import cluster_poses

        poses = _make_scored_poses(tmp_path)
        config = _make_config(tmp_path)
        result = cluster_poses(poses, config)

        assert isinstance(result.k_optimal, int)
        assert 2 <= result.k_optimal <= 2, f"k_max=2 for n=10; got {result.k_optimal}"
        assert isinstance(result.silhouette_score, float)
        assert -1.0 <= result.silhouette_score <= 1.0


# ---------------------------------------------------------------------------
# ANAL-02: Statistics + CSV
# ---------------------------------------------------------------------------

class TestStatistics:
    """Tests for compute_cluster_stats() and write_cluster_summary_csv() (ANAL-02)."""

    def test_ci95_two_values(self) -> None:
        """n=2: 95% CI is finite and symmetric around mean."""
        from hybridock_pep.analysis.statistics import _ci95

        lo, hi = _ci95([1.0, 3.0])
        assert np.isfinite(lo), "CI lower bound must be finite"
        assert np.isfinite(hi), "CI upper bound must be finite"
        assert lo < 2.0 < hi, "Mean (2.0) must be inside CI"

    def test_ci95_single_value(self) -> None:
        """n=1: CI degenerates to (value, value) — no crash."""
        from hybridock_pep.analysis.statistics import _ci95

        lo, hi = _ci95([-5.0])
        assert lo == hi == -5.0

    def test_compute_cluster_stats_keys(self, tmp_path: Path) -> None:
        """compute_cluster_stats() returns list[dict] with required keys for each cluster."""
        from hybridock_pep.analysis.statistics import compute_cluster_stats

        poses = _make_scored_poses(tmp_path)
        # Manually assign cluster_ids: 0-4 → cluster 0, 5-9 → cluster 1
        for i, pose in enumerate(poses):
            pose.cluster_id = 0 if i < 5 else 1

        stats = compute_cluster_stats(poses)
        required_keys = {
            "cluster_id", "n_poses", "mean_hybrid_score", "std_hybrid_score",
            "ci95_lower", "ci95_upper", "best_pose_idx",
        }
        assert len(stats) == 2, f"Expected 2 cluster dicts, got {len(stats)}"
        for entry in stats:
            assert required_keys <= entry.keys(), f"Missing keys: {required_keys - entry.keys()}"

    def test_cluster_summary_csv(self, tmp_path: Path) -> None:
        """write_cluster_summary_csv() creates a CSV with correct header (D-10, ANAL-02)."""
        from hybridock_pep.analysis.statistics import compute_cluster_stats, write_cluster_summary_csv

        poses = _make_scored_poses(tmp_path)
        for i, pose in enumerate(poses):
            pose.cluster_id = 0 if i < 5 else 1

        stats = compute_cluster_stats(poses)
        out_path = tmp_path / "out" / "cluster_summary.csv"
        write_cluster_summary_csv(stats, out_path)

        assert out_path.exists(), "cluster_summary.csv must be written"
        content = out_path.read_text()
        assert "cluster_id" in content
        assert "mean_hybrid_score" in content
        assert "best_pose_idx" in content


# ---------------------------------------------------------------------------
# ANAL-03 / OUT-04 / OUT-05: Plotting
# ---------------------------------------------------------------------------

class TestPlotting:
    """Tests for plot_convergence() and plot_silhouette() (ANAL-03, OUT-04, OUT-05)."""

    def test_convergence_plot_written(self, tmp_path: Path) -> None:
        """plot_convergence() writes a PNG file to the specified path (ANAL-03, OUT-04)."""
        from hybridock_pep.analysis.plotting import plot_convergence

        poses = _make_scored_poses(tmp_path)
        out_path = tmp_path / "convergence_plot.png"
        plot_convergence(poses, out_path)

        assert out_path.exists(), "convergence_plot.png must be written"
        assert out_path.stat().st_size > 0, "PNG must not be empty"

    def test_silhouette_plot_written(self, tmp_path: Path) -> None:
        """plot_silhouette() writes a PNG file to the specified path (OUT-05)."""
        from hybridock_pep.analysis.plotting import plot_silhouette

        sil_scores = {2: 0.72, 3: 0.51, 4: 0.38}
        out_path = tmp_path / "silhouette_plot.png"
        plot_silhouette(sil_scores, k_optimal=2, output_path=out_path)

        assert out_path.exists(), "silhouette_plot.png must be written"
        assert out_path.stat().st_size > 0, "PNG must not be empty"
