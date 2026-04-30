"""Tests for hybridock_pep.output.csv_writer (OUT-01, OUT-02, OUT-03)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_config(tmp_path: Path, output_dir: Path | None = None):
    """Build a minimal DockConfig backed by the real receptor_tiny.pdb fixture."""
    from hybridock_pep.models import DockConfig

    return DockConfig(
        peptide_sequence="ACDEF",
        receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=output_dir or tmp_path / "out",
        seed=42,
    )


def _make_scored_pose(
    idx: int,
    tmp_path: Path,
    hybrid_score: float = -5.0,
    vina_score: float = -4.0,
    ad4_score: float = -3.0,
    entropy_correction: float = 0.5,
    cluster_id: int = 0,
    is_ad4_anomaly: bool = False,
    is_clipped: bool = False,
):
    """Build a ScoredPose with all score fields populated. pdb_path need not exist on disk."""
    from hybridock_pep.models import ScoredPose

    return ScoredPose(
        pose_idx=idx,
        pdb_path=tmp_path / f"pose_{idx:03d}.pdb",
        sequence="ACDEF",
        ca_coords=np.zeros((5, 3)),
        pdbqt_path=tmp_path / f"pose_{idx:03d}.pdbqt",
        hybrid_score=hybrid_score,
        vina_score=vina_score,
        ad4_score=ad4_score,
        entropy_correction=entropy_correction,
        cluster_id=cluster_id,
        is_ad4_anomaly=is_ad4_anomaly,
        is_clipped=is_clipped,
    )


class TestWriteRankedCsv:
    def test_write_ranked_csv_creates_file(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_ranked_csv

        config = _make_config(tmp_path)
        poses = [_make_scored_pose(i, tmp_path, hybrid_score=-float(i + 1)) for i in range(3)]
        result = write_ranked_csv(poses, config)

        assert result.exists(), "ranked_poses.csv must be created"

    def test_write_ranked_csv_columns(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_ranked_csv, FIELDNAMES

        config = _make_config(tmp_path)
        poses = [_make_scored_pose(0, tmp_path)]
        write_ranked_csv(poses, config)

        csv_path = config.output_dir / "ranked_poses.csv"
        rows = list(csv.DictReader(csv_path.open()))
        assert set(rows[0].keys()) == set(FIELDNAMES), (
            f"CSV columns mismatch. Got: {sorted(rows[0].keys())}"
        )

    def test_write_ranked_csv_sorted_ascending(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_ranked_csv

        config = _make_config(tmp_path)
        poses = [
            _make_scored_pose(0, tmp_path, hybrid_score=-1.0),
            _make_scored_pose(1, tmp_path, hybrid_score=-5.0),
            _make_scored_pose(2, tmp_path, hybrid_score=-3.0),
        ]
        write_ranked_csv(poses, config)

        csv_path = config.output_dir / "ranked_poses.csv"
        rows = list(csv.DictReader(csv_path.open()))
        scores = [float(r["hybrid_score"]) for r in rows]
        assert scores == sorted(scores), f"Rows not sorted ascending: {scores}"
        assert scores[0] == -5.0, f"Best score should be first; got {scores[0]}"

    def test_write_ranked_csv_top10_limit(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_ranked_csv

        config = _make_config(tmp_path)
        poses = [_make_scored_pose(i, tmp_path, hybrid_score=-float(i + 1)) for i in range(15)]
        write_ranked_csv(poses, config)

        csv_path = config.output_dir / "ranked_poses.csv"
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 10, f"Expected 10 rows, got {len(rows)}"

    def test_write_ranked_csv_delta_g_equals_hybrid(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_ranked_csv

        config = _make_config(tmp_path)
        poses = [_make_scored_pose(i, tmp_path, hybrid_score=-float(i + 1) * 1.5) for i in range(5)]
        write_ranked_csv(poses, config)

        csv_path = config.output_dir / "ranked_poses.csv"
        rows = list(csv.DictReader(csv_path.open()))
        for row in rows:
            assert row["delta_g"] == row["hybrid_score"], (
                f"delta_g {row['delta_g']} != hybrid_score {row['hybrid_score']} for rank {row['rank']}"
            )


class TestWriteBestPosePdb:
    def _make_cluster_result(self, tmp_path: Path, poses_dir: Path) -> Any:
        """Build a ClusterResult with two clusters; cluster 1 is better."""
        from hybridock_pep.analysis.clustering import ClusterResult

        (poses_dir / "pose_0.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        (poses_dir / "pose_1.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C\nEND\n"
        )

        return ClusterResult(
            k_optimal=2,
            silhouette_score=0.75,
            per_cluster_stats=[
                {
                    "cluster_id": 0,
                    "n_poses": 5,
                    "mean_hybrid_score": -3.0,
                    "std_hybrid_score": 0.5,
                    "best_pose_idx": 0,
                    "ci_lower": -3.5,
                    "ci_upper": -2.5,
                },
                {
                    "cluster_id": 1,
                    "n_poses": 5,
                    "mean_hybrid_score": -7.0,
                    "std_hybrid_score": 0.3,
                    "best_pose_idx": 1,
                    "ci_lower": -7.3,
                    "ci_upper": -6.7,
                },
            ],
        )

    def _make_scored_poses(self, poses_dir: Path) -> list:
        """Build minimal ScoredPose list matching _make_cluster_result's best_pose_idx values."""
        from hybridock_pep.models import ScoredPose

        poses = []
        for idx in (0, 1):
            p = ScoredPose(
                pose_idx=idx,
                pdb_path=poses_dir / f"pose_{idx}.pdb",
                sequence="ACDEF",
                ca_coords=__import__("numpy").zeros((5, 3)),
                hybrid_score=-3.0 * (idx + 1),
            )
            poses.append(p)
        return poses

    def test_write_best_pose_pdb_copies_file(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_best_pose_pdb

        poses_dir = tmp_path / "out" / "poses"
        poses_dir.mkdir(parents=True)
        config = _make_config(tmp_path, output_dir=tmp_path / "out")
        cluster_result = self._make_cluster_result(tmp_path, poses_dir)
        scored_poses = self._make_scored_poses(poses_dir)

        result = write_best_pose_pdb(cluster_result, config, scored_poses)

        assert result.exists(), "best_pose.pdb must be created"
        assert result.stat().st_size > 0, "best_pose.pdb must not be empty"

    def test_write_best_pose_pdb_selects_best_cluster(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_best_pose_pdb

        poses_dir = tmp_path / "out" / "poses"
        poses_dir.mkdir(parents=True)
        config = _make_config(tmp_path, output_dir=tmp_path / "out")
        cluster_result = self._make_cluster_result(tmp_path, poses_dir)
        scored_poses = self._make_scored_poses(poses_dir)

        write_best_pose_pdb(cluster_result, config, scored_poses)
        dest_content = (tmp_path / "out" / "best_pose.pdb").read_text()
        assert "1.000   1.000   1.000" in dest_content, (
            "best_pose.pdb should contain pose_1.pdb content (cluster 1, score -7.0), "
            "not pose_0.pdb (cluster 0, score -3.0)"
        )
