"""Tests for hybridock_pep.reproducibility."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hybridock_pep.models import ScoredPose
from hybridock_pep.reproducibility import (
    ReproducibilityResult,
    _kabsch_rmsd,
    _per_residue_pearson,
    _verdict,
    run_reproducibility,
)


def _scored(coords: np.ndarray, hybrid: float, tmp_path: Path) -> ScoredPose:
    return ScoredPose(
        pose_idx=0,
        pdb_path=tmp_path / "p.pdb",
        sequence="LIS",
        ca_coords=coords,
        hybrid_score=hybrid,
    )


class TestKabschRMSD:
    def test_identical_coords_give_zero_rmsd(self) -> None:
        a = np.random.RandomState(0).randn(10, 3)
        assert _kabsch_rmsd(a, a.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_translated_coords_give_zero_after_centering(self) -> None:
        a = np.random.RandomState(0).randn(8, 3)
        b = a + np.array([5.0, -3.0, 2.0])
        assert _kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_rotated_coords_give_zero_after_alignment(self) -> None:
        a = np.random.RandomState(0).randn(8, 3)
        # Random rotation around z-axis 30°
        c, s = np.cos(np.pi/6), np.sin(np.pi/6)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        b = a @ R.T
        assert _kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_independent_random_coords_give_nonzero(self) -> None:
        a = np.random.RandomState(0).randn(10, 3) * 3
        b = np.random.RandomState(1).randn(10, 3) * 3
        assert _kabsch_rmsd(a, b) > 0.5


class TestPearson:
    def test_identical_gives_perfect_pearson(self) -> None:
        a = np.random.RandomState(0).randn(10, 3)
        assert _per_residue_pearson(a, a.copy()) == pytest.approx(1.0, abs=1e-9)


class TestVerdict:
    def test_high_quality(self) -> None:
        assert _verdict(0.5, 0.98) == "highly reproducible"

    def test_reproducible(self) -> None:
        assert _verdict(1.8, 0.90) == "reproducible"

    def test_moderate(self) -> None:
        assert _verdict(4.0, 0.70) == "moderately reproducible"

    def test_low(self) -> None:
        assert _verdict(6.5, 0.40) == "low reproducibility (RAPiDock pose diversity dominates)"


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    return p


class TestRunReproducibility:
    def test_rejects_fewer_than_two_seeds(self, tmp_path: Path, valid_receptor: Path) -> None:
        from hybridock_pep.models import DockConfig
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="≥2 seeds"):
            run_reproducibility(cfg, tmp_path / "c.json", seeds=[42])

    def test_three_seed_run_with_mocked_driver(
        self, tmp_path: Path, valid_receptor: Path
    ) -> None:
        from hybridock_pep.models import DockConfig

        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        # Three runs whose top-1 poses sit ~1 Å apart in Cα RMSD
        rng = np.random.RandomState(0)
        c1 = rng.randn(5, 3)
        c2 = c1 + rng.randn(5, 3) * 0.4
        c3 = c1 + rng.randn(5, 3) * 0.4

        with patch("hybridock_pep.reproducibility.driver.run_dock") as mock:
            mock.side_effect = [
                ([_scored(c1, -10.0, tmp_path)], None),
                ([_scored(c2, -10.3, tmp_path)], None),
                ([_scored(c3, -9.8, tmp_path)], None),
            ]
            result = run_reproducibility(
                cfg, tmp_path / "c.json", seeds=[1, 2, 3]
            )

        assert result.n_runs == 3
        assert len(result.top1_dg_per_run) == 3
        assert result.mean_pairwise_rmsd > 0
        assert result.max_pairwise_rmsd >= result.mean_pairwise_rmsd
        # Tiny perturbations should still register decent Pearson
        assert result.mean_pairwise_pearson > 0.5
        assert result.verdict in {
            "highly reproducible", "reproducible",
            "moderately reproducible",
            "low reproducibility (RAPiDock pose diversity dominates)",
        }

    def test_inconsistent_pose_lengths_raise(
        self, tmp_path: Path, valid_receptor: Path
    ) -> None:
        from hybridock_pep.models import DockConfig
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        with patch("hybridock_pep.reproducibility.driver.run_dock") as mock:
            mock.side_effect = [
                ([_scored(np.zeros((5, 3)), -10.0, tmp_path)], None),
                ([_scored(np.zeros((7, 3)), -10.0, tmp_path)], None),
            ]
            with pytest.raises(ValueError, match="inconsistent Cα counts"):
                run_reproducibility(cfg, tmp_path / "c.json", seeds=[1, 2])

    def test_no_scored_poses_raises(self, tmp_path: Path, valid_receptor: Path) -> None:
        from hybridock_pep.models import DockConfig
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        empty_pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "p.pdb",
            sequence="LIS", ca_coords=np.zeros((3, 3)),
        )
        with patch("hybridock_pep.reproducibility.driver.run_dock") as mock:
            mock.side_effect = [([empty_pose], None), ([empty_pose], None)]
            with pytest.raises(ValueError, match="no scored poses"):
                run_reproducibility(cfg, tmp_path / "c.json", seeds=[1, 2])


class TestResultJson:
    def test_to_json_includes_all_fields(self) -> None:
        r = ReproducibilityResult(
            seeds=[1, 2, 3], mean_pairwise_rmsd=0.8, max_pairwise_rmsd=1.1,
            mean_pairwise_pearson=0.97, n_runs=3,
            top1_dg_per_run=[-9.5, -9.6, -9.4], dg_std=0.1,
            verdict="highly reproducible",
        )
        j = r.to_json()
        assert j["n_runs"] == 3
        assert j["verdict"] == "highly reproducible"
        assert "mean_pairwise_rmsd_A" in j
