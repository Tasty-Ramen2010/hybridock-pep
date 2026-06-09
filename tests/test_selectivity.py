"""Tests for hybridock_pep.selectivity."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hybridock_pep.models import ScoredPose
from hybridock_pep.selectivity import (
    SelectivityResult,
    _bootstrap_ddg,
    _top_k_dg,
    run_selectivity,
)


def _make_scored(idx: int, hybrid: float, tmp_path: Path) -> ScoredPose:
    # Selectivity ΔΔG now ranks on vina_score (the cross-receptor signal), not the
    # entropy-corrected hybrid which cancels in ΔΔG. Set both for the helper.
    return ScoredPose(
        pose_idx=idx,
        pdb_path=tmp_path / f"p{idx}.pdb",
        sequence="LIS",
        ca_coords=np.zeros((3, 3)),
        vina_score=hybrid,
        hybrid_score=hybrid,
    )


class TestTopKDg:
    def test_sorts_ascending_and_caps_at_k(self, tmp_path: Path) -> None:
        poses = [
            _make_scored(0, -5.0, tmp_path),
            _make_scored(1, -8.0, tmp_path),
            _make_scored(2, -6.0, tmp_path),
            _make_scored(3, -7.0, tmp_path),
        ]
        result = _top_k_dg(poses, k=2)
        assert result == [-8.0, -7.0]

    def test_returns_all_when_k_exceeds_n(self, tmp_path: Path) -> None:
        poses = [_make_scored(0, -5.0, tmp_path), _make_scored(1, -7.0, tmp_path)]
        assert _top_k_dg(poses, k=10) == [-7.0, -5.0]

    def test_skips_poses_without_score(self, tmp_path: Path) -> None:
        poses = [
            _make_scored(0, -5.0, tmp_path),
            ScoredPose(pose_idx=1, pdb_path=tmp_path / "p1.pdb",
                       sequence="LIS", ca_coords=np.zeros((3, 3))),  # no vina_score
            _make_scored(2, -7.0, tmp_path),
        ]
        assert _top_k_dg(poses, k=5) == [-7.0, -5.0]

    def test_raises_when_no_scored_poses(self, tmp_path: Path) -> None:
        poses = [ScoredPose(pose_idx=0, pdb_path=tmp_path / "p.pdb",
                            sequence="LIS", ca_coords=np.zeros((3, 3)))]
        with pytest.raises(ValueError, match="vina_score"):
            _top_k_dg(poses, k=1)

    def test_selectivity_ignores_hybrid_uses_vina(self, tmp_path: Path) -> None:
        # A pose with a great hybrid but poor vina must rank by vina for ΔΔG.
        p = ScoredPose(pose_idx=0, pdb_path=tmp_path / "p.pdb", sequence="LIS",
                       ca_coords=np.zeros((3, 3)), vina_score=-3.0, hybrid_score=-99.0)
        q = ScoredPose(pose_idx=1, pdb_path=tmp_path / "q.pdb", sequence="LIS",
                       ca_coords=np.zeros((3, 3)), vina_score=-8.0, hybrid_score=-1.0)
        assert _top_k_dg([p, q], k=1) == [-8.0]  # ranked by vina, not hybrid


class TestBootstrapDdg:
    def test_zero_difference_gives_tight_ci_around_zero(self) -> None:
        rng = np.random.default_rng(0)
        # Identical samples → ΔΔG always 0
        t = [-8.0, -8.0, -8.0, -8.0]
        o = [-8.0, -8.0, -8.0, -8.0]
        lo, hi = _bootstrap_ddg(t, o, n_iter=500, rng=rng)
        assert lo == 0.0 and hi == 0.0

    def test_distinct_means_give_ci_around_point_estimate(self) -> None:
        rng = np.random.default_rng(42)
        t = [-10.0, -10.5, -11.0, -9.5, -10.0]  # mean −10.2
        o = [-7.0, -7.5, -8.0, -6.5, -7.0]      # mean −7.2
        # ΔΔG ≈ −3.0; CI should bracket that
        lo, hi = _bootstrap_ddg(t, o, n_iter=2000, rng=rng)
        assert lo < -3.0 < hi
        assert hi - lo < 2.5  # tight-ish for these tiny samples

    def test_ci_widens_with_noisy_inputs(self) -> None:
        rng = np.random.default_rng(7)
        t = list(np.random.RandomState(1).normal(-10, 3, 20))
        o = list(np.random.RandomState(2).normal(-10, 3, 20))
        lo, hi = _bootstrap_ddg(t, o, n_iter=1000, rng=rng)
        # Identical distributions → CI should straddle 0
        assert lo < 0 < hi


class TestSelectivityResult:
    def test_to_json_marks_selective_for_target(self) -> None:
        r = SelectivityResult(
            peptide="LIS", target_dg=-10.0, offtarget_dg=-6.0,
            ddg=-4.0, ddg_ci_low=-5.0, ddg_ci_high=-3.0,
            n_target_poses=10, n_offtarget_poses=10,
            bootstrap_n=1000, top_k=10,
        )
        assert r.to_json()["interpretation"] == "Selective for target"

    def test_to_json_marks_selective_for_offtarget(self) -> None:
        r = SelectivityResult(
            peptide="LIS", target_dg=-6.0, offtarget_dg=-10.0,
            ddg=4.0, ddg_ci_low=3.0, ddg_ci_high=5.0,
            n_target_poses=10, n_offtarget_poses=10,
            bootstrap_n=1000, top_k=10,
        )
        assert r.to_json()["interpretation"] == "Selective for off-target"

    def test_to_json_marks_inconclusive_when_ci_crosses_zero(self) -> None:
        r = SelectivityResult(
            peptide="LIS", target_dg=-8.0, offtarget_dg=-7.5,
            ddg=-0.5, ddg_ci_low=-1.5, ddg_ci_high=+0.5,
            n_target_poses=10, n_offtarget_poses=10,
            bootstrap_n=1000, top_k=10,
        )
        assert r.to_json()["interpretation"] == "Inconclusive (CI crosses zero)"


class TestRunSelectivity:
    """Integration test with the docking driver mocked.

    Mocks `driver.run_dock` so the test runs without a CUDA + Vina environment.
    Verifies that ΔΔG = mean(target top-K) − mean(offtarget top-K) and the
    bootstrap CI bounds the point estimate.
    """

    def test_end_to_end_with_mocked_driver(self, tmp_path: Path, valid_receptor) -> None:
        from hybridock_pep.models import DockConfig

        target_cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0,
            output_dir=tmp_path / "tgt",
        )
        offtarget_cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0,
            output_dir=tmp_path / "off",
        )

        tgt_poses = [_make_scored(i, -10.0 - i * 0.1, tmp_path) for i in range(20)]
        off_poses = [_make_scored(i, -7.0 - i * 0.1, tmp_path) for i in range(20)]

        with patch("hybridock_pep.selectivity.driver.run_dock") as mock_dock:
            mock_dock.side_effect = [(tgt_poses, None), (off_poses, None)]
            result = run_selectivity(
                peptide="LIS",
                target_config=target_cfg,
                offtarget_config=offtarget_cfg,
                calibration_path=tmp_path / "cal.json",
                top_k=5,
                bootstrap_n=200,
                seed=0,
            )

        # 20 poses at -10.0..-11.9 / -7.0..-8.9; top-5 are -11.5..-11.9 mean=-11.7
        # and -8.5..-8.9 mean=-8.7
        assert result.target_dg == pytest.approx(-11.7, abs=1e-6)
        assert result.offtarget_dg == pytest.approx(-8.7, abs=1e-6)
        assert result.ddg == pytest.approx(-3.0, abs=1e-6)
        assert result.ddg_ci_low <= result.ddg <= result.ddg_ci_high
        assert result.ddg_ci_high < 0  # selective for target with non-overlapping samples

    def test_rejects_shared_output_dir(self, tmp_path: Path, valid_receptor) -> None:
        from hybridock_pep.models import DockConfig
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0,
            output_dir=tmp_path / "shared",
        )
        with pytest.raises(ValueError, match="output_dir must differ"):
            run_selectivity("LIS", cfg, cfg, calibration_path=tmp_path / "c.json")


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    return p
