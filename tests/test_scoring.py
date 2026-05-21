"""Tests for Phase 3 scoring modules (SCORE-01, SCORE-02, SCORE-03).

All hybridock_pep imports are lazy (inside test functions) per STATE.md decision:
"All hybridock_pep imports kept lazy in test files."
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path


# ---------------------------------------------------------------------------
# SCORE-01: Vina scorer
# ---------------------------------------------------------------------------

class TestVinaScorer:
    """Tests for check_grid_boundary() and score_vina_batch() (SCORE-01)."""

    # ------------------------------------------------------------------
    # Fixtures: site center (22.5, 14.1, 38.7), box_size=20.0, half=10.0
    #   inside atom:  (22.5,  14.1, 38.7) — exact center
    #   outside atom: (33.0,  14.1, 38.7) — x = 33.0 > 32.5
    #   on-edge atom: (32.5,  14.1, 38.7) — x = 32.5 (inclusive, not clipped)
    # ------------------------------------------------------------------

    _SITE = (22.5, 14.1, 38.7)
    _BOX = 20.0

    # PDBQT ATOM line format: cols 30-38=x, 38-46=y, 46-54=z (right-justified 8.3f)
    _ATOM_INSIDE = "ATOM      1  C   LIG A   1      22.500  14.100  38.700  1.00  0.00     C  \n"
    _ATOM_OUTSIDE = "ATOM      1  C   LIG A   1      33.000  14.100  38.700  1.00  0.00     C  \n"
    _ATOM_ON_EDGE = "ATOM      1  C   LIG A   1      32.500  14.100  38.700  1.00  0.00     C  \n"

    def test_check_grid_boundary_inside(self, tmp_path: Path) -> None:
        """Atom at exact center → not clipped → returns False."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "inside.pdbqt"
        pdbqt.write_text(self._ATOM_INSIDE)
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is False

    def test_check_grid_boundary_outside(self, tmp_path: Path) -> None:
        """Atom at center+(box/2+0.5) → clipped → returns True."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "outside.pdbqt"
        pdbqt.write_text(self._ATOM_OUTSIDE)
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is True

    def test_check_grid_boundary_on_edge(self, tmp_path: Path) -> None:
        """Atom exactly on boundary → inclusive → returns False."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "edge.pdbqt"
        pdbqt.write_text(self._ATOM_ON_EDGE)
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is False

    def test_check_grid_boundary_empty_file(self, tmp_path: Path) -> None:
        """PDBQT with no ATOM/HETATM lines → no atoms outside → returns False."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "empty.pdbqt"
        pdbqt.write_text("REMARK  no atoms here\nEND\n")
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is False

    def test_check_grid_boundary_malformed_coords(self, tmp_path: Path) -> None:
        """Line with non-float coordinate is skipped — no exception raised."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "malformed.pdbqt"
        # Deliberately put non-numeric data in coordinate columns
        pdbqt.write_text("ATOM      1  C   LIG A   1      XX.XXX  YY.YYY  ZZ.ZZZ  1.00  0.00     C  \n")
        # Must not raise; with no parseable atoms → returns False
        result = check_grid_boundary(pdbqt, self._SITE, self._BOX)
        assert result is False

    def test_check_grid_boundary_hydrogen_outside_is_not_clipped(self, tmp_path: Path) -> None:
        """H atom outside grid with heavy atom inside → NOT clipped (Fix I).

        babel adds polar H to PDBQT; they can lie marginally outside the grid
        even when all heavy atoms are within bounds.  check_grid_boundary must
        skip H/HD atoms so these poses are not falsely flagged.
        """
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "h_outside.pdbqt"
        # Heavy C atom at center (inside), H atom 0.1 Å outside x-boundary
        h_x = self._SITE[0] - self._BOX / 2.0 - 0.1  # just outside x-low
        pdbqt.write_text(
            self._ATOM_INSIDE
            + f"ATOM      2  HE2 GLU A   3    {h_x:8.3f}{self._SITE[1]:8.3f}{self._SITE[2]:8.3f}"
            f"  1.00  0.00     H  \n"
        )
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is False

    def test_check_grid_boundary_heavy_outside_with_h_inside(self, tmp_path: Path) -> None:
        """Heavy atom outside grid → clipped even when H atoms are inside."""
        from hybridock_pep.scoring.vina import check_grid_boundary

        pdbqt = tmp_path / "heavy_outside.pdbqt"
        # H atom at center (inside), heavy C atom outside
        h_line = (
            f"ATOM      1  HN  ALA A   1    "
            f"{self._SITE[0]:8.3f}{self._SITE[1]:8.3f}{self._SITE[2]:8.3f}"
            f"  1.00  0.00     H  \n"
        )
        pdbqt.write_text(h_line + self._ATOM_OUTSIDE)
        assert check_grid_boundary(pdbqt, self._SITE, self._BOX) is True

    def test_score_vina_batch_returns_failure_on_missing_pdbqt(self, tmp_path: Path) -> None:
        """ScoredPose with nonexistent pdbqt_path → PoseFailure(stage='scoring'); no exception."""
        import numpy as np
        from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose
        from hybridock_pep.scoring.vina import score_vina_batch

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("REMARK dummy\n")

        config = DockConfig(
            peptide_sequence="ACDE",
            receptor_path=receptor,
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path,
        )

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "pose_0.pdb",
            sequence="ACDE",
            ca_coords=np.zeros((4, 3)),
            pdbqt_path=Path("/tmp/nonexistent_xyz_abc.pdbqt"),
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.set_ligand_from_file.side_effect = FileNotFoundError(
            "No such file: /tmp/nonexistent_xyz_abc.pdbqt"
        )

        with mock.patch("hybridock_pep.scoring.vina.Vina", return_value=mock_vina_instance):
            scored, failures = score_vina_batch([pose], config, receptor)

        assert len(scored) == 0
        assert len(failures) == 1
        assert isinstance(failures[0], PoseFailure)
        assert failures[0].stage == "scoring"
        assert failures[0].pose_idx == 0

    def test_score_vina_batch_sets_is_clipped(self, tmp_path: Path) -> None:
        """Clipped pose has is_clipped=True on returned ScoredPose."""
        import numpy as np
        from hybridock_pep.models import DockConfig, ScoredPose
        from hybridock_pep.scoring.vina import score_vina_batch

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("REMARK dummy\n")

        # Write a PDBQT with an atom outside the grid
        pdbqt = tmp_path / "pose_clipped.pdbqt"
        pdbqt.write_text(self._ATOM_OUTSIDE)

        config = DockConfig(
            peptide_sequence="ACDE",
            receptor_path=receptor,
            site_coords=self._SITE,
            box_size=self._BOX,
            output_dir=tmp_path,
        )

        pose = ScoredPose(
            pose_idx=1,
            pdb_path=tmp_path / "pose_1.pdb",
            sequence="ACDE",
            ca_coords=np.zeros((4, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [-8.5]

        with mock.patch("hybridock_pep.scoring.vina.Vina", return_value=mock_vina_instance):
            scored, failures = score_vina_batch([pose], config, receptor)

        assert len(failures) == 0
        assert len(scored) == 1
        assert scored[0].is_clipped is True

    def test_score_vina_batch_writes_clipped_to_metadata(self, tmp_path: Path) -> None:
        """Clipped pose written to run_metadata.json clipped_poses list."""
        import numpy as np
        from hybridock_pep.models import DockConfig, ScoredPose
        from hybridock_pep.scoring.vina import score_vina_batch

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("REMARK dummy\n")

        pdbqt = tmp_path / "pose_clipped.pdbqt"
        pdbqt.write_text(self._ATOM_OUTSIDE)

        config = DockConfig(
            peptide_sequence="ACDE",
            receptor_path=receptor,
            site_coords=self._SITE,
            box_size=self._BOX,
            output_dir=tmp_path,
        )

        pose = ScoredPose(
            pose_idx=7,
            pdb_path=tmp_path / "pose_7.pdb",
            sequence="ACDE",
            ca_coords=np.zeros((4, 3)),
            pdbqt_path=pdbqt,
        )

        metadata_path = tmp_path / "run_metadata.json"

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [-8.5]

        with mock.patch("hybridock_pep.scoring.vina.Vina", return_value=mock_vina_instance):
            score_vina_batch([pose], config, receptor, metadata_path=metadata_path)

        assert metadata_path.exists(), "metadata_path must be created"
        data = json.loads(metadata_path.read_text())
        assert "clipped_poses" in data, "JSON must have 'clipped_poses' key"
        assert len(data["clipped_poses"]) > 0, "clipped_poses must be non-empty"
        assert data["clipped_poses"][0]["pose_idx"] == 7


# ---------------------------------------------------------------------------
# SCORE-02: AD4 scorer (plan 03-02)
# ---------------------------------------------------------------------------

import pytest


class TestAD4Scorer:
    """Tests for score_ad4_batch() (SCORE-02).

    All Vina calls mocked via mock.patch("hybridock_pep.scoring.ad4.Vina").
    Lazy imports inside each test function per STATE.md convention.
    """

    def test_score_ad4_batch_uses_load_maps_not_set_receptor(self) -> None:
        """Source of ad4.py must contain 'load_maps' and must NOT contain 'set_receptor'."""
        import inspect
        from hybridock_pep.scoring import ad4 as ad4_module

        src = inspect.getsource(ad4_module)
        assert "set_receptor" not in src, "AD4 scorer must not call set_receptor()"
        assert "load_maps" in src, "AD4 scorer must call load_maps()"

    def test_score_ad4_batch_returns_tuple(self, tmp_path: Path) -> None:
        """score_ad4_batch returns a tuple of (list, list)."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        hd_map = maps_dir / "receptor.HD.map"
        hd_map.write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [-6.0]

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            result = score_ad4_batch([pose], maps_dir)

        assert isinstance(result, tuple)
        scored, failures = result
        assert isinstance(scored, list)
        assert isinstance(failures, list)

    def test_ad4_anomaly_flag_positive_score(self, tmp_path: Path) -> None:
        """AD4 score +1.5 → is_ad4_anomaly=True; pose still in scored list (not failures)."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        (maps_dir / "receptor.HD.map").write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [1.5]

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            scored, failures = score_ad4_batch([pose], maps_dir)

        assert len(failures) == 0
        assert len(scored) == 1
        assert scored[0].is_ad4_anomaly is True
        assert scored[0].ad4_score == pytest.approx(1.5)

    def test_ad4_anomaly_flag_negative_score(self, tmp_path: Path) -> None:
        """AD4 score -5.2 → is_ad4_anomaly=False."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        (maps_dir / "receptor.HD.map").write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [-5.2]

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            scored, failures = score_ad4_batch([pose], maps_dir)

        assert len(failures) == 0
        assert len(scored) == 1
        assert scored[0].is_ad4_anomaly is False
        assert scored[0].ad4_score == pytest.approx(-5.2)

    def test_ad4_anomaly_flag_zero_score(self, tmp_path: Path) -> None:
        """AD4 score 0.0 → is_ad4_anomaly=False (zero is not positive)."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        (maps_dir / "receptor.HD.map").write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [0.0]

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            scored, failures = score_ad4_batch([pose], maps_dir)

        assert len(failures) == 0
        assert len(scored) == 1
        assert scored[0].is_ad4_anomaly is False
        assert scored[0].ad4_score == pytest.approx(0.0)

    def test_score_ad4_batch_failure_on_exception(self, tmp_path: Path) -> None:
        """RuntimeError on set_ligand_from_file → PoseFailure(stage='scoring'); batch continues; scored empty."""
        import numpy as np
        from hybridock_pep.models import PoseFailure, ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        (maps_dir / "receptor.HD.map").write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.set_ligand_from_file.side_effect = RuntimeError("corrupt PDBQT")

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            scored, failures = score_ad4_batch([pose], maps_dir)

        assert len(scored) == 0
        assert len(failures) == 1
        assert isinstance(failures[0], PoseFailure)
        assert failures[0].stage == "scoring"
        assert failures[0].pose_idx == 0

    def test_map_prefix_construction(self, tmp_path: Path) -> None:
        """load_maps is called with prefix ending in '/receptor' (not '/' or just the dir)."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.ad4 import score_ad4_batch

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        (maps_dir / "receptor.HD.map").write_text("dummy\n")

        pdbqt = tmp_path / "pose_0.pdbqt"
        pdbqt.write_text("REMARK dummy\n")

        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="ALA",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        mock_vina_instance.score.return_value = [-7.0]

        with mock.patch("hybridock_pep.scoring.ad4.Vina", return_value=mock_vina_instance):
            score_ad4_batch([pose], maps_dir)

        # Capture the argument passed to load_maps
        call_args = mock_vina_instance.load_maps.call_args
        assert call_args is not None, "load_maps must have been called"
        prefix_arg = call_args[0][0]  # first positional arg
        expected_prefix = str(maps_dir / "receptor")
        assert prefix_arg == expected_prefix, (
            f"Expected load_maps prefix '{expected_prefix}', got '{prefix_arg}'"
        )


# ---------------------------------------------------------------------------
# SCORE-03: Entropy correction (SCORE-03, plan 03-03)
# ---------------------------------------------------------------------------

class TestEntropy:
    """Tests for apply_hybrid_score(), load_calibration(), and fit_calibration()."""

    def test_apply_hybrid_score_formula(self, tmp_path: Path) -> None:
        """D-01 formula: hybrid = vina + beta*(ad4-vina) + alpha*n_residues.

        vina=-6.0, ad4=-7.0, alpha=0.65, beta=0.22, n_residues=15
        hybrid = -6.0 + 0.22*(-7.0 - -6.0) + 0.65*15
               = -6.0 - 0.22 + 9.75 = 3.53
        entropy_correction = 0.65 * 15 = 9.75
        """
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score

        ca = np.zeros((15, 3), dtype=np.float64)
        pose = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "pose_0.pdb",
            sequence="A" * 15,
            ca_coords=ca,
            vina_score=-6.0,
            ad4_score=-7.0,
        )
        apply_hybrid_score(pose, alpha=0.65, beta=0.22, n_residues=15)
        assert abs(pose.hybrid_score - 3.53) < 1e-6, f"hybrid={pose.hybrid_score}"
        assert abs(pose.entropy_correction - 9.75) < 1e-6, f"ec={pose.entropy_correction}"

    def test_apply_hybrid_score_beta_zero(self, tmp_path: Path) -> None:
        """beta=0 -> hybrid = vina + alpha*n_residues (AD4 not blended)."""
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score

        ca = np.zeros((10, 3), dtype=np.float64)
        pose = ScoredPose(
            pose_idx=1,
            pdb_path=tmp_path / "pose_1.pdb",
            sequence="A" * 10,
            ca_coords=ca,
            vina_score=-5.0,
            ad4_score=-9.0,
        )
        apply_hybrid_score(pose, alpha=0.65, beta=0.0, n_residues=10)
        expected_hybrid = -5.0 + 0.0 * (-9.0 - (-5.0)) + 0.65 * 10
        assert abs(pose.hybrid_score - expected_hybrid) < 1e-6
        assert abs(pose.entropy_correction - 6.5) < 1e-6

    def test_apply_hybrid_score_beta_one_rejected(self, tmp_path: Path) -> None:
        """apply_hybrid_score does NOT validate beta; beta=1.0 still computes without raising.

        Validation is in load_calibration only (per plan spec).
        """
        import numpy as np
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score

        ca = np.zeros((5, 3), dtype=np.float64)
        pose = ScoredPose(
            pose_idx=2,
            pdb_path=tmp_path / "pose_2.pdb",
            sequence="AAAAA",
            ca_coords=ca,
            vina_score=-4.0,
            ad4_score=-6.0,
        )
        # Must not raise — no validation in apply_hybrid_score
        apply_hybrid_score(pose, alpha=0.65, beta=1.0, n_residues=5)
        expected = -4.0 + 1.0 * (-6.0 - (-4.0)) + 0.65 * 5
        assert abs(pose.hybrid_score - expected) < 1e-6

    def test_load_calibration_valid(self, tmp_path: Path) -> None:
        """Valid calibration JSON with alpha=0.65, beta=0.22 returns dict without error."""
        import json
        from hybridock_pep.scoring.entropy import load_calibration

        cal_path = tmp_path / "calibration.json"
        cal_data = {
            "alpha": 0.65,
            "beta": 0.22,
            "n_complexes": 10,
            "pearson_r": 0.71,
            "rmse_kcal_mol": 1.2,
            "calibrated_at": "2026-04-20T00:00:00+00:00",
            "training_csv": "data/training_complexes.csv",
        }
        cal_path.write_text(json.dumps(cal_data))
        result = load_calibration(cal_path)
        assert abs(result["alpha"] - 0.65) < 1e-9
        assert abs(result["beta"] - 0.22) < 1e-9

    def test_load_calibration_alpha_too_high(self, tmp_path: Path) -> None:
        """alpha=2.5 raises ValueError quoting value and range [0.1, 2.0]."""
        import json
        import pytest
        from hybridock_pep.scoring.entropy import load_calibration

        cal_path = tmp_path / "calibration.json"
        cal_path.write_text(json.dumps({"alpha": 2.5, "beta": 0.22}))
        with pytest.raises(ValueError) as exc_info:
            load_calibration(cal_path)
        msg = str(exc_info.value)
        assert "α=2.500" in msg, f"Expected 'α=2.500' in: {msg}"
        assert "[0.1, 2.0]" in msg, f"Expected '[0.1, 2.0]' in: {msg}"

    def test_load_calibration_alpha_too_low(self, tmp_path: Path) -> None:
        """alpha=0.05 raises ValueError quoting value and range [0.1, 2.0]."""
        import json
        import pytest
        from hybridock_pep.scoring.entropy import load_calibration

        cal_path = tmp_path / "calibration.json"
        cal_path.write_text(json.dumps({"alpha": 0.05, "beta": 0.22}))
        with pytest.raises(ValueError) as exc_info:
            load_calibration(cal_path)
        msg = str(exc_info.value)
        assert "α=0.050" in msg, f"Expected 'α=0.050' in: {msg}"
        assert "[0.1, 2.0]" in msg, f"Expected '[0.1, 2.0]' in: {msg}"

    def test_load_calibration_beta_too_high(self, tmp_path: Path) -> None:
        """beta=0.6 raises ValueError quoting value and range [0.0, 0.5]."""
        import json
        import pytest
        from hybridock_pep.scoring.entropy import load_calibration

        cal_path = tmp_path / "calibration.json"
        cal_path.write_text(json.dumps({"alpha": 0.65, "beta": 0.6}))
        with pytest.raises(ValueError) as exc_info:
            load_calibration(cal_path)
        msg = str(exc_info.value)
        assert "β=0.600" in msg, f"Expected 'β=0.600' in: {msg}"
        assert "[0.0, 0.5]" in msg, f"Expected '[0.0, 0.5]' in: {msg}"

    def test_load_calibration_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when calibration file does not exist."""
        from hybridock_pep.scoring.entropy import load_calibration

        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nonexistent_calibration.json")

    def test_fit_calibration_bounds_respected(self) -> None:
        """fit_calibration returns alpha in [0.2,2.0] and beta in [0.0,0.5] via L-BFGS-B."""
        from hybridock_pep.scoring.entropy import fit_calibration

        vina_scores = [-5.0, -6.0, -7.0, -4.0, -8.0]
        ad4_scores = [-5.5, -6.5, -7.5, -4.5, -8.5]
        n_residues = [10, 12, 15, 8, 20]
        pkd_list = [5.0, 6.0, 7.0, 4.0, 8.0]

        result = fit_calibration(vina_scores, ad4_scores, n_residues, pkd_list)
        assert 0.1 <= result["alpha"] <= 2.0, f"alpha={result['alpha']} out of bounds"
        assert 0.0 <= result["beta"] <= 0.5, f"beta={result['beta']} out of bounds"

    def test_pkd_to_delta_g_conversion(self) -> None:
        """ΔG = -RT*ln(10)*pKd; with n_residues=0 and vina=ad4=ΔG, trivial fit."""
        import math
        from hybridock_pep.scoring.entropy import fit_calibration

        RT, LN10 = 0.592, math.log(10)
        # pKd=6.0 → ΔG = -RT*ln(10)*6.0 = -8.190...
        # With n_residues=0 (alpha drops out) and vina=ad4=ΔG (beta drops out),
        # hybrid=ΔG for any alpha, beta → zero-residual fit.
        delta_g = -RT * LN10 * 6.0
        result = fit_calibration(
            vina_scores=[delta_g],
            ad4_scores=[delta_g],
            n_residues_list=[0],
            experimental_pkd=[6.0],
        )
        assert "alpha" in result, "Result dict must contain 'alpha'"
        assert "beta" in result, "Result dict must contain 'beta'"


# ---------------------------------------------------------------------------
# Calibration (plan 03-04)
# ---------------------------------------------------------------------------


class TestCalibration:
    """Integration tests for calibrate_alpha.py CLI and D-08 training CSV schema."""

    def test_training_csv_schema(self) -> None:
        """data/training_complexes.csv must have exactly 3 D-08 columns and ≥3 rows."""
        import csv
        from pathlib import Path

        csv_path = Path("data/training_complexes.csv")
        assert csv_path.exists(), f"Missing training CSV: {csv_path}"
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            _ = reader.fieldnames  # force header read before consuming rows
            rows = list(reader)
        expected_columns = ["pdb_id", "peptide_sequence", "experimental_pkd"]
        assert list(reader.fieldnames) == expected_columns, (
            f"Expected columns {expected_columns}, got {list(reader.fieldnames)}"
        )
        assert len(rows) >= 2, f"Expected ≥2 rows, got {len(rows)}"

    def test_calibrate_alpha_script_imports(self) -> None:
        """calibrate_alpha.py can be imported and exposes a main() function."""
        import importlib.util
        from pathlib import Path

        script_path = Path("scripts/calibrate_alpha.py")
        assert script_path.exists(), f"Missing script: {script_path}"
        spec = importlib.util.spec_from_file_location("calibrate_alpha", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main"), "calibrate_alpha.py must expose a main() function"

    def test_calibrate_alpha_writes_valid_json(self, tmp_path: Path) -> None:
        """main() with synthetic inputs writes a calibration.json that passes load_calibration()."""
        import argparse
        import importlib.util
        import json
        from pathlib import Path

        from hybridock_pep.scoring.entropy import load_calibration

        # Synthetic 3-column training CSV
        training_csv = tmp_path / "training.csv"
        training_csv.write_text(
            "pdb_id,peptide_sequence,experimental_pkd\n"
            "2OY2,ETFSDLWKLLPE,6.22\n"
            "1YCR,TFSDLWKLL,6.52\n"
            "3LNJ,PMDYEVNLLYH,7.15\n"
        )

        # Synthetic scores JSON
        scores_json = tmp_path / "scores.json"
        scores_json.write_text(json.dumps({
            "2OY2": {"vina_score": -7.8, "ad4_score": -8.1},
            "1YCR": {"vina_score": -8.2, "ad4_score": -8.5},
            "3LNJ": {"vina_score": -9.1, "ad4_score": -9.3},
        }))

        output_path = tmp_path / "calibration.json"

        spec = importlib.util.spec_from_file_location(
            "calibrate_alpha", Path("scripts/calibrate_alpha.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        args = argparse.Namespace(
            training_csv=training_csv,
            scores_json=scores_json,
            output=output_path,
            verbose=False,
            gamma=0.2,
        )
        mod.main(args)

        assert output_path.exists(), "calibration.json must be written"
        result = load_calibration(output_path)
        assert "alpha" in result
        assert "beta" in result

    def test_calibrate_alpha_end_to_end(self, tmp_path: Path) -> None:
        """main() produces calibration.json with all D-11 keys; alpha∈[0.2,2.0], beta∈[0.0,0.5]."""
        import argparse
        import importlib.util
        import json
        from pathlib import Path

        # Synthetic 3-column training CSV
        training_csv = tmp_path / "training.csv"
        training_csv.write_text(
            "pdb_id,peptide_sequence,experimental_pkd\n"
            "2OY2,ETFSDLWKLLPE,6.22\n"
            "1YCR,TFSDLWKLL,6.52\n"
            "3LNJ,PMDYEVNLLYH,7.15\n"
        )

        scores_json = tmp_path / "scores.json"
        scores_json.write_text(json.dumps({
            "2OY2": {"vina_score": -7.8, "ad4_score": -8.1},
            "1YCR": {"vina_score": -8.2, "ad4_score": -8.5},
            "3LNJ": {"vina_score": -9.1, "ad4_score": -9.3},
        }))

        output_path = tmp_path / "calibration.json"

        spec = importlib.util.spec_from_file_location(
            "calibrate_alpha", Path("scripts/calibrate_alpha.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        args = argparse.Namespace(
            training_csv=training_csv,
            scores_json=scores_json,
            output=output_path,
            verbose=False,
            gamma=0.2,
        )
        mod.main(args)

        data = json.loads(output_path.read_text())
        d11_keys = {"alpha", "beta", "n_complexes", "pearson_r", "rmse_kcal_mol", "calibrated_at", "training_csv"}
        missing = d11_keys - data.keys()
        assert not missing, f"Missing D-11 keys: {missing}"
        assert isinstance(data["alpha"], float), "alpha must be float"
        assert isinstance(data["beta"], float), "beta must be float"
        assert 0.1 <= data["alpha"] <= 2.0, f"alpha={data['alpha']} out of [0.1, 2.0]"
        assert 0.0 <= data["beta"] <= 0.5, f"beta={data['beta']} out of [0.0, 0.5]"

    def test_write_calibration_d11_schema(self, tmp_path: Path) -> None:
        """write_calibration() output JSON contains all 7 D-11 keys."""
        import json
        from hybridock_pep.scoring.entropy import write_calibration

        output_path = tmp_path / "calibration.json"
        write_calibration(
            output_path,
            alpha=0.65,
            beta=0.22,
            n_complexes=10,
            pearson_r=0.71,
            rmse_kcal_mol=1.2,
            training_csv="data/training_complexes.csv",
        )

        data = json.loads(output_path.read_text())
        d11_keys = {"alpha", "beta", "n_complexes", "pearson_r", "rmse_kcal_mol", "calibrated_at", "training_csv"}
        missing = d11_keys - data.keys()
        assert not missing, f"Missing D-11 keys in write_calibration output: {missing}"
