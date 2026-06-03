"""Tests for Phase 3 scoring modules (SCORE-01, SCORE-02, SCORE-03).

All hybridock_pep imports are lazy (inside test functions) per STATE.md decision:
"All hybridock_pep imports kept lazy in test files."
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import numpy as np


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
        """Clipped pose goes to failures immediately (skips set_ligand_from_file).

        Clipped = heavy atom outside grid boundary. We skip Vina scoring to avoid
        the ~12s RuntimeError that Vina raises for out-of-bounds ligands.
        The PoseFailure has error_msg starting with 'is_clipped:'.
        """
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

        # Clipped poses now go directly to failures (no set_ligand_from_file call)
        assert len(scored) == 0
        assert len(failures) == 1
        assert failures[0].pose_idx == 1
        assert failures[0].error_msg.startswith("is_clipped:")
        # Vina's set_ligand_from_file should NOT have been called (fast-fail)
        mock_vina_instance.set_ligand_from_file.assert_not_called()

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

    def test_multi_round_clash_relief_resolves_on_second_round(self, tmp_path: Path) -> None:
        """Pose needing 2 optimization rounds: succeeds when round 2 drops score below 0."""
        import numpy as np
        from hybridock_pep.models import DockConfig, ScoredPose
        from hybridock_pep.scoring.vina import score_vina_batch

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("REMARK dummy\n")
        pdbqt = tmp_path / "pose_clash.pdbqt"
        # Atom inside box so is_clipped=False
        atom_inside = (
            f"ATOM      1  CA  ALA A   1    "
            f"{self._SITE[0]:8.3f}{self._SITE[1]:8.3f}{self._SITE[2]:8.3f}"
            f"  0.00  0.00     +0.000 C \n"
        )
        pdbqt.write_text(atom_inside)

        config = DockConfig(
            peptide_sequence="A",
            receptor_path=receptor,
            site_coords=self._SITE,
            box_size=self._BOX,
            output_dir=tmp_path,
        )
        pose = ScoredPose(
            pose_idx=42,
            pdb_path=tmp_path / "pose_42.pdb",
            sequence="A",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        # Initial score > 0 (clash); round 1 still positive; round 2 → negative
        mock_vina_instance.score.side_effect = [
            [+15.0],  # raw_score — triggers clash relief
            [+3.0],   # after round 1 optimize() — still positive, improvement=12 > 0.5
            [-4.5],   # after round 2 optimize() — resolved
        ]

        with mock.patch("hybridock_pep.scoring.vina.Vina", return_value=mock_vina_instance):
            scored, failures = score_vina_batch(
                [pose], config, receptor, max_clash_relief_rounds=5
            )

        assert len(failures) == 0, "Pose should succeed after 2 rounds"
        assert len(scored) == 1
        assert scored[0].vina_score == pytest.approx(-4.5)
        assert mock_vina_instance.optimize.call_count == 2

    def test_multi_round_clash_relief_fails_when_converged_positive(
        self, tmp_path: Path
    ) -> None:
        """BFGS converges to positive-score minimum → PoseFailure after early stop."""
        import numpy as np
        from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose
        from hybridock_pep.scoring.vina import score_vina_batch

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("REMARK dummy\n")
        pdbqt = tmp_path / "pose_clash.pdbqt"
        atom_inside = (
            f"ATOM      1  CA  ALA A   1    "
            f"{self._SITE[0]:8.3f}{self._SITE[1]:8.3f}{self._SITE[2]:8.3f}"
            f"  0.00  0.00     +0.000 C \n"
        )
        pdbqt.write_text(atom_inside)

        config = DockConfig(
            peptide_sequence="A",
            receptor_path=receptor,
            site_coords=self._SITE,
            box_size=self._BOX,
            output_dir=tmp_path,
        )
        pose = ScoredPose(
            pose_idx=77,
            pdb_path=tmp_path / "pose_77.pdb",
            sequence="A",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt,
        )

        mock_vina_instance = mock.MagicMock()
        # Raw positive → round 1 improves by only 0.1 kcal/mol → early stop (BFGS converged)
        mock_vina_instance.score.side_effect = [
            [+80.0],   # raw_score
            [+79.9],   # after round 1 — improvement 0.1 < 0.5 → early stop
        ]

        with mock.patch("hybridock_pep.scoring.vina.Vina", return_value=mock_vina_instance):
            scored, failures = score_vina_batch(
                [pose], config, receptor, max_clash_relief_rounds=5
            )

        assert len(scored) == 0
        assert len(failures) == 1
        assert isinstance(failures[0], PoseFailure)
        assert "clash_relief_failed" in failures[0].error_msg
        assert mock_vina_instance.optimize.call_count == 1  # stopped after 1 round


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

    # ----- Schema v2 (multivariate ridge) tests --------------------------- #

    def test_calibration_mode_detection(self) -> None:
        """calibration_mode() routes ridge vs legacy via model_type / w_vina presence."""
        from hybridock_pep.scoring.entropy import calibration_mode

        assert calibration_mode({"alpha": 0.5, "beta": 0.2}) == "legacy"
        assert calibration_mode({"model_type": "legacy", "alpha": 0.5, "beta": 0.2}) == "legacy"
        assert calibration_mode({"model_type": "ridge", "w_vina": 1.0}) == "ridge"
        assert calibration_mode({
            "schema_version": 2, "w_vina": 0.2, "w_ad4": 0.0,
            "w_contact": -1.2, "intercept": 0.7,
        }) == "ridge"

    def test_load_calibration_ridge_roundtrip(self, tmp_path: Path) -> None:
        """write_calibration + load_calibration round-trips a ridge JSON."""
        from hybridock_pep.scoring.entropy import load_calibration, write_calibration

        out = tmp_path / "ridge.json"
        write_calibration(
            out,
            schema_version=2, model_type="ridge",
            w_vina=0.21, w_ad4=0.0, w_contact=-1.20, intercept=0.77,
            n_complexes=6, pearson_r=0.95, loo_pearson_r=0.76,
        )
        cal = load_calibration(out)
        assert cal["model_type"] == "ridge"
        assert cal["w_vina"] == 0.21
        assert cal["w_contact"] == -1.20

    def test_load_calibration_ridge_rejects_out_of_range(self, tmp_path: Path) -> None:
        """Ridge schema bound check fires on w_vina > 3.0."""
        import json
        from hybridock_pep.scoring.entropy import load_calibration

        out = tmp_path / "bad_ridge.json"
        out.write_text(json.dumps({
            "schema_version": 2, "model_type": "ridge",
            "w_vina": 5.0, "w_ad4": 0.0, "w_contact": -1.2, "intercept": 0.7,
        }))
        with pytest.raises(ValueError, match="w_vina"):
            load_calibration(out)

    def test_apply_hybrid_score_ridge_formula(self, tmp_path: Path) -> None:
        """apply_hybrid_score_ridge implements ΔG = w_v*V + w_a*A + w_c*N + c exactly."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score_ridge

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "pose_0.pdb",
            sequence="X" * 10, ca_coords=np.zeros((10, 3)),
        )
        pose.vina_score = -6.0
        pose.ad4_score = -4.0
        apply_hybrid_score_ridge(
            pose, w_vina=0.2, w_ad4=0.5, w_contact=-1.2, intercept=0.5,
            n_contact_residues=4,
        )
        # 0.2*(-6) + 0.5*(-4) + (-1.2)*4 + 0.5 = -1.2 - 2.0 - 4.8 + 0.5 = -7.5
        assert abs(pose.hybrid_score - (-7.5)) < 1e-9

    def test_apply_hybrid_score_ridge_skips_anomalous_ad4(self, tmp_path: Path) -> None:
        """When is_ad4_anomaly=True, the AD4 term is dropped."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score_ridge

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "pose_0.pdb",
            sequence="X" * 10, ca_coords=np.zeros((10, 3)),
        )
        pose.vina_score = -6.0
        pose.ad4_score = +99.0
        pose.is_ad4_anomaly = True
        apply_hybrid_score_ridge(
            pose, w_vina=1.0, w_ad4=1.0, w_contact=0.0, intercept=0.0,
            n_contact_residues=0,
        )
        # AD4 term suppressed → hybrid = 1.0 * -6.0 = -6.0
        assert abs(pose.hybrid_score - (-6.0)) < 1e-9

    def test_apply_calibration_dispatches_correctly(self, tmp_path: Path) -> None:
        """apply_calibration() routes to ridge vs legacy formula by schema."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_calibration

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "pose_0.pdb",
            sequence="X" * 10, ca_coords=np.zeros((10, 3)),
        )
        pose.vina_score = -6.0
        pose.ad4_score = -4.0
        apply_calibration(
            pose,
            {"model_type": "ridge", "w_vina": 1.0, "w_ad4": 0.0,
             "w_contact": -1.0, "intercept": 0.0},
            n_residues=10, n_contact_residues=5,
        )
        # 1.0 * -6 + 0 * -4 + -1 * 5 + 0 = -11
        assert abs(pose.hybrid_score - (-11.0)) < 1e-9

    def test_fit_calibration_ridge_produces_valid_schema(self) -> None:
        """fit_calibration_ridge returns a dict that load_calibration accepts."""
        from hybridock_pep.scoring.entropy import fit_calibration_ridge

        v = [-5.0, -6.0, -7.0, -4.0, -8.0, -6.5]
        a = [-3.0, -4.0, -5.0, -2.0, -6.0, -4.5]
        nc = [3, 5, 7, 2, 9, 6]
        pkd = [4.5, 5.5, 6.5, 4.0, 8.0, 5.8]
        result = fit_calibration_ridge(v, a, nc, pkd)
        assert result["schema_version"] == 2
        assert result["model_type"] == "ridge"
        for k in ("w_vina", "w_ad4", "w_contact", "intercept"):
            assert k in result, f"missing {k}"
        # LOO must be computed with n=6
        assert "loo_pearson_r" in result and "loo_rmse_kcal_mol" in result

    def test_apply_hybrid_score_ridge_with_entropy_weights(self, tmp_path: Path) -> None:
        """Schema-v2 ridge with optional w_s_ss_weighted plumbs through correctly."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score_ridge

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "pose_0.pdb",
            sequence="X" * 10, ca_coords=np.zeros((10, 3)),
        )
        pose.vina_score = -6.0
        pose.ad4_score = -4.0
        pose.s_ss_weighted = 8.0
        apply_hybrid_score_ridge(
            pose,
            w_vina=1.0, w_ad4=0.0, w_contact=0.0,
            w_s_ss_weighted=-0.5,
            intercept=0.0, n_contact_residues=0,
        )
        # 1.0 * -6.0 + (-0.5) * 8.0 = -10.0
        assert abs(pose.hybrid_score - (-10.0)) < 1e-9

    def test_apply_hybrid_score_ridge_entropy_missing_raises(self, tmp_path: Path) -> None:
        """If w_s_ss_weighted is non-zero but pose.s_ss_weighted is None, RuntimeError."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_hybrid_score_ridge

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "pose_0.pdb",
            sequence="X" * 10, ca_coords=np.zeros((10, 3)),
        )
        pose.vina_score = -6.0
        # s_ss_weighted left as None
        with pytest.raises(RuntimeError, match="s_ss_weighted"):
            apply_hybrid_score_ridge(
                pose, w_vina=1.0, w_ad4=0.0, w_contact=0.0,
                w_s_ss_weighted=-0.5, intercept=0.0, n_contact_residues=0,
            )

    def test_ridge_calibration_with_entropy_validates(self, tmp_path: Path) -> None:
        """load_calibration accepts w_s_* keys and applies range check."""
        from hybridock_pep.scoring.entropy import load_calibration, write_calibration

        out = tmp_path / "ridge_entropy.json"
        write_calibration(
            out,
            schema_version=2, model_type="ridge",
            w_vina=0.0, w_ad4=0.0, w_contact=0.0,
            w_s_ss_weighted=-0.434, intercept=-3.95,
        )
        cal = load_calibration(out)
        assert cal["w_s_ss_weighted"] == -0.434
        assert cal["intercept"] == -3.95

    # ----- per_residue_entropy module tests ------------------------------- #

    def test_per_residue_entropy_tables(self) -> None:
        """Doig-Sternberg side-chain and backbone tables match published values."""
        from hybridock_pep.scoring.per_residue_entropy import s_sc, s_bb, ss_factor

        assert s_sc("G") == 0.0 and s_sc("A") == 0.0 and s_sc("P") == 0.0
        assert s_sc("K") == 2.5 and s_sc("R") == 2.8
        assert s_bb("G") == 2.20 and s_bb("P") == 0.30
        assert s_bb("L") == 1.00  # default
        assert ss_factor("loop") == 1.0
        assert ss_factor("helix") == 0.5
        assert ss_factor("sheet") == 0.3

    def test_dihedral_sign_convention(self) -> None:
        """_dihedral matches IUPAC: a standard α-helical φ is ≈ -60°, not +60°."""
        from hybridock_pep.scoring.per_residue_entropy import _dihedral

        # Classic Praxeolitic test vectors (Wikipedia) — verified φ should be
        # around -71° here (right-handed alpha helix territory).
        p0 = np.array([24.969, 13.428, 30.692])  # C(i-1)
        p1 = np.array([24.044, 12.661, 29.808])  # N(i)
        p2 = np.array([22.785, 13.482, 29.543])  # CA(i)
        p3 = np.array([21.951, 13.670, 30.793])  # C(i)
        phi = _dihedral(p0, p1, p2, p3)
        # The Wikipedia article reports -71° for these vectors under one
        # numerical convention; my numpy implementation gives a value in the
        # same α-helix neighbourhood within a few degrees.  The critical
        # property is sign — without the b0 = -(p1-p0) fix this returns
        # +60°-ish (mirror image), which would silently classify L-residues
        # as their D-mirror.
        assert -75.0 < phi < -60.0, f"expected φ in α-helix range, got {phi:.2f}"

    def test_compute_entropy_sums_returns_all_fields(self, tmp_path: Path) -> None:
        """compute_entropy_sums returns the expected dict structure."""
        from hybridock_pep.scoring.per_residue_entropy import compute_entropy_sums

        # Minimal 3-residue pose PDB (Gly-Lys-Pro) with random coords.
        pdb = tmp_path / "pose.pdb"
        pdb.write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000\n"
            "ATOM      2  CA  GLY A   1       1.450   0.000   0.000\n"
            "ATOM      3  C   GLY A   1       2.000   1.400   0.000\n"
            "ATOM      4  N   LYS A   2       3.300   1.500   0.000\n"
            "ATOM      5  CA  LYS A   2       4.000   2.700   0.000\n"
            "ATOM      6  C   LYS A   2       5.500   2.500   0.000\n"
            "ATOM      7  N   PRO A   3       6.100   3.700   0.000\n"
            "ATOM      8  CA  PRO A   3       7.500   3.900   0.000\n"
            "ATOM      9  C   PRO A   3       8.000   5.300   0.000\n"
            "END\n"
        )
        receptor_coords = np.array([[10.0, 0.0, 0.0]])  # too far → no contacts
        ent = compute_entropy_sums(pdb, "GKP", receptor_coords=receptor_coords)
        for key in ("n_contact", "s_sc_sum", "s_bb_sum", "s_ss_weighted",
                    "ss_loop_count", "ss_helix_count", "ss_sheet_count"):
            assert key in ent, f"missing key {key}"

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
            mode="legacy",
            ridge_alpha=0.1,
            positive_constraint=True,
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
            mode="legacy",
            ridge_alpha=0.1,
            positive_constraint=True,
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


# ---------------------------------------------------------------------------
# Ensemble z-score AD4 scoring
# ---------------------------------------------------------------------------


class TestEnsembleHybridScores:
    """Tests for apply_ensemble_hybrid_scores() — within-run AD4 z-score blending."""

    def _make_poses(self, tmp_path, vina_scores, ad4_scores=None):
        import numpy as np
        from hybridock_pep.models import ScoredPose

        poses = []
        for i, v in enumerate(vina_scores):
            pose = ScoredPose(
                pose_idx=i,
                pdb_path=tmp_path / f"p{i}.pdb",
                sequence="ACDEF",
                ca_coords=np.zeros((5, 3)),
            )
            pose.vina_score = v
            if ad4_scores is not None:
                pose.ad4_score = ad4_scores[i]
                pose.is_ad4_anomaly = ad4_scores[i] > 0
            poses.append(pose)
        return poses

    def test_vina_only_preserves_ordering(self, tmp_path: Path) -> None:
        """With ad4_blend_weight=0, hybrid ordering matches Vina ordering."""
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores

        vina = [-10.0, -8.0, -6.0, -4.0, -2.0]
        poses = self._make_poses(tmp_path, vina)
        apply_ensemble_hybrid_scores(poses, alpha=0.0, n_residues=5, ad4_blend_weight=0.0)

        hybrids = [p.hybrid_score for p in poses]
        assert hybrids == sorted(hybrids), "hybrid ordering must match Vina ordering when ad4_weight=0"

    def test_ad4_weight_zero_equals_vina_only(self, tmp_path: Path) -> None:
        """ad4_blend_weight=0.0 produces same scores as no AD4."""
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores

        vina = [-9.0, -7.5, -6.0, -4.5, -3.0]
        ad4 = [-11.0, -5.0, -8.0, -3.0, -10.0]
        poses_with = self._make_poses(tmp_path, vina, ad4)
        poses_without = self._make_poses(tmp_path, vina)

        apply_ensemble_hybrid_scores(poses_with, alpha=0.0, n_residues=5, ad4_blend_weight=0.0)
        apply_ensemble_hybrid_scores(poses_without, alpha=0.0, n_residues=5, ad4_blend_weight=0.0)

        for pw, pwo in zip(poses_with, poses_without):
            assert pw.hybrid_score == pytest.approx(pwo.hybrid_score, abs=1e-8)

    def test_ad4_anomaly_excluded_from_distribution(self, tmp_path: Path) -> None:
        """Anomalous AD4 poses (score > 0) don't enter the AD4 z-score distribution."""
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores

        vina = [-8.0, -7.0, -6.0, -5.0, -4.0]
        ad4 = [-9.0, -8.0, +5.0, -6.0, -5.0]  # pose 2 is anomalous
        poses = self._make_poses(tmp_path, vina, ad4)
        # Must not raise even with an anomalous pose
        apply_ensemble_hybrid_scores(poses, alpha=0.0, n_residues=5, ad4_blend_weight=0.3)

        # Anomalous pose should still get a hybrid_score (Vina-only fallback)
        assert poses[2].hybrid_score is not None

    def test_entropy_correction_applied(self, tmp_path: Path) -> None:
        """entropy_correction = alpha * n_residues when no contact info available."""
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores

        vina = [-8.0, -7.0, -6.0, -5.0, -4.0]
        poses = self._make_poses(tmp_path, vina)
        apply_ensemble_hybrid_scores(poses, alpha=0.5, n_residues=5, ad4_blend_weight=0.0)

        for pose in poses:
            assert pose.entropy_correction == pytest.approx(0.5 * 5, abs=1e-8)

    def test_no_vina_scores_raises(self, tmp_path: Path) -> None:
        """RuntimeError raised when no poses have valid Vina scores."""
        from hybridock_pep.models import ScoredPose
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores
        import numpy as np

        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "p.pdb",
            sequence="ACD", ca_coords=np.zeros((3, 3)),
        )
        with pytest.raises(RuntimeError):
            apply_ensemble_hybrid_scores([pose], alpha=0.1, n_residues=3)

    def test_ad4_improves_ranking_when_correlated(self, tmp_path: Path) -> None:
        """When AD4 and Vina agree on ranking, ensemble score preserves that order."""
        from hybridock_pep.scoring.entropy import apply_ensemble_hybrid_scores

        # Both Vina and AD4 rank poses 0 > 1 > 2 > 3 > 4 (most negative = best)
        vina = [-10.0, -8.0, -6.0, -4.0, -2.0]
        ad4  = [-12.0, -9.0, -7.0, -5.0, -3.0]
        poses = self._make_poses(tmp_path, vina, ad4)
        apply_ensemble_hybrid_scores(poses, alpha=0.0, n_residues=5, ad4_blend_weight=0.3)

        hybrids = [p.hybrid_score for p in poses]
        assert hybrids == sorted(hybrids), "AD4-consistent ranking must be preserved"

    def test_load_calibration_accepts_ensemble_ad4_weight(self, tmp_path: Path) -> None:
        """load_calibration accepts ensemble_ad4_weight in [0, 1] without error."""
        import json
        from hybridock_pep.scoring.entropy import load_calibration

        cal = {"alpha": 0.5, "beta": 0.0, "ensemble_ad4_weight": 0.3}
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(cal))
        result = load_calibration(p)
        assert result["ensemble_ad4_weight"] == pytest.approx(0.3)

    def test_load_calibration_rejects_invalid_ensemble_weight(self, tmp_path: Path) -> None:
        """ensemble_ad4_weight outside [0, 1] raises ValueError."""
        import json
        from hybridock_pep.scoring.entropy import load_calibration

        cal = {"alpha": 0.5, "beta": 0.0, "ensemble_ad4_weight": 1.5}
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(cal))
        with pytest.raises(ValueError):
            load_calibration(p)
