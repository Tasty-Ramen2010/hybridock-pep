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
# SCORE-02: AD4 scorer (stub — implemented in plan 03-02)
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.skip(reason="implemented in plan 03-02")
class TestAD4Scorer:
    pass


# ---------------------------------------------------------------------------
# SCORE-03: Entropy correction (stub — implemented in plan 03-03)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="implemented in plan 03-03")
class TestEntropy:
    pass


# ---------------------------------------------------------------------------
# Calibration (stub — implemented in plan 03-03)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="implemented in plan 03-03")
class TestCalibration:
    pass
