from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hybridock_pep import DockConfig, PoseFailure, PoseRecord, ScoredPose


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    return p


class TestDockConfig:
    def test_valid_construction(self, valid_receptor: Path, tmp_path: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="LISDAELEAIFEADC",
            receptor_path=valid_receptor,
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path,
        )
        assert cfg.peptide_sequence == "LISDAELEAIFEADC"
        assert isinstance(cfg.run_id, str) and len(cfg.run_id) > 0
        assert cfg.n_samples == 100
        # Default scoring set narrowed from {"vina", "ad4"} to {"vina"} in
        # June 2026 — AD4 dropped after production-pose ridge fit gave w_ad4=0
        # (see docs/calibration_notes.md "Why AD4 commits nothing" section).
        # AD4 still opt-in via DockConfig(scoring={"vina", "ad4"}).
        assert cfg.scoring == {"vina"}

    def test_uppercases_sequence(self, valid_receptor: Path, tmp_path: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="lisdaele",
            receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path,
        )
        assert cfg.peptide_sequence == "LISDAELE"

    def test_invalid_peptide_sequence(self, valid_receptor: Path, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="Non-standard amino acid"):
            DockConfig(
                peptide_sequence="LISB",
                receptor_path=valid_receptor,
                site_coords=(0.0, 0.0, 0.0),
                box_size=20.0,
                output_dir=tmp_path,
            )

    def test_missing_receptor(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="does not exist"):
            DockConfig(
                peptide_sequence="LIS",
                receptor_path=tmp_path / "nonexistent.pdb",
                site_coords=(0.0, 0.0, 0.0),
                box_size=20.0,
                output_dir=tmp_path,
            )

    def test_nonpositive_box_size(self, valid_receptor: Path, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="box_size must be positive"):
            DockConfig(
                peptide_sequence="LIS",
                receptor_path=valid_receptor,
                site_coords=(0.0, 0.0, 0.0),
                box_size=0.0,
                output_dir=tmp_path,
            )

    def test_empty_peptide_sequence(self, valid_receptor: Path, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            DockConfig(
                peptide_sequence="",
                receptor_path=valid_receptor,
                site_coords=(0.0, 0.0, 0.0),
                box_size=20.0,
                output_dir=tmp_path,
            )

    def test_nonpositive_n_samples(self, valid_receptor: Path, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="n_samples must be positive"):
            DockConfig(
                peptide_sequence="LIS",
                receptor_path=valid_receptor,
                site_coords=(0.0, 0.0, 0.0),
                box_size=20.0,
                n_samples=0,
                output_dir=tmp_path,
            )


class TestPoseRecord:
    def test_construction(self, tmp_path: Path) -> None:
        ca = np.zeros((15, 3), dtype=np.float64)
        pr = PoseRecord(
            pose_idx=0,
            pdb_path=tmp_path / "pose_0.pdb",
            sequence="LISDAELEAIFEADC",
            ca_coords=ca,
        )
        assert pr.ca_coords.shape == (15, 3)
        assert pr.pose_idx == 0


class TestScoredPose:
    def test_is_pose_record(self, tmp_path: Path) -> None:
        ca = np.zeros((3, 3), dtype=np.float64)
        sp = ScoredPose(
            pose_idx=0,
            pdb_path=tmp_path / "p.pdb",
            sequence="LIS",
            ca_coords=ca,
        )
        assert isinstance(sp, PoseRecord)
        assert sp.vina_score is None
        assert sp.ad4_score is None
        assert sp.is_ad4_anomaly is False
        assert sp.is_clipped is False


class TestPoseFailure:
    def test_construction(self) -> None:
        pf = PoseFailure(pose_idx=7, stage="parsing", error_msg="malformed ATOM record")
        assert pf.stage == "parsing"
        assert pf.pose_idx == 7
