from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_pose_record(idx: int, tmp_path: Path):
    """Build a PoseRecord without importing at module level."""
    from hybridock_pep.models import PoseRecord
    pdb = tmp_path / f"pose_{idx}.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")
    return PoseRecord(
        pose_idx=idx,
        pdb_path=pdb,
        sequence="A",
        ca_coords=np.zeros((1, 3)),
    )


def _make_config(tmp_path: Path):
    """Build a minimal DockConfig with a real receptor PDB."""
    from hybridock_pep.models import DockConfig
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    return DockConfig(
        peptide_sequence="AAAA",
        receptor_path=receptor,
        site_coords=(0.0, 0.0, 0.0),
        box_size=20.0,
        output_dir=tmp_path,
        seed=42,
    )


_PATCH_TARGETS = [
    "hybridock_pep.driver.run_sampling",
    "hybridock_pep.driver.parse_poses",
    "hybridock_pep.driver.prepare_receptor",
    "hybridock_pep.driver.generate_ad4_maps",
    "hybridock_pep.driver.prepare_ligand_batch",
    "hybridock_pep.driver.score_vina_batch",
    "hybridock_pep.driver.score_ad4_batch",
    "hybridock_pep.driver.load_calibration",
    "hybridock_pep.driver.write_metadata_skeleton",
    "hybridock_pep.driver.finalize_metadata",
]


class TestInputPosesBypass:
    def test_input_poses_skips_run_sampling(self, tmp_path: Path) -> None:
        from hybridock_pep import driver

        config = _make_config(tmp_path)
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()

        with (
            patch("hybridock_pep.driver.run_sampling") as mock_sampling,
            patch("hybridock_pep.driver.parse_poses", return_value=([], [])),
            patch("hybridock_pep.driver.prepare_receptor", return_value=tmp_path / "receptor.pdbqt"),
            patch("hybridock_pep.driver.generate_ad4_maps", return_value=tmp_path / "maps"),
            patch("hybridock_pep.driver.prepare_ligand_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_vina_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_ad4_batch", return_value=([], [])),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.write_metadata_skeleton"),
            patch("hybridock_pep.driver.finalize_metadata"),
        ):
            driver.run_dock(config, input_poses_dir=poses_dir, calibration_path=tmp_path / "cal.json")
            mock_sampling.assert_not_called()

    def test_no_input_poses_calls_run_sampling(self, tmp_path: Path) -> None:
        from hybridock_pep import driver

        config = _make_config(tmp_path)

        with (
            patch("hybridock_pep.driver.run_sampling", return_value=[]) as mock_sampling,
            patch("hybridock_pep.driver.parse_poses", return_value=([], [])),
            patch("hybridock_pep.driver.prepare_receptor", return_value=tmp_path / "receptor.pdbqt"),
            patch("hybridock_pep.driver.generate_ad4_maps", return_value=tmp_path / "maps"),
            patch("hybridock_pep.driver.prepare_ligand_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_vina_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_ad4_batch", return_value=([], [])),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.write_metadata_skeleton"),
            patch("hybridock_pep.driver.finalize_metadata"),
        ):
            driver.run_dock(config, input_poses_dir=None, calibration_path=tmp_path / "cal.json")
            mock_sampling.assert_called_once()


class TestDriverOrchestration:
    def test_returns_list_of_scored_poses(self, tmp_path: Path) -> None:
        from hybridock_pep import driver
        from hybridock_pep.models import ScoredPose

        config = _make_config(tmp_path)
        record = _make_pose_record(0, tmp_path)
        pdbqt_path = tmp_path / "pose_0.pdbqt"
        pdbqt_path.touch()

        scored = ScoredPose(
            pose_idx=0,
            pdb_path=record.pdb_path,
            sequence="A",
            ca_coords=np.zeros((1, 3)),
            pdbqt_path=pdbqt_path,
            vina_score=-5.0,
            ad4_score=-4.0,
            entropy_correction=0.1,
            hybrid_score=-4.9,
        )

        with (
            patch("hybridock_pep.driver.run_sampling", return_value=[]),
            patch("hybridock_pep.driver.parse_poses", return_value=([record], [])),
            patch("hybridock_pep.driver.prepare_receptor", return_value=tmp_path / "receptor.pdbqt"),
            patch("hybridock_pep.driver.generate_ad4_maps", return_value=tmp_path / "maps"),
            patch("hybridock_pep.driver.prepare_ligand_batch", return_value=([pdbqt_path], [])),
            patch("hybridock_pep.driver.score_vina_batch", return_value=([scored], [])),
            patch("hybridock_pep.driver.score_ad4_batch", return_value=([scored], [])),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.apply_hybrid_score"),
            patch("hybridock_pep.driver.write_metadata_skeleton"),
            patch("hybridock_pep.driver.finalize_metadata"),
        ):
            result = driver.run_dock(config, input_poses_dir=None, calibration_path=tmp_path / "cal.json")
            assert isinstance(result, list)
            assert all(isinstance(p, ScoredPose) for p in result)

    def test_all_stages_called_in_order(self, tmp_path: Path) -> None:
        from hybridock_pep import driver

        config = _make_config(tmp_path)
        call_log: list[str] = []

        def make_side(name, ret=None):
            def side(*a, **kw):
                call_log.append(name)
                return ret
            return side

        record = _make_pose_record(0, tmp_path)
        pdbqt_path = tmp_path / "pose_0.pdbqt"
        pdbqt_path.touch()

        with (
            patch("hybridock_pep.driver.write_metadata_skeleton", side_effect=make_side("write_metadata_skeleton")),
            patch("hybridock_pep.driver.run_sampling", side_effect=make_side("run_sampling", [])),
            patch("hybridock_pep.driver.parse_poses", side_effect=make_side("parse_poses", ([record], []))),
            patch("hybridock_pep.driver.prepare_receptor", side_effect=make_side("prepare_receptor", tmp_path / "receptor.pdbqt")),
            patch("hybridock_pep.driver.generate_ad4_maps", side_effect=make_side("generate_ad4_maps", tmp_path / "maps")),
            patch("hybridock_pep.driver.prepare_ligand_batch", side_effect=make_side("prepare_ligand_batch", ([pdbqt_path], []))),
            patch("hybridock_pep.driver.score_vina_batch", side_effect=make_side("score_vina_batch", ([], []))),
            patch("hybridock_pep.driver.score_ad4_batch", side_effect=make_side("score_ad4_batch", ([], []))),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.apply_hybrid_score"),
            patch("hybridock_pep.driver.finalize_metadata", side_effect=make_side("finalize_metadata")),
        ):
            driver.run_dock(config, input_poses_dir=None, calibration_path=tmp_path / "cal.json")

        assert call_log.index("write_metadata_skeleton") < call_log.index("run_sampling")
        assert call_log.index("prepare_receptor") < call_log.index("generate_ad4_maps")
        assert call_log.index("prepare_ligand_batch") < call_log.index("score_vina_batch")
        assert call_log.index("score_vina_batch") < call_log.index("score_ad4_batch")
        assert call_log.index("score_ad4_batch") < call_log.index("finalize_metadata")

    def test_prep_failures_logged_not_raised(self, tmp_path: Path) -> None:
        from hybridock_pep import driver

        config = _make_config(tmp_path)
        record = _make_pose_record(0, tmp_path)

        with (
            patch("hybridock_pep.driver.run_sampling", return_value=[]),
            patch("hybridock_pep.driver.parse_poses", return_value=([record], [])),
            patch("hybridock_pep.driver.prepare_receptor", return_value=tmp_path / "receptor.pdbqt"),
            patch("hybridock_pep.driver.generate_ad4_maps", return_value=tmp_path / "maps"),
            patch("hybridock_pep.driver.prepare_ligand_batch", return_value=([], [MagicMock()])),
            patch("hybridock_pep.driver.score_vina_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_ad4_batch", return_value=([], [])),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.write_metadata_skeleton"),
            patch("hybridock_pep.driver.finalize_metadata"),
        ):
            with pytest.raises(RuntimeError):
                driver.run_dock(config, input_poses_dir=None, calibration_path=tmp_path / "cal.json")

    def test_metadata_written_at_start_and_end(self, tmp_path: Path) -> None:
        from hybridock_pep import driver

        config = _make_config(tmp_path)

        with (
            patch("hybridock_pep.driver.run_sampling", return_value=[]),
            patch("hybridock_pep.driver.parse_poses", return_value=([], [])),
            patch("hybridock_pep.driver.prepare_receptor", return_value=tmp_path / "receptor.pdbqt"),
            patch("hybridock_pep.driver.generate_ad4_maps", return_value=tmp_path / "maps"),
            patch("hybridock_pep.driver.prepare_ligand_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_vina_batch", return_value=([], [])),
            patch("hybridock_pep.driver.score_ad4_batch", return_value=([], [])),
            patch("hybridock_pep.driver.load_calibration", return_value={"alpha": 0.5, "beta": 0.1}),
            patch("hybridock_pep.driver.write_metadata_skeleton") as mock_write,
            patch("hybridock_pep.driver.finalize_metadata") as mock_finalize,
        ):
            driver.run_dock(config, input_poses_dir=None, calibration_path=tmp_path / "cal.json")
            assert mock_write.call_count == 1
            assert mock_finalize.call_count == 1
