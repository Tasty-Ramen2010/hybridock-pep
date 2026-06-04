"""Tests for off-pocket pose filter + auto-box cap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridock_pep.driver import (
    _AUTO_BOX_MAX_AA, _OFFPOCKET_CENTROID_AA,
    _auto_expand_box_for_poses, _filter_offpocket_poses,
)
from hybridock_pep.models import DockConfig, PoseRecord


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    return p


def _pose_at(centroid_xyz: tuple[float, float, float], idx: int,
             tmp_path: Path, n_res: int = 5) -> PoseRecord:
    """Build a PoseRecord whose Cα coords are clustered near centroid_xyz."""
    rng = np.random.default_rng(idx)
    ca = np.tile(centroid_xyz, (n_res, 1)) + rng.normal(0, 0.5, (n_res, 3))
    pdb_path = tmp_path / f"pose_{idx}.pdb"
    lines = []
    for i, (x, y, z) in enumerate(ca):
        lines.append(
            f"ATOM  {i+1:>5d}  CA  ALA A{i+1:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n"
        )
    pdb_path.write_text("".join(lines) + "END\n")
    return PoseRecord(
        pose_idx=idx, pdb_path=pdb_path, sequence="A" * n_res, ca_coords=ca,
    )


class TestOffpocketFilter:
    def test_keeps_in_pocket_poses(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="A" * 5, receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=30.0, output_dir=tmp_path,
        )
        # All 3 poses within 10 Å of site
        recs = [_pose_at((5, 0, 0), 0, tmp_path),
                _pose_at((-3, 4, 0), 1, tmp_path),
                _pose_at((2, -2, 5), 2, tmp_path)]
        out = _filter_offpocket_poses(cfg, recs)
        assert len(out) == 3

    def test_drops_far_poses(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="A" * 5, receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=30.0, output_dir=tmp_path,
        )
        # 2 in-pocket, 2 way off (50 Å on x axis)
        recs = [_pose_at((5, 0, 0), 0, tmp_path),
                _pose_at((50, 0, 0), 1, tmp_path),
                _pose_at((-3, 4, 0), 2, tmp_path),
                _pose_at((0, 60, 0), 3, tmp_path)]
        out = _filter_offpocket_poses(cfg, recs)
        kept_idx = {r.pose_idx for r in out}
        assert kept_idx == {0, 2}

    def test_keeps_borderline_poses_just_inside_threshold(
        self, tmp_path: Path, valid_receptor: Path
    ) -> None:
        cfg = DockConfig(
            peptide_sequence="A" * 5, receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=30.0, output_dir=tmp_path,
        )
        # Centroid at exactly the threshold should be kept (<=, not <)
        recs = [_pose_at((_OFFPOCKET_CENTROID_AA - 0.5, 0, 0), 0, tmp_path)]
        out = _filter_offpocket_poses(cfg, recs)
        assert len(out) == 1

    def test_empty_input_returns_empty(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        assert _filter_offpocket_poses(cfg, []) == []


class TestAutoBoxCap:
    def test_caps_at_max_when_expansion_would_exceed(
        self, tmp_path: Path, valid_receptor: Path
    ) -> None:
        cfg = DockConfig(
            peptide_sequence="A" * 5, receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        # Write a pose with an atom 80 Å away → would need 168 Å box → capped
        path = tmp_path / "pose_huge.pdb"
        path.write_text(
            "ATOM      1  CA  ALA A   1      80.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        rec = PoseRecord(pose_idx=0, pdb_path=path, sequence="A",
                         ca_coords=np.array([[80.0, 0.0, 0.0]]))
        out = _auto_expand_box_for_poses(cfg, [rec])
        # Must be capped at _AUTO_BOX_MAX_AA, not 168 Å
        assert out.box_size == _AUTO_BOX_MAX_AA

    def test_expands_normally_below_cap(
        self, tmp_path: Path, valid_receptor: Path
    ) -> None:
        cfg = DockConfig(
            peptide_sequence="A" * 5, receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        )
        path = tmp_path / "pose.pdb"
        path.write_text(
            "ATOM      1  CA  ALA A   1      25.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        rec = PoseRecord(pose_idx=0, pdb_path=path, sequence="A",
                         ca_coords=np.array([[25.0, 0.0, 0.0]]))
        out = _auto_expand_box_for_poses(cfg, [rec])
        # Needed ~58 Å (well below the 100 Å cap), should expand normally
        assert 56.0 < out.box_size < 60.0
