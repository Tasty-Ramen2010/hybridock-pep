"""Tests for crop_to_pocket() and _auto_expand_box_for_poses()."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridock_pep.driver import _auto_expand_box_for_poses
from hybridock_pep.models import DockConfig, PoseRecord
from hybridock_pep.prep.receptor import crop_to_pocket


def _make_pdb(lines: list[str], path: Path) -> Path:
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _atom_line(serial: int, name: str, resn: str, chain: str, resseq: int,
               x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:>5d}  {name:<4s}{resn:<3s} {chain}{resseq:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {name[0]}"
    )


class TestCropToPocket:
    def test_keeps_only_residues_within_radius(self, tmp_path: Path) -> None:
        # 4 residues: two near (0,0,0), two far away
        lines = [
            # residue 1 — Cα at (5,0,0), inside 10 Å
            _atom_line(1, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0),
            _atom_line(2, "C",  "ALA", "A", 1, 6.0, 0.0, 0.0),
            # residue 2 — Cα at (-3,4,0), inside 10 Å
            _atom_line(3, "CA", "GLY", "A", 2, -3.0, 4.0, 0.0),
            # residue 3 — Cα at (50,0,0), far
            _atom_line(4, "CA", "VAL", "A", 3, 50.0, 0.0, 0.0),
            # residue 4 — Cα at (0,40,0), far
            _atom_line(5, "CA", "LEU", "A", 4, 0.0, 40.0, 0.0),
        ]
        src = _make_pdb(lines, tmp_path / "in.pdb")
        out = tmp_path / "out.pdb"
        kept = crop_to_pocket(src, (0.0, 0.0, 0.0), radius=10.0, output_path=out)
        assert kept == 2

        out_text = out.read_text()
        # residues 1 + 2 in, 3 + 4 out
        assert " ALA A   1" in out_text or "ALA A   1" in out_text
        assert "GLY A   2" in out_text or " GLY A   2" in out_text
        assert " VAL A   3" not in out_text
        assert " LEU A   4" not in out_text

    def test_residue_atomically_intact(self, tmp_path: Path) -> None:
        """If any atom of a residue is in range, ALL atoms of that residue
        are written (not just the in-range ones) — preserves connectivity."""
        lines = [
            # residue 1: CA inside (5 Å), C far away (50 Å). Both should be kept.
            _atom_line(1, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0),
            _atom_line(2, "C",  "ALA", "A", 1, 50.0, 0.0, 0.0),
        ]
        src = _make_pdb(lines, tmp_path / "in.pdb")
        out = tmp_path / "out.pdb"
        kept = crop_to_pocket(src, (0.0, 0.0, 0.0), radius=10.0, output_path=out)
        assert kept == 1
        # Both atom lines should appear in the output
        text = out.read_text()
        assert "  CA" in text
        assert "  C " in text

    def test_writes_remark_header(self, tmp_path: Path) -> None:
        lines = [_atom_line(1, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0)]
        src = _make_pdb(lines, tmp_path / "in.pdb")
        out = tmp_path / "out.pdb"
        crop_to_pocket(src, (0.0, 0.0, 0.0), radius=10.0, output_path=out)
        assert "REMARK   Pocket crop" in out.read_text()

    def test_hydrogen_atoms_ignored_for_distance_check(self, tmp_path: Path) -> None:
        # Only hydrogens are in range; their residue should not be kept
        lines = [_atom_line(1, "H", "ALA", "A", 1, 5.0, 0.0, 0.0)]
        src = _make_pdb(lines, tmp_path / "in.pdb")
        out = tmp_path / "out.pdb"
        kept = crop_to_pocket(src, (0.0, 0.0, 0.0), radius=10.0, output_path=out)
        assert kept == 0


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    return p


def _pose_record(coords: list[list[float]], pose_idx: int, tmp_path: Path) -> PoseRecord:
    """Write a tiny pose PDB with the given heavy-atom coords and return a record."""
    lines = []
    for i, (x, y, z) in enumerate(coords):
        lines.append(_atom_line(i+1, "CA", "ALA", "A", i+1, x, y, z))
    path = tmp_path / f"pose_{pose_idx}.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return PoseRecord(
        pose_idx=pose_idx, pdb_path=path,
        sequence="A" * len(coords), ca_coords=np.array(coords),
    )


class TestAutoExpandBox:
    def test_no_expansion_when_poses_fit(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=40.0,
            output_dir=tmp_path / "out",
        )
        # All poses within 10 Å of site
        recs = [
            _pose_record([[5.0, 0.0, 0.0], [3.0, 4.0, 0.0]], 0, tmp_path),
            _pose_record([[-8.0, 2.0, -1.0]], 1, tmp_path),
        ]
        out = _auto_expand_box_for_poses(cfg, recs)
        assert out.box_size == 40.0  # unchanged
        assert out is cfg  # exact identity, no copy

    def test_expands_when_pose_overflows(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0,
            output_dir=tmp_path / "out",
        )
        # A pose with an atom 30 Å away on x axis — user box (half=10) is too small
        recs = [_pose_record([[30.0, 0.0, 0.0]], 0, tmp_path)]
        out = _auto_expand_box_for_poses(cfg, recs, safety_margin=4.0)
        # New half-edge must cover 30 Å + 4 Å safety = 34; new edge ≥ 68
        assert out.box_size >= 68.0
        # site_coords unchanged
        assert out.site_coords == cfg.site_coords
        # Copy, not original
        assert out is not cfg

    def test_empty_records_returns_original(self, tmp_path: Path, valid_receptor: Path) -> None:
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=20.0,
            output_dir=tmp_path / "out",
        )
        out = _auto_expand_box_for_poses(cfg, [])
        assert out is cfg

    def test_ignores_hydrogens(self, tmp_path: Path, valid_receptor: Path) -> None:
        """A hydrogen 50 Å away shouldn't trigger expansion (Vina ignores H anyway)."""
        cfg = DockConfig(
            peptide_sequence="LIS", receptor_path=valid_receptor,
            site_coords=(0.0, 0.0, 0.0), box_size=40.0,
            output_dir=tmp_path / "out",
        )
        # Build a pose where the only far atom is a hydrogen
        path = tmp_path / "pose_h.pdb"
        path.write_text(
            _atom_line(1, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0) + "\n"
            + _atom_line(2, "H",  "ALA", "A", 1, 50.0, 0.0, 0.0) + "\n"
            + "END\n"
        )
        rec = PoseRecord(pose_idx=0, pdb_path=path,
                         sequence="A", ca_coords=np.array([[5.0, 0.0, 0.0]]))
        out = _auto_expand_box_for_poses(cfg, [rec])
        assert out.box_size == 40.0
