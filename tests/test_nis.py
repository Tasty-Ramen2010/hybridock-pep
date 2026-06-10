"""Tests for the NIS (non-interacting surface) composition module.

NIS is the within-target RELATIVE affinity signal (scoring/nis.py). Imports are
lazy per the test-file convention in test_scoring.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _atom(serial: int, name: str, resname: str, resseq: int, xyz: tuple[float, float, float],
          chain: str = "A", element: str = "C") -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n"
    )


def _write_pdb(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(lines) + "END\n")
    return path


class TestComputeNisFeatures:
    """compute_nis_features() on controlled synthetic geometry."""

    def _make(self, tmp_path: Path) -> tuple[Path, Path]:
        # Receptor: one heavy atom at the origin.
        rec = _write_pdb(tmp_path / "rec.pdb",
                         [_atom(1, "CA", "GLY", 1, (0.0, 0.0, 0.0))])
        # Peptide: 4 residues. Res1 contacts the receptor (3 Å); res2-4 are NIS.
        # NIS composition: SER (polar), LYS (charged), ALA (apolar) -> 1/3, 1/3.
        pep = _write_pdb(tmp_path / "pep.pdb", [
            _atom(1, "CA", "LEU", 1, (3.0, 0.0, 0.0)),    # contacting (apolar, ignored)
            _atom(2, "CA", "SER", 2, (50.0, 0.0, 0.0)),   # NIS polar
            _atom(3, "CA", "LYS", 3, (60.0, 0.0, 0.0)),   # NIS charged
            _atom(4, "CA", "ALA", 4, (70.0, 0.0, 0.0)),   # NIS apolar
        ])
        return pep, rec

    def test_fractions(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.nis import compute_nis_features

        pep, rec = self._make(tmp_path)
        polar, charged = compute_nis_features(pep, rec)
        assert polar == pytest.approx(1 / 3)
        assert charged == pytest.approx(1 / 3)

    def test_fully_buried_returns_zero(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.nis import compute_nis_features

        rec = _write_pdb(tmp_path / "rec.pdb", [_atom(1, "CA", "GLY", 1, (0.0, 0.0, 0.0))])
        # single residue, contacting -> no NIS residues
        pep = _write_pdb(tmp_path / "pep.pdb", [_atom(1, "CA", "SER", 1, (3.0, 0.0, 0.0))])
        assert compute_nis_features(pep, rec) == (0.0, 0.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.nis import compute_nis_features

        rec = _write_pdb(tmp_path / "rec.pdb", [_atom(1, "CA", "GLY", 1, (0.0, 0.0, 0.0))])
        with pytest.raises(FileNotFoundError):
            compute_nis_features(tmp_path / "nope.pdb", rec)


class TestRelativeNisRanking:
    """relative_nis_ranking() z-normalisation + orientation."""

    def test_orientation_lower_is_stronger(self) -> None:
        from hybridock_pep.scoring.nis import relative_nis_ranking

        # candidate A has lower raw nis_score (more polar NIS) -> stronger -> lower z
        z = relative_nis_ranking([-0.5, 0.5])
        assert z[0] < z[1]
        assert z.mean() == pytest.approx(0.0)
        assert z.std() == pytest.approx(1.0)

    def test_constant_returns_zeros(self) -> None:
        from hybridock_pep.scoring.nis import relative_nis_ranking

        assert np.allclose(relative_nis_ranking([0.2, 0.2, 0.2]), 0.0)

    def test_needs_two(self) -> None:
        from hybridock_pep.scoring.nis import relative_nis_ranking

        with pytest.raises(ValueError):
            relative_nis_ranking([0.1])
