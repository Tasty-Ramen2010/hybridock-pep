"""Tests for hybridock_pep.prep — PREP-01, PREP-02, PREP-03."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Task 1: PrepError + fixtures
# ---------------------------------------------------------------------------


class TestPrepError:
    def test_importable(self) -> None:
        from hybridock_pep.prep import PrepError  # noqa: F401

    def test_is_runtime_error(self) -> None:
        from hybridock_pep.prep import PrepError

        err = PrepError("something went wrong")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self) -> None:
        from hybridock_pep.prep import PrepError

        err = PrepError("bad news")
        assert str(err) == "bad news"


class TestFixtures:
    def test_receptor_tiny_exists(self) -> None:
        p = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
        assert p.exists(), f"Missing fixture: {p}"

    def test_receptor_tiny_has_ca_ala_a1(self) -> None:
        p = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
        content = p.read_text()
        assert "CA  ALA A   1" in content

    def test_pose_tiny_exists(self) -> None:
        p = Path(__file__).parent / "fixtures" / "pose_tiny.pdb"
        assert p.exists(), f"Missing fixture: {p}"

    def test_pose_tiny_has_three_residues(self) -> None:
        p = Path(__file__).parent / "fixtures" / "pose_tiny.pdb"
        content = p.read_text()
        assert "ALA A   1" in content
        assert "ALA A   2" in content
        assert "ALA A   3" in content


# ---------------------------------------------------------------------------
# Task 2: prep/receptor.py — prepare_receptor + _filter_pdb_lines
# ---------------------------------------------------------------------------


class TestFilterPdbLines:
    """Unit tests for _filter_pdb_lines — no external tools needed."""

    def test_passthrough_atom_records(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            "END\n"
        )
        lines = _filter_pdb_lines(pdb)
        assert any("ATOM" in line for line in lines)

    def test_drops_non_water_hetatm(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            "HETATM    2  C1  LIG A 100       4.000   5.000   6.000  1.00  0.00           C\n"
        )
        lines = _filter_pdb_lines(pdb)
        assert not any("LIG" in line for line in lines)

    def test_keeps_water_hetatm(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "HETATM    1  O   HOH A 200       7.000   8.000   9.000  1.00  0.00           O\n"
        )
        lines = _filter_pdb_lines(pdb)
        assert any("HOH" in line for line in lines)

    def test_drops_alternate_occupancy_b(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "test.pdb"
        # altLoc 'B' at column index 16
        pdb.write_text(
            "ATOM      1  CA BALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        )
        lines = _filter_pdb_lines(pdb)
        assert not any("BALA" in line for line in lines)

    def test_keeps_altloc_a(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA AALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        )
        lines = _filter_pdb_lines(pdb)
        assert any("ATOM" in line for line in lines)


class TestPrepareReceptor:
    """Unit tests for prepare_receptor — subprocess mocked."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        """Build a minimal DockConfig with the tiny receptor fixture."""
        from hybridock_pep.models import DockConfig

        receptor = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
        return DockConfig(
            peptide_sequence="ALA",
            receptor_path=receptor,
            site_coords=(1.0, 2.0, 3.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    def test_returns_pdbqt_path(self, config, tmp_path: Path) -> None:
        """prepare_receptor returns output_dir/receptor.pdbqt on success."""
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        # pdbfixer + openmm need mocking too
        with (
            patch("hybridock_pep.prep.receptor.PDBFixer") as mock_fixer_cls,
            patch("hybridock_pep.prep.receptor.PDBFile") as mock_pdbfile,
            patch("hybridock_pep.prep.receptor.subprocess.run", return_value=mock_result),
            patch("hybridock_pep.prep.receptor.tempfile.NamedTemporaryFile") as mock_ntf,
        ):
            # Set up temp file mock to produce real paths
            tmp_file1 = tmp_path / "tmp1.pdb"
            tmp_file1.write_text("")
            tmp_file2 = tmp_path / "tmp2.pdb"
            tmp_file2.write_text("")

            ntf_ctx1 = MagicMock()
            ntf_ctx1.__enter__ = MagicMock(return_value=ntf_ctx1)
            ntf_ctx1.__exit__ = MagicMock(return_value=False)
            ntf_ctx1.name = str(tmp_file1)

            ntf_ctx2 = MagicMock()
            ntf_ctx2.__enter__ = MagicMock(return_value=ntf_ctx2)
            ntf_ctx2.__exit__ = MagicMock(return_value=False)
            ntf_ctx2.name = str(tmp_file2)

            mock_ntf.side_effect = [ntf_ctx1, ntf_ctx2]

            mock_fixer = MagicMock()
            mock_fixer_cls.return_value = mock_fixer
            mock_fixer.topology = MagicMock()
            mock_fixer.positions = MagicMock()

            result = prepare_receptor(config)
            assert result == config.output_dir / "receptor.pdbqt"

    def test_non_zero_exit_raises_prep_error(self, config, tmp_path: Path) -> None:
        """Non-zero exit from prepare_receptor4.py raises PrepError."""
        from hybridock_pep.prep import PrepError
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "prepare_receptor4.py: error in receptor"

        with (
            patch("hybridock_pep.prep.receptor.PDBFixer") as mock_fixer_cls,
            patch("hybridock_pep.prep.receptor.PDBFile"),
            patch("hybridock_pep.prep.receptor.subprocess.run", return_value=mock_result),
            patch("hybridock_pep.prep.receptor.tempfile.NamedTemporaryFile") as mock_ntf,
        ):
            tmp_file1 = tmp_path / "tmp1.pdb"
            tmp_file1.write_text("")
            tmp_file2 = tmp_path / "tmp2.pdb"
            tmp_file2.write_text("")

            ntf_ctx1 = MagicMock()
            ntf_ctx1.__enter__ = MagicMock(return_value=ntf_ctx1)
            ntf_ctx1.__exit__ = MagicMock(return_value=False)
            ntf_ctx1.name = str(tmp_file1)

            ntf_ctx2 = MagicMock()
            ntf_ctx2.__enter__ = MagicMock(return_value=ntf_ctx2)
            ntf_ctx2.__exit__ = MagicMock(return_value=False)
            ntf_ctx2.name = str(tmp_file2)

            mock_ntf.side_effect = [ntf_ctx1, ntf_ctx2]

            mock_fixer = MagicMock()
            mock_fixer_cls.return_value = mock_fixer

            with pytest.raises(PrepError, match="prepare_receptor4.py failed"):
                prepare_receptor(config)

    def test_no_caching_guard(self) -> None:
        """receptor.py source must not contain a skip-if-exists guard (D-02)."""
        source = Path(__file__).parent.parent / "src" / "hybridock_pep" / "prep" / "receptor.py"
        assert source.exists(), "receptor.py not found"
        content = source.read_text()
        assert "pdbqt_path.exists()" not in content, "Caching guard found — violates D-02"

    def test_pdbfixer_called_with_three_steps(self, config, tmp_path: Path) -> None:
        """pdbfixer must call findMissingResidues, findMissingAtoms, addMissingHydrogens."""
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with (
            patch("hybridock_pep.prep.receptor.PDBFixer") as mock_fixer_cls,
            patch("hybridock_pep.prep.receptor.PDBFile"),
            patch("hybridock_pep.prep.receptor.subprocess.run", return_value=mock_result),
            patch("hybridock_pep.prep.receptor.tempfile.NamedTemporaryFile") as mock_ntf,
        ):
            tmp_file1 = tmp_path / "tmp1.pdb"
            tmp_file1.write_text("")
            tmp_file2 = tmp_path / "tmp2.pdb"
            tmp_file2.write_text("")

            ntf_ctx1 = MagicMock()
            ntf_ctx1.__enter__ = MagicMock(return_value=ntf_ctx1)
            ntf_ctx1.__exit__ = MagicMock(return_value=False)
            ntf_ctx1.name = str(tmp_file1)

            ntf_ctx2 = MagicMock()
            ntf_ctx2.__enter__ = MagicMock(return_value=ntf_ctx2)
            ntf_ctx2.__exit__ = MagicMock(return_value=False)
            ntf_ctx2.name = str(tmp_file2)

            mock_ntf.side_effect = [ntf_ctx1, ntf_ctx2]

            mock_fixer = MagicMock()
            mock_fixer_cls.return_value = mock_fixer

            prepare_receptor(config)

            mock_fixer.findMissingResidues.assert_called_once()
            mock_fixer.findMissingAtoms.assert_called_once()
            mock_fixer.addMissingHydrogens.assert_called_once_with(7.4)
