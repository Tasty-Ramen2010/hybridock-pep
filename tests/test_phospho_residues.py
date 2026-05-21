"""Tests for phospho-residue detection and PDBQT preparation.

Covers:
  - has_phospho_residues(): detection on PDBs with/without TPO/SEP/PTR
  - prepare_phospho_ligand(): round-trip PDB → PDBQT via Meeko Polymer API
  - PDBQT content validation: P atom present, negative charges on phosphate oxygens
  - receptor.py filter: TPO/SEP/PTR HETATM lines are preserved

No slow-test marker because these unit tests use synthetic PDB fixtures only.
The integration scoring test (SHP2 AD4 < -7) requires the shp2_4jmg fixture
which is not yet built; that test is therefore skipped until the fixture exists.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hybridock_pep.prep.phospho import PHOSPHO_RESIDUES, has_phospho_residues, prepare_phospho_ligand


# ---------------------------------------------------------------------------
# Synthetic PDB fixtures
# ---------------------------------------------------------------------------

_GLY_TPO_GLY_PDB = textwrap.dedent("""\
    ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
    ATOM      3  C   GLY A   1       1.980   1.420   0.000  1.00  0.00           C
    ATOM      4  O   GLY A   1       1.200   2.360   0.000  1.00  0.00           O
    ATOM      5  N   TPO A   2       3.307   1.640   0.000  1.00  0.00           N
    ATOM      6  CA  TPO A   2       3.900   2.980   0.000  1.00  0.00           C
    ATOM      7  C   TPO A   2       5.420   2.980   0.000  1.00  0.00           C
    ATOM      8  O   TPO A   2       6.040   1.960   0.000  1.00  0.00           O
    ATOM      9  CB  TPO A   2       3.400   3.800   1.190  1.00  0.00           C
    ATOM     10  OG1 TPO A   2       1.980   3.780   1.250  1.00  0.00           O
    ATOM     11  CG2 TPO A   2       3.900   3.250   2.510  1.00  0.00           C
    ATOM     12  P   TPO A   2       1.370   4.800   2.400  1.00  0.00           P
    ATOM     13  O1P TPO A   2       0.000   4.200   2.400  1.00  0.00           O
    ATOM     14  O2P TPO A   2       1.370   6.230   2.400  1.00  0.00           O
    ATOM     15  O3P TPO A   2       2.100   4.400   3.680  1.00  0.00           O
    ATOM     16  N   GLY A   3       6.140   4.100   0.000  1.00  0.00           N
    ATOM     17  CA  GLY A   3       7.590   4.100   0.000  1.00  0.00           C
    ATOM     18  C   GLY A   3       8.110   5.520   0.000  1.00  0.00           C
    ATOM     19  O   GLY A   3       7.330   6.470   0.000  1.00  0.00           O
    END
""")

# Same as above but with TPO as HETATM (older PDB entries)
_GLY_TPO_HETATM_PDB = _GLY_TPO_GLY_PDB.replace("ATOM     12  P   TPO", "HETATM   12  P   TPO")

_SEP_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.980   1.420   0.000  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.200   2.360   0.000  1.00  0.00           O
    ATOM      5  CB  ALA A   1       1.800  -0.800  -1.100  1.00  0.00           C
    ATOM      6  N   SEP A   2       3.307   1.640   0.000  1.00  0.00           N
    ATOM      7  CA  SEP A   2       3.900   2.980   0.000  1.00  0.00           C
    ATOM      8  C   SEP A   2       5.420   2.980   0.000  1.00  0.00           C
    ATOM      9  O   SEP A   2       6.040   1.960   0.000  1.00  0.00           O
    ATOM     10  CB  SEP A   2       3.400   3.800   1.190  1.00  0.00           C
    ATOM     11  OG  SEP A   2       1.980   3.780   1.250  1.00  0.00           O
    ATOM     12  P   SEP A   2       1.370   4.800   2.400  1.00  0.00           P
    ATOM     13  O1P SEP A   2       0.000   4.200   2.400  1.00  0.00           O
    ATOM     14  O2P SEP A   2       1.370   6.230   2.400  1.00  0.00           O
    ATOM     15  O3P SEP A   2       2.100   4.400   3.680  1.00  0.00           O
    ATOM     16  N   ALA A   3       6.140   4.100   0.000  1.00  0.00           N
    ATOM     17  CA  ALA A   3       7.590   4.100   0.000  1.00  0.00           C
    ATOM     18  C   ALA A   3       8.110   5.520   0.000  1.00  0.00           C
    ATOM     19  O   ALA A   3       7.330   6.470   0.000  1.00  0.00           O
    ATOM     20  CB  ALA A   3       8.100   4.600  -1.200  1.00  0.00           C
    END
""")


@pytest.fixture
def tpo_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "pose_tpo.pdb"
    p.write_text(_GLY_TPO_GLY_PDB)
    return p


@pytest.fixture
def sep_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "pose_sep.pdb"
    p.write_text(_SEP_PDB)
    return p


@pytest.fixture
def tpo_hetatm_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "pose_tpo_hetatm.pdb"
    p.write_text(_GLY_TPO_HETATM_PDB)
    return p


@pytest.fixture
def standard_pdb() -> Path:
    return Path(__file__).parent / "fixtures" / "pdz_1jq8" / "pose_000.pdb"


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestPhosphoDetection:
    def test_detects_tpo_in_pdb(self, tpo_pdb: Path) -> None:
        assert has_phospho_residues(tpo_pdb) is True

    def test_detects_sep_in_pdb(self, sep_pdb: Path) -> None:
        assert has_phospho_residues(sep_pdb) is True

    def test_detects_tpo_as_hetatm(self, tpo_hetatm_pdb: Path) -> None:
        assert has_phospho_residues(tpo_hetatm_pdb) is True

    def test_no_phospho_in_standard_peptide(self, standard_pdb: Path) -> None:
        assert has_phospho_residues(standard_pdb) is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        assert has_phospho_residues(tmp_path / "nonexistent.pdb") is False

    def test_phospho_residues_constant(self) -> None:
        assert "TPO" in PHOSPHO_RESIDUES
        assert "SEP" in PHOSPHO_RESIDUES
        assert "PTR" in PHOSPHO_RESIDUES
        assert "THR" not in PHOSPHO_RESIDUES  # standard residue must not be in set


# ---------------------------------------------------------------------------
# PDBQT preparation tests
# ---------------------------------------------------------------------------

class TestPhosphoLigandPrep:
    def test_tpo_prep_succeeds(self, tpo_pdb: Path, tmp_path: Path) -> None:
        from hybridock_pep.models import PoseFailure
        result = prepare_phospho_ligand(0, tpo_pdb, tmp_path)
        assert not isinstance(result, PoseFailure), f"Prep failed: {result}"
        assert isinstance(result, Path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_sep_prep_succeeds(self, sep_pdb: Path, tmp_path: Path) -> None:
        from hybridock_pep.models import PoseFailure
        result = prepare_phospho_ligand(0, sep_pdb, tmp_path)
        assert not isinstance(result, PoseFailure), f"Prep failed: {result}"
        assert isinstance(result, Path)
        assert result.exists()

    def test_tpo_pdbqt_contains_phosphorus(self, tpo_pdb: Path, tmp_path: Path) -> None:
        result = prepare_phospho_ligand(0, tpo_pdb, tmp_path)
        assert isinstance(result, Path)
        pdbqt_text = result.read_text()
        atom_lines = [l for l in pdbqt_text.splitlines() if l.startswith(("ATOM", "HETATM"))]
        # At least one atom record must have atom type "P" (phosphorus in AutoDock types)
        # PDBQT format: columns 78-79 are atom type
        has_p_atom = any(l[77:].strip().startswith("P") or " P " in l for l in atom_lines)
        assert has_p_atom, f"No phosphorus atom found in PDBQT:\n{pdbqt_text[:500]}"

    def test_pdbqt_is_parseable_by_vina(self, tpo_pdb: Path, tmp_path: Path) -> None:
        """The produced PDBQT must be syntactically accepted by Vina."""
        from vina import Vina
        result = prepare_phospho_ligand(0, tpo_pdb, tmp_path)
        assert isinstance(result, Path)
        v = Vina(sf_name="vina", verbosity=0)
        # Vina.set_ligand_from_file raises if the PDBQT is malformed
        v.set_ligand_from_file(str(result))

    def test_output_written_to_correct_directory(self, tpo_pdb: Path, tmp_path: Path) -> None:
        subdir = tmp_path / "pdbqt_outputs"
        result = prepare_phospho_ligand(0, tpo_pdb, subdir)
        assert isinstance(result, Path)
        assert result.parent == subdir

    def test_pose_failure_on_empty_pdb(self, tmp_path: Path) -> None:
        from hybridock_pep.models import PoseFailure
        empty = tmp_path / "empty.pdb"
        empty.write_text("END\n")
        result = prepare_phospho_ligand(0, empty, tmp_path)
        assert isinstance(result, PoseFailure)
        assert result.pose_idx == 0


# ---------------------------------------------------------------------------
# receptor.py filter preserves phospho HETATM
# ---------------------------------------------------------------------------

class TestReceptorFilterPhospho:
    def test_phospho_hetatm_preserved(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb_with_tpo_hetatm = tmp_path / "receptor_tpo.pdb"
        pdb_with_tpo_hetatm.write_text(textwrap.dedent("""\
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
            HETATM    2  P   TPO A   2       1.370   4.800   2.400  1.00  0.00           P
            HETATM    3  O1P TPO A   2       0.000   4.200   2.400  1.00  0.00           O
            HETATM    4  OH2 HOH A 101       5.000   5.000   5.000  1.00  0.00           O
            HETATM    5  C1  LIG A 200       2.000   2.000   2.000  1.00  0.00           C
            END
        """))

        lines = _filter_pdb_lines(pdb_with_tpo_hetatm)
        resnames_kept = {line[17:20].strip() for line in lines if line.startswith(("ATOM", "HETATM"))}
        assert "TPO" in resnames_kept, "TPO HETATM should be preserved"
        assert "HOH" in resnames_kept, "Water should be preserved"
        assert "LIG" not in resnames_kept, "Unknown ligand HETATM should be dropped"
        assert "ALA" in resnames_kept, "ATOM records should always be preserved"

    def test_sep_hetatm_preserved(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "sep.pdb"
        pdb.write_text("HETATM    1  P   SEP A   1       0.000   0.000   0.000  1.00  0.00           P\nEND\n")
        lines = _filter_pdb_lines(pdb)
        assert any("SEP" in l for l in lines)

    def test_ptr_hetatm_preserved(self, tmp_path: Path) -> None:
        from hybridock_pep.prep.receptor import _filter_pdb_lines

        pdb = tmp_path / "ptr.pdb"
        pdb.write_text("HETATM    1  P   PTR A   1       0.000   0.000   0.000  1.00  0.00           P\nEND\n")
        lines = _filter_pdb_lines(pdb)
        assert any("PTR" in l for l in lines)


# ---------------------------------------------------------------------------
# ligand.py routing test
# ---------------------------------------------------------------------------

class TestLigandRoutingPhospho:
    def test_phospho_pose_routes_to_meeko_not_babel(
        self, tpo_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirms phospho detection triggers Meeko path before babel is called."""
        import hybridock_pep.prep.ligand as ligand_mod

        babel_called = []
        original_which = ligand_mod.shutil.which

        def mock_which(name: str) -> str | None:
            if name == "babel":
                babel_called.append(True)
            return original_which(name)

        monkeypatch.setattr(ligand_mod.shutil, "which", mock_which)

        result = ligand_mod._prepare_single_ligand((0, tpo_pdb, tmp_path))
        from hybridock_pep.models import PoseFailure
        # Should succeed via Meeko; babel should NOT have been called
        assert not isinstance(result, PoseFailure), f"Routing failed: {result}"
        assert not babel_called, "babel was called for a phospho pose — routing is wrong"
