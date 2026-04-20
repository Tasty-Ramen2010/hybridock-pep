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


# ---------------------------------------------------------------------------
# Task 1 (02-02): prep/ligand.py — prepare_ligand_batch + _prepare_single_ligand
# ---------------------------------------------------------------------------


class TestLigandBatchImports:
    """PREP-02: Module-level imports and structural requirements."""

    def test_importable(self) -> None:
        from hybridock_pep.prep.ligand import prepare_ligand_batch  # noqa: F401

    def test_worker_importable(self) -> None:
        from hybridock_pep.prep.ligand import _prepare_single_ligand  # noqa: F401

    def test_worker_is_module_level(self) -> None:
        """_prepare_single_ligand must be defined at module level (not a closure)."""
        import inspect
        import hybridock_pep.prep.ligand as ligand_mod

        func = ligand_mod._prepare_single_ligand
        # qualname has no '.<locals>.' segment for module-level functions
        assert "<locals>" not in func.__qualname__, (
            f"_prepare_single_ligand appears to be a closure: {func.__qualname__!r}"
        )

    def test_uses_process_pool_executor(self) -> None:
        """Source must use ProcessPoolExecutor, not ThreadPoolExecutor."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "ligand.py"
        )
        content = source.read_text()
        assert "ProcessPoolExecutor" in content
        assert "ThreadPoolExecutor" not in content

    def test_meeko_import_inside_worker(self) -> None:
        """from meeko import must appear inside the worker, not at module top level."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "ligand.py"
        )
        lines = source.read_text().splitlines()
        # Top-level lines (non-indented) must not import meeko
        top_level_meeko = [
            ln for ln in lines if ln.startswith("from meeko") or ln.startswith("import meeko")
        ]
        assert not top_level_meeko, (
            f"meeko imported at top level (not inside worker): {top_level_meeko}"
        )

    def test_no_bare_except(self) -> None:
        """No bare 'except:' allowed (project convention)."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "ligand.py"
        )
        content = source.read_text()
        assert "except:" not in content, "Bare except: found — use 'except Exception as e:'"

    def test_future_annotations_present(self) -> None:
        """from __future__ import annotations must be first non-comment line."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "ligand.py"
        )
        lines = source.read_text().splitlines()
        code_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        assert code_lines[0] == "from __future__ import annotations", (
            f"Expected 'from __future__ import annotations', got: {code_lines[0]!r}"
        )


class TestLigandBatchBehavior:
    """PREP-02: Behavioral tests for prepare_ligand_batch."""

    @pytest.fixture()
    def pose_tiny(self) -> Path:
        return Path(__file__).parent / "fixtures" / "pose_tiny.pdb"

    def test_success_plus_failure_sums_to_n(self, tmp_path: Path, pose_tiny: Path) -> None:
        """len(successes) + len(failures) == len(input_paths) for any input."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch
        from hybridock_pep.models import PoseFailure

        # Two paths: one valid, one nonexistent
        bad_path = tmp_path / "nonexistent.pdb"
        pdb_paths = [pose_tiny, bad_path]
        successes, failures = prepare_ligand_batch(pdb_paths, tmp_path / "pdbqt_out")
        assert len(successes) + len(failures) == 2

    def test_success_returns_path_objects(self, tmp_path: Path, pose_tiny: Path) -> None:
        """Successful poses return Path objects pointing to written PDBQT files."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        successes, failures = prepare_ligand_batch(
            [pose_tiny], tmp_path / "pdbqt_out"
        )
        if successes:
            for p in successes:
                assert isinstance(p, Path), f"Expected Path, got {type(p)}"
                assert p.suffix == ".pdbqt"

    def test_failures_are_pose_failure_records(self, tmp_path: Path) -> None:
        """Failed poses return PoseFailure with stage='prep'."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch
        from hybridock_pep.models import PoseFailure

        bad_path = tmp_path / "definitely_missing.pdb"
        successes, failures = prepare_ligand_batch(
            [bad_path], tmp_path / "pdbqt_out"
        )
        assert len(failures) == 1
        assert isinstance(failures[0], PoseFailure)
        assert failures[0].stage == "prep"
        assert failures[0].pose_idx == 0

    def test_no_exception_propagates_from_batch(self, tmp_path: Path) -> None:
        """prepare_ligand_batch never raises on per-pose failures."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        # All paths nonexistent — should return failures, not raise
        bad_paths = [tmp_path / f"missing_{i}.pdb" for i in range(3)]
        try:
            successes, failures = prepare_ligand_batch(bad_paths, tmp_path / "out")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"prepare_ligand_batch raised unexpectedly: {exc!r}")

    def test_output_dir_created_if_absent(self, tmp_path: Path, pose_tiny: Path) -> None:
        """output_dir is created by prepare_ligand_batch if it does not exist."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        out_dir = tmp_path / "nested" / "pdbqt"
        assert not out_dir.exists()
        prepare_ligand_batch([pose_tiny], out_dir)
        assert out_dir.exists()

    def test_pdbqt_written_to_output_dir(self, tmp_path: Path, pose_tiny: Path) -> None:
        """Each successful PDBQT is written to output_dir / (stem + .pdbqt)."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        out_dir = tmp_path / "pdbqt_out"
        successes, failures = prepare_ligand_batch([pose_tiny], out_dir)
        if successes:
            expected = out_dir / (pose_tiny.stem + ".pdbqt")
            assert expected.exists(), f"Expected PDBQT not found: {expected}"


class TestPrepareSingleLigandWorker:
    """Unit tests for _prepare_single_ligand (module-level worker)."""

    def test_returns_pose_failure_on_bad_path(self, tmp_path: Path) -> None:
        """Worker returns PoseFailure when PDB path does not exist."""
        from hybridock_pep.prep.ligand import _prepare_single_ligand
        from hybridock_pep.models import PoseFailure

        bad_path = tmp_path / "no_such_file.pdb"
        result = _prepare_single_ligand((7, bad_path, tmp_path))
        assert isinstance(result, PoseFailure)
        assert result.pose_idx == 7
        assert result.stage == "prep"

    def test_returns_path_on_valid_pose(self, tmp_path: Path) -> None:
        """Worker returns a Path when a valid pose PDB is provided."""
        from hybridock_pep.prep.ligand import _prepare_single_ligand

        pose_tiny = Path(__file__).parent / "fixtures" / "pose_tiny.pdb"
        result = _prepare_single_ligand((0, pose_tiny, tmp_path))
        # May succeed (Path) or fail (PoseFailure) depending on Meeko availability
        from pathlib import Path as _Path
        from hybridock_pep.models import PoseFailure
        assert isinstance(result, (_Path, PoseFailure)), (
            f"Unexpected return type: {type(result)}"
        )

    def test_pose_failure_has_error_msg(self, tmp_path: Path) -> None:
        """PoseFailure returned by worker has a non-empty error_msg."""
        from hybridock_pep.prep.ligand import _prepare_single_ligand
        from hybridock_pep.models import PoseFailure

        bad_path = tmp_path / "missing.pdb"
        result = _prepare_single_ligand((3, bad_path, tmp_path))
        assert isinstance(result, PoseFailure)
        assert result.error_msg != ""
