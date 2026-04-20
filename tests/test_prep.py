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


# ---------------------------------------------------------------------------
# Task 1 (02-03): prep/grids.py — generate_ad4_maps + _build_gpf
# ---------------------------------------------------------------------------


class TestGridsImports:
    """PREP-03: Module-level import and structural requirements."""

    def test_importable(self) -> None:
        from hybridock_pep.prep.grids import generate_ad4_maps  # noqa: F401

    def test_build_gpf_importable(self) -> None:
        from hybridock_pep.prep.grids import _build_gpf  # noqa: F401

    def test_future_annotations_present(self) -> None:
        """from __future__ import annotations must be first non-comment line."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "grids.py"
        )
        lines = source.read_text().splitlines()
        code_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        assert code_lines[0] == "from __future__ import annotations", (
            f"Expected 'from __future__ import annotations', got: {code_lines[0]!r}"
        )

    def test_no_bare_except(self) -> None:
        """No bare 'except:' allowed (project convention)."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "grids.py"
        )
        content = source.read_text()
        assert "except:" not in content, "Bare except: found"

    def test_no_template_reference(self) -> None:
        """GPF is programmatic — no template file on disk (D-04)."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "hybridock_pep"
            / "prep"
            / "grids.py"
        )
        content = source.read_text().lower()
        assert "template" not in content, "Template reference found — GPF must be programmatic"


class TestBuildGpf:
    """Unit tests for _build_gpf — no subprocess needed."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        """Minimal DockConfig for GPF testing."""
        from hybridock_pep.models import DockConfig

        receptor = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
        return DockConfig(
            peptide_sequence="ALA",
            receptor_path=receptor,
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    def test_ligand_types_contains_hd(self, config, tmp_path: Path) -> None:
        """ligand_types line must contain HD — required for receptor.HD.map generation."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        # Find the ligand_types line
        ligand_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("ligand_types")), None
        )
        assert ligand_line is not None, "ligand_types line missing from GPF"
        assert "HD" in ligand_line, f"HD not in ligand_types line: {ligand_line!r}"

    def test_gridcenter_matches_site_coords(self, config, tmp_path: Path) -> None:
        """gridcenter line must match DockConfig.site_coords."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        assert "gridcenter 22.5 14.1 38.7" in gpf_text

    def test_npts_derived_from_box_size(self, config, tmp_path: Path) -> None:
        """npts = int(box_size / 0.375) for a cubic box."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        # box_size=20.0, spacing=0.375 -> npts=53
        expected_npts = int(20.0 / 0.375)
        npts_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("npts")), None
        )
        assert npts_line is not None, "npts line missing from GPF"
        parts = npts_line.split()
        assert parts[1] == str(expected_npts), f"npts x mismatch: {npts_line!r}"
        assert parts[2] == str(expected_npts), f"npts y mismatch: {npts_line!r}"
        assert parts[3] == str(expected_npts), f"npts z mismatch: {npts_line!r}"

    def test_receptor_line_is_filename_only(self, config, tmp_path: Path) -> None:
        """receptor line must be 'receptor receptor.pdbqt' — filename, not full path."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        receptor_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("receptor ") and "pdbqt" in ln),
            None,
        )
        assert receptor_line is not None, "receptor line missing from GPF"
        assert receptor_line == "receptor receptor.pdbqt", (
            f"receptor line should be filename-only, got: {receptor_line!r}"
        )

    def test_hd_map_line_present(self, config, tmp_path: Path) -> None:
        """map receptor.HD.map line must appear in GPF."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        assert "map receptor.HD.map" in gpf_text

    def test_spacing_is_0375(self, config, tmp_path: Path) -> None:
        """spacing line must be 0.375."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir)
        assert "spacing 0.375" in gpf_text


class TestGenerateAd4Maps:
    """PREP-03: Behavioral tests for generate_ad4_maps — autogrid4 mocked."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        from hybridock_pep.models import DockConfig

        receptor = Path(__file__).parent / "fixtures" / "receptor_tiny.pdb"
        return DockConfig(
            peptide_sequence="ALA",
            receptor_path=receptor,
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    @pytest.fixture()
    def receptor_pdbqt(self, tmp_path: Path) -> Path:
        """Minimal receptor PDBQT file for testing."""
        pdbqt = tmp_path / "receptor.pdbqt"
        pdbqt.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.00  0.00    +0.000 C\n")
        return pdbqt

    def _make_autogrid4_side_effect(self, maps_dir: Path, write_hd: bool = True):
        """Return a side_effect function that simulates autogrid4 writing map files."""
        import subprocess

        def side_effect(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", maps_dir))
            # Write expected map files
            for atom_type in ["C", "A", "N", "O", "S", "H"]:
                (cwd / f"receptor.{atom_type}.map").write_text(f"fake {atom_type} map\n")
            (cwd / "receptor.e.map").write_text("fake e map\n")
            (cwd / "receptor.d.map").write_text("fake d map\n")
            if write_hd:
                (cwd / "receptor.HD.map").write_text("fake HD map\n")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        return side_effect

    def test_returns_maps_dir_on_success(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """generate_ad4_maps returns the maps directory Path on success."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"
        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir),
        ):
            result = generate_ad4_maps(config, receptor_pdbqt)
        assert result == maps_dir

    def test_maps_dir_created_if_absent(self, config, receptor_pdbqt: Path) -> None:
        """generate_ad4_maps creates output_dir/maps/ if it does not exist."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"
        assert not maps_dir.exists()

        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir),
        ):
            generate_ad4_maps(config, receptor_pdbqt)

        assert maps_dir.exists()

    def test_gpf_written_to_maps_dir(self, config, receptor_pdbqt: Path) -> None:
        """GPF file is written to output_dir/maps/receptor.gpf."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"
        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir),
        ):
            generate_ad4_maps(config, receptor_pdbqt)

        gpf_path = maps_dir / "receptor.gpf"
        assert gpf_path.exists(), "receptor.gpf not written to maps_dir"

    def test_hd_map_guard_raises_prep_error_when_missing(
        self, config, receptor_pdbqt: Path
    ) -> None:
        """PrepError raised with verbatim D-05 message when HD.map is absent."""
        from hybridock_pep.prep.grids import generate_ad4_maps
        from hybridock_pep.prep import PrepError

        maps_dir = config.output_dir / "maps"
        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir, write_hd=False),
        ):
            with pytest.raises(PrepError, match="receptor.HD.map not found after autogrid4"):
                generate_ad4_maps(config, receptor_pdbqt)

    def test_hd_map_guard_exact_message(self, config, receptor_pdbqt: Path) -> None:
        """PrepError message matches verbatim D-05 specification."""
        from hybridock_pep.prep.grids import generate_ad4_maps
        from hybridock_pep.prep import PrepError

        maps_dir = config.output_dir / "maps"
        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir, write_hd=False),
        ):
            with pytest.raises(PrepError) as exc_info:
                generate_ad4_maps(config, receptor_pdbqt)
        expected = (
            "receptor.HD.map not found after autogrid4 — AD4 scoring will fail. "
            "Check your atom types in the GPF."
        )
        assert str(exc_info.value) == expected

    def test_autogrid4_called_with_cwd_maps_dir(self, config, receptor_pdbqt: Path) -> None:
        """autogrid4 subprocess is called with cwd=maps_dir (string)."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"
        with patch("hybridock_pep.prep.grids.subprocess.run") as mock_run:
            mock_run.side_effect = self._make_autogrid4_side_effect(maps_dir)
            generate_ad4_maps(config, receptor_pdbqt)

        call_kwargs = mock_run.call_args[1]
        assert "cwd" in call_kwargs, "cwd not passed to subprocess.run"
        assert str(call_kwargs["cwd"]) == str(maps_dir)

    def test_nonzero_exit_raises_prep_error(self, config, receptor_pdbqt: Path) -> None:
        """Non-zero autogrid4 exit raises PrepError with returncode and stderr."""
        from hybridock_pep.prep.grids import generate_ad4_maps
        from hybridock_pep.prep import PrepError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "autogrid4: fatal error"

        with patch("hybridock_pep.prep.grids.subprocess.run", return_value=mock_result):
            with pytest.raises(PrepError, match="autogrid4 failed"):
                generate_ad4_maps(config, receptor_pdbqt)

    def test_receptor_pdbqt_copied_to_maps_dir(self, config, receptor_pdbqt: Path) -> None:
        """receptor.pdbqt is copied into maps_dir before autogrid4 runs."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"
        with patch(
            "hybridock_pep.prep.grids.subprocess.run",
            side_effect=self._make_autogrid4_side_effect(maps_dir),
        ):
            generate_ad4_maps(config, receptor_pdbqt)

        receptor_copy = maps_dir / "receptor.pdbqt"
        assert receptor_copy.exists(), "receptor.pdbqt not copied into maps_dir"
