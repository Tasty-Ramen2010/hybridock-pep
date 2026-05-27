"""Tests for hybridock_pep.prep — PREP-01, PREP-02, PREP-03."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# NOTE: hybridock_pep imports (which pull in numpy) are kept inside test methods/fixtures
# to avoid pytest-cov double-import conflicts in the system Python 3.13 environment.
# The three classes below (TestReceptorPrep, TestLigandBatch, TestGrids) use lazy imports
# in the same style as the rest of this file.

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def meeko_available() -> None:
    """Session-scoped fixture that skips if meeko is not installed."""
    pytest.importorskip("meeko", reason="meeko not installed (score-env only)")


@pytest.fixture(scope="session")
def babel_available() -> None:
    """Session-scoped fixture that skips if ADFRsuite's babel is not on PATH."""
    import shutil
    if shutil.which("babel") is None:
        pytest.skip("babel not found on PATH — install ADFRsuite (score-env only)")


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

            with pytest.raises(PrepError, match="prepare_receptor failed"):
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

    @pytest.fixture()
    def receptor_pdbqt(self, tmp_path: Path) -> Path:
        """Minimal receptor PDBQT for _build_gpf tests."""
        pdbqt = tmp_path / "receptor.pdbqt"
        pdbqt.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.00  0.00    +0.000 C\n"
        )
        return pdbqt

    def test_ligand_types_contains_hd(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """ligand_types line must contain HD — required for receptor.HD.map generation."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
        # Find the ligand_types line
        ligand_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("ligand_types")), None
        )
        assert ligand_line is not None, "ligand_types line missing from GPF"
        assert "HD" in ligand_line, f"HD not in ligand_types line: {ligand_line!r}"

    def test_gridcenter_matches_site_coords(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """gridcenter line must match DockConfig.site_coords."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
        assert "gridcenter 22.5 14.1 38.7" in gpf_text

    def test_npts_derived_from_box_size(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """npts = int((box_size + 2*AD4_GRID_PADDING) / 0.375) for a cubic box.

        The AD4 grid is padded by _AD4_GRID_PADDING (2.0 Å) on each side beyond
        the Vina box to prevent boundary interpolation artefacts.
        """
        from hybridock_pep.prep.grids import _AD4_GRID_PADDING, _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
        # box_size=20.0, padding=2.0 each side → padded=24.0, spacing=0.375 → npts=64
        padded_box = 20.0 + 2 * _AD4_GRID_PADDING
        expected_npts = int(padded_box / 0.375)
        npts_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("npts")), None
        )
        assert npts_line is not None, "npts line missing from GPF"
        parts = npts_line.split()
        assert parts[1] == str(expected_npts), f"npts x mismatch: {npts_line!r}"
        assert parts[2] == str(expected_npts), f"npts y mismatch: {npts_line!r}"
        assert parts[3] == str(expected_npts), f"npts z mismatch: {npts_line!r}"

    def test_receptor_line_is_filename_only(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """receptor line must be 'receptor receptor.pdbqt' — filename, not full path."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
        receptor_line = next(
            (ln for ln in gpf_text.splitlines() if ln.startswith("receptor ") and "pdbqt" in ln),
            None,
        )
        assert receptor_line is not None, "receptor line missing from GPF"
        assert receptor_line == "receptor receptor.pdbqt", (
            f"receptor line should be filename-only, got: {receptor_line!r}"
        )

    def test_hd_map_line_present(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """map receptor.HD.map line must appear in GPF."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
        assert "map receptor.HD.map" in gpf_text

    def test_spacing_is_0375(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """spacing line must be 0.375."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_text = _build_gpf(config, maps_dir, receptor_pdbqt)
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


# ---------------------------------------------------------------------------
# Plan 02-04: Required class names — TestReceptorPrep, TestLigandBatch, TestGrids
# These classes satisfy the acceptance criteria for plan 02-04 (class names,
# monkeypatch style, exact match strings). Existing classes above are preserved.
# ---------------------------------------------------------------------------


class TestReceptorPrep:
    """PREP-01 contract tests using pytest monkeypatch style (02-04 acceptance criteria)."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        from hybridock_pep.models import DockConfig

        return DockConfig(
            peptide_sequence="LISDAELEAIFEADC",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    def _mock_ntf(self, monkeypatch, tmp_path: Path) -> None:
        """Set up NamedTemporaryFile mock so receptor.py path resolution works."""
        tmp_file1 = tmp_path / "tmp_cleaned.pdb"
        tmp_file1.write_text("")
        tmp_file2 = tmp_path / "tmp_fixed.pdb"
        tmp_file2.write_text("")

        ntf_ctx1 = MagicMock()
        ntf_ctx1.__enter__ = MagicMock(return_value=ntf_ctx1)
        ntf_ctx1.__exit__ = MagicMock(return_value=False)
        ntf_ctx1.name = str(tmp_file1)

        ntf_ctx2 = MagicMock()
        ntf_ctx2.__enter__ = MagicMock(return_value=ntf_ctx2)
        ntf_ctx2.__exit__ = MagicMock(return_value=False)
        ntf_ctx2.name = str(tmp_file2)

        call_count = {"n": 0}
        contexts = [ntf_ctx1, ntf_ctx2]

        def ntf_side_effect(*args, **kwargs):
            ctx = contexts[call_count["n"] % 2]
            call_count["n"] += 1
            return ctx

        monkeypatch.setattr("hybridock_pep.prep.receptor.tempfile.NamedTemporaryFile", ntf_side_effect)

    def test_prepare_receptor_calls_prepare_receptor4(
        self, config, tmp_path: Path, monkeypatch
    ) -> None:
        """prepare_receptor calls prepare_receptor4.py and returns receptor.pdbqt path."""
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return mock_result

        monkeypatch.setattr("hybridock_pep.prep.receptor.subprocess.run", fake_run)
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFixer", MagicMock())
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFile", MagicMock())
        self._mock_ntf(monkeypatch, tmp_path)

        result = prepare_receptor(config)

        assert result == config.output_dir / "receptor.pdbqt"
        assert len(calls) == 1, "subprocess.run should have been called exactly once"
        assert calls[0][0] == "prepare_receptor", (
            f"First arg to subprocess must be 'prepare_receptor', got: {calls[0][0]!r}"
        )

    def test_prepare_receptor_nonzero_exit_raises_prep_error(
        self, config, tmp_path: Path, monkeypatch
    ) -> None:
        """Non-zero returncode from prepare_receptor4.py raises PrepError containing stderr."""
        from hybridock_pep.prep import PrepError
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="fatal: bad input"
        )
        monkeypatch.setattr(
            "hybridock_pep.prep.receptor.subprocess.run", lambda *a, **kw: mock_result
        )
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFixer", MagicMock())
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFile", MagicMock())
        self._mock_ntf(monkeypatch, tmp_path)

        with pytest.raises(PrepError, match="fatal: bad input"):
            prepare_receptor(config)

    def test_prepare_receptor_always_regenerates(
        self, config, tmp_path: Path, monkeypatch
    ) -> None:
        """Calling prepare_receptor twice with the same config raises no exception (no cache guard)."""
        from hybridock_pep.prep.receptor import prepare_receptor

        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        ntf_calls = [0]

        # Need to provide enough temp file mocks for two calls (2 calls × 2 NTF each = 4)
        tmp_files = [tmp_path / f"tmp_{i}.pdb" for i in range(4)]
        for f in tmp_files:
            f.write_text("")

        contexts = []
        for f in tmp_files:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.name = str(f)
            contexts.append(ctx)

        def ntf_side_effect(*args, **kwargs):
            idx = ntf_calls[0] % len(contexts)
            ntf_calls[0] += 1
            return contexts[idx]

        monkeypatch.setattr("hybridock_pep.prep.receptor.subprocess.run", lambda *a, **kw: mock_result)
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFixer", MagicMock())
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFile", MagicMock())
        monkeypatch.setattr("hybridock_pep.prep.receptor.tempfile.NamedTemporaryFile", ntf_side_effect)

        # Must not raise on either call
        prepare_receptor(config)
        prepare_receptor(config)

    def test_pdbfixer_called_before_subprocess(
        self, config, tmp_path: Path, monkeypatch
    ) -> None:
        """pdbfixer findMissingResidues/findMissingAtoms/addMissingHydrogens called before subprocess."""
        from hybridock_pep.prep.receptor import prepare_receptor

        call_order: list[str] = []

        class SpyFixer:
            def __init__(self, filename: str):
                self.topology = MagicMock()
                self.positions = MagicMock()

            def findMissingResidues(self):
                call_order.append("findMissingResidues")

            def findMissingAtoms(self):
                call_order.append("findMissingAtoms")

            def addMissingHydrogens(self, ph: float):
                call_order.append(f"addMissingHydrogens({ph})")

        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def spy_subprocess(*args, **kwargs):
            call_order.append("subprocess.run")
            return mock_result

        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFixer", SpyFixer)
        monkeypatch.setattr("hybridock_pep.prep.receptor.PDBFile", MagicMock())
        monkeypatch.setattr("hybridock_pep.prep.receptor.subprocess.run", spy_subprocess)
        self._mock_ntf(monkeypatch, tmp_path)

        prepare_receptor(config)

        assert "findMissingResidues" in call_order
        assert "findMissingAtoms" in call_order
        assert any("addMissingHydrogens" in c for c in call_order)
        # All three pdbfixer methods must appear before subprocess.run
        subprocess_idx = call_order.index("subprocess.run")
        for method in ("findMissingResidues", "findMissingAtoms"):
            assert call_order.index(method) < subprocess_idx, (
                f"{method} was not called before subprocess.run"
            )


class TestLigandBatch:
    """PREP-02 contract tests — collect-all-failures semantics (02-04 acceptance criteria)."""

    @pytest.fixture()
    def pose_tiny(self) -> Path:
        return FIXTURES_DIR / "pose_tiny.pdb"

    def test_batch_single_pose_success(
        self, tmp_path: Path, pose_tiny: Path, babel_available
    ) -> None:
        """Single valid pose produces one success and zero failures."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        successes, failures = prepare_ligand_batch(
            [pose_tiny], tmp_path / "pdbqt_out", max_workers=1
        )
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(failures) == 0, f"Expected 0 failures, got {len(failures)}"
        assert successes[0].exists(), f"PDBQT file not written: {successes[0]}"

    def test_batch_missing_pdb_collected_as_failure(self, tmp_path: Path) -> None:
        """A nonexistent PDB path is collected as a PoseFailure with stage='prep'."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        bad_path = tmp_path / "nonexistent_pose.pdb"
        successes, failures = prepare_ligand_batch(
            [bad_path], tmp_path / "pdbqt_out", max_workers=1
        )
        assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}"
        assert failures[0].stage == "prep", (
            f"Expected stage='prep', got {failures[0].stage!r}"
        )
        assert len(successes) == 0, f"Expected 0 successes, got {len(successes)}"

    def test_batch_successes_plus_failures_equals_input(
        self, tmp_path: Path, pose_tiny: Path
    ) -> None:
        """len(successes) + len(failures) always equals len(input paths)."""
        from hybridock_pep.prep.ligand import prepare_ligand_batch

        bad_path = tmp_path / "missing.pdb"
        pdb_paths = [pose_tiny, bad_path]
        successes, failures = prepare_ligand_batch(
            pdb_paths, tmp_path / "pdbqt_out", max_workers=1
        )
        assert len(successes) + len(failures) == 2, (
            f"successes ({len(successes)}) + failures ({len(failures)}) != 2"
        )


class TestGrids:
    """PREP-03 contract tests — GPF content + HD map guard (02-04 acceptance criteria)."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        from hybridock_pep.models import DockConfig

        return DockConfig(
            peptide_sequence="LISDAELEAIFEADC",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(22.5, 14.1, 38.7),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    @pytest.fixture()
    def receptor_pdbqt(self, tmp_path: Path) -> Path:
        pdbqt = tmp_path / "receptor.pdbqt"
        pdbqt.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.00  0.00    +0.000 C\n"
        )
        return pdbqt

    def test_build_gpf_contains_hd_type(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """ligand_types line in GPF must contain HD."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_content = _build_gpf(config, maps_dir, receptor_pdbqt)
        assert "HD" in gpf_content, "HD not found anywhere in GPF"
        assert "ligand_types" in gpf_content and "HD" in gpf_content, (
            f"Expected ligand_types containing HD in GPF, got:\n{gpf_content}"
        )

    def test_build_gpf_npts_from_box_size(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """npts = int((box_size + 2*_AD4_GRID_PADDING) / 0.375) for a cubic box.

        The AD4 grid is padded by 2 Å on each side beyond the Vina box to prevent
        boundary interpolation artefacts from autogrid4.
        """
        from hybridock_pep.prep.grids import _AD4_GRID_PADDING, _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_content = _build_gpf(config, maps_dir, receptor_pdbqt)
        # box_size=20.0, _AD4_GRID_PADDING=2.0 → padded=24.0, spacing=0.375 → npts=64
        padded = 20.0 + 2 * _AD4_GRID_PADDING
        expected = int(padded / 0.375)
        assert f"npts {expected} {expected} {expected}" in gpf_content, (
            f"Expected 'npts {expected} {expected} {expected}' in GPF, got:\n{gpf_content}"
        )

    def test_build_gpf_gridcenter_from_site_coords(self, config, receptor_pdbqt: Path, tmp_path: Path) -> None:
        """gridcenter line must match DockConfig.site_coords exactly."""
        from hybridock_pep.prep.grids import _build_gpf

        maps_dir = tmp_path / "maps"
        maps_dir.mkdir()
        gpf_content = _build_gpf(config, maps_dir, receptor_pdbqt)
        assert "gridcenter 22.5 14.1 38.7" in gpf_content, (
            f"Expected 'gridcenter 22.5 14.1 38.7' in GPF, got:\n{gpf_content}"
        )

    def test_generate_ad4_maps_hd_map_missing_raises(
        self,
        config,
        receptor_pdbqt: Path,
        monkeypatch,
    ) -> None:
        """PrepError raised with 'receptor.HD.map not found after autogrid4' when HD.map absent."""
        from hybridock_pep.prep import PrepError
        from hybridock_pep.prep.grids import generate_ad4_maps

        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def fake_autogrid4_no_hd(cmd, **kwargs):
            # Run succeeds but does NOT write receptor.HD.map
            return mock_result

        monkeypatch.setattr("hybridock_pep.prep.grids.subprocess.run", fake_autogrid4_no_hd)

        with pytest.raises(PrepError, match="receptor.HD.map not found after autogrid4"):
            generate_ad4_maps(config, receptor_pdbqt)

    def test_generate_ad4_maps_success_returns_maps_dir(
        self,
        config,
        receptor_pdbqt: Path,
        monkeypatch,
    ) -> None:
        """generate_ad4_maps returns output_dir/maps/ when HD.map is present."""
        from hybridock_pep.prep.grids import generate_ad4_maps

        maps_dir = config.output_dir / "maps"

        def fake_autogrid4(*args, **kwargs):
            cwd = Path(kwargs.get("cwd", str(maps_dir)))
            (cwd / "receptor.HD.map").write_text("fake HD map")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr("hybridock_pep.prep.grids.subprocess.run", fake_autogrid4)

        result = generate_ad4_maps(config, receptor_pdbqt)

        assert result == config.output_dir / "maps"
        assert result.is_dir()
