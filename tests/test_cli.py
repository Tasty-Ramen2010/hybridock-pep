from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestSubcommands:
    def test_dock_subcommand_exists(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "dock", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_calibrate_subcommand_exists(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "calibrate", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_prep_subcommand_exists(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "prep", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_benchmark_subcommand_exists(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "benchmark", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_crystal_score_subcommand_exists(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "crystal-score", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_crystal_score_rejects_unk_pose(self, tmp_path: Path) -> None:
        """A PDBQT-derived (UNK-labelled) pose is rejected with a clear, actionable error."""
        from hybridock_pep import cli
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        pep = tmp_path / "unk.pdb"
        pep.write_text("ATOM      1  CA  UNK A   1       3.000   0.000   0.000  1.00  0.00           C\n")
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "crystal-score", "--receptor", str(rec),
                        "--peptide-pdb", str(pep), "--peptide", "A"]
            cli.main()
        assert exc.value.code == 2  # argparse error exit


class TestDockSubcommand:
    def test_dock_all_flags_defined(self) -> None:
        from hybridock_pep import cli
        parser = cli._build_parser()
        _, unknown = parser.parse_known_args([
            "dock", "--peptide", "LISDAELEAIFEADC",
            "--receptor", "/tmp/r.pdb",
            "--site", "1.0", "2.0", "3.0",
            "--box", "20.0",
            "--output-dir", "/tmp/out",
            "--n-samples", "10",
            "--scoring", "vina,ad4",
            "--seed", "42",
            "--calibration", "data/calibration.json",
        ])
        assert unknown == []


class TestValidation:
    def test_invalid_peptide_exits_2(self, tmp_path: Path) -> None:
        from hybridock_pep import cli
        receptor = tmp_path / "receptor.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        with pytest.raises(SystemExit) as exc:
            sys.argv = [
                "hybridock-pep", "dock",
                "--peptide", "XYZ123",
                "--receptor", str(receptor),
                "--site", "1.0", "2.0", "3.0",
                "--box", "20.0",
                "--output-dir", str(tmp_path / "out"),
            ]
            cli.main()
        assert exc.value.code == 2

    def test_missing_receptor_exits_2(self, tmp_path: Path) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = [
                "hybridock-pep", "dock",
                "--peptide", "AAAA",
                "--receptor", "/nonexistent/r.pdb",
                "--site", "1.0", "2.0", "3.0",
                "--box", "20.0",
                "--output-dir", str(tmp_path / "out"),
            ]
            cli.main()
        assert exc.value.code == 2

    def test_input_poses_and_n_samples_exclusive(self, tmp_path: Path) -> None:
        from hybridock_pep import cli
        receptor = tmp_path / "receptor.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        with pytest.raises(SystemExit) as exc:
            sys.argv = [
                "hybridock-pep", "dock",
                "--peptide", "AAAA",
                "--receptor", str(receptor),
                "--site", "1.0", "2.0", "3.0",
                "--box", "20.0",
                "--output-dir", str(tmp_path / "out"),
                "--input-poses", str(tmp_path),
                "--n-samples", "10",
            ]
            cli.main()
        assert exc.value.code == 2


class TestSeed:
    def test_seed_stored_in_dockconfig(self, tmp_path: Path) -> None:
        from hybridock_pep.models import DockConfig
        receptor = tmp_path / "receptor.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        config = DockConfig(
            peptide_sequence="AAAA",
            receptor_path=receptor,
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path,
            seed=42,
        )
        assert config.seed == 42

    def test_no_seed_is_none(self, tmp_path: Path) -> None:
        from hybridock_pep.models import DockConfig
        receptor = tmp_path / "receptor.pdb"
        receptor.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        config = DockConfig(
            peptide_sequence="AAAA",
            receptor_path=receptor,
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path,
        )
        assert config.seed is None


class TestLogLevel:
    """Default verbosity must be quiet — PipelineProgress is the intended
    default-mode UI; routine logger.info() calls from every module should
    not print unless the user asked for -v/-vv. Regression test for the bug
    where main() always set INFO regardless of verbosity 0 vs 1."""

    @pytest.fixture(autouse=True)
    def _reset_root_logger(self):
        # logging.basicConfig() is a no-op after the first call in a process
        # (handlers already attached) — clear them so each test's call to
        # cli.main() genuinely re-applies its own level, and restore the
        # original root logger state afterward so this doesn't leak into
        # other test files' logging expectations.
        import logging
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers.clear()
        yield
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)

    def _run_main_with_verbosity(self, monkeypatch, verbose_flags: list[str]) -> int:
        import logging
        from hybridock_pep import cli

        monkeypatch.setattr(sys, "argv", ["hybridock-pep", *verbose_flags])
        # main() with no subcommand just prints help and returns — enough to
        # exercise the logging.basicConfig() call without running a real dock.
        cli.main()
        return logging.getLogger().getEffectiveLevel()

    def test_default_verbosity_is_warning(self, monkeypatch) -> None:
        import logging
        level = self._run_main_with_verbosity(monkeypatch, [])
        assert level == logging.WARNING

    def test_single_v_is_info(self, monkeypatch) -> None:
        import logging
        level = self._run_main_with_verbosity(monkeypatch, ["-v"])
        assert level == logging.INFO

    def test_double_v_is_debug(self, monkeypatch) -> None:
        import logging
        level = self._run_main_with_verbosity(monkeypatch, ["-vv"])
        assert level == logging.DEBUG
