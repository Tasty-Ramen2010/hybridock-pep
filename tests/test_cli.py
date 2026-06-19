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
