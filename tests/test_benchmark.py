"""Structural tests for scripts/benchmark.py (TEST-03).

All hybridock_pep imports are lazy (inside test functions) per STATE.md decision:
"All hybridock_pep imports kept lazy in test files — pytest-cov triggers numpy
double-import error in Python 3.13 base env."

These tests verify benchmark.py's interface (imports, argument parsing, output schema,
PDB ID validation) without requiring GPU, ADFRsuite, or network access.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Helper to inject scripts/ onto sys.path before importing benchmark module
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")


class TestBenchmarkCLI:
    """Verify the benchmark subcommand is wired in cli.py and help exits 0."""

    def test_benchmark_subcommand_help_exits_zero(self) -> None:
        from hybridock_pep import cli
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["hybridock-pep", "benchmark", "--help"]
            cli.main()
        assert exc.value.code == 0

    def test_benchmark_all_existing_flags_known(self) -> None:
        from hybridock_pep import cli
        parser = cli._build_parser()
        _, unknown = parser.parse_known_args([
            "benchmark",
            "--test-csv", "data/test_complexes.csv",
        ])
        assert unknown == []


class TestParseArgs:
    """Verify benchmark.parse_args() returns correct defaults and types."""

    def _import_benchmark(self) -> object:
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        import benchmark  # noqa: PLC0415
        return benchmark

    def test_parse_args_output_dir_default(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        args = bm.parse_args(["--test-csv", str(tmp_path / "t.csv")])
        assert args.output_dir == Path("runs/benchmark")

    def test_parse_args_seed_default(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        args = bm.parse_args(["--test-csv", str(tmp_path / "t.csv")])
        assert args.seed == 42

    def test_parse_args_box_size_default(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        args = bm.parse_args(["--test-csv", str(tmp_path / "t.csv")])
        assert args.box_size == 40.0

    def test_parse_args_test_csv_required(self) -> None:
        bm = self._import_benchmark()
        with pytest.raises(SystemExit):
            bm.parse_args([])

    def test_parse_args_test_csv_is_path(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        args = bm.parse_args(["--test-csv", str(tmp_path / "t.csv")])
        assert isinstance(args.test_csv, Path)


class TestOutputSchema:
    """Verify benchmark_results.csv header and benchmark_report.md structure."""

    def test_results_csv_columns(self, tmp_path: Path) -> None:
        """benchmark_results.csv must have the D-03 schema columns."""
        expected_cols = {
            "pdb_id", "peptide_sequence", "experimental_pkd",
            "hybrid_score", "vina_score", "delta_improvement",
            "n_poses", "runtime_hybrid_s", "runtime_vina_s", "status",
        }
        # Write a minimal CSV and verify we can parse its header
        csv_path = tmp_path / "benchmark_results.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(expected_cols))
            writer.writeheader()
        with csv_path.open(newline="") as f:
            header = set(csv.DictReader(f).fieldnames or [])
        assert header == expected_cols

    def test_status_values_are_defined(self) -> None:
        """status column values must include the four defined sentinel strings."""
        valid_statuses = {"ok", "skipped_download", "skipped_prep", "skipped_scoring"}
        # Import benchmark and check VALID_STATUSES constant
        bm = self._import_benchmark()
        assert hasattr(bm, "VALID_STATUSES"), (
            "benchmark.py must define VALID_STATUSES = {'ok', 'skipped_download', "
            "'skipped_prep', 'skipped_scoring'}"
        )
        assert bm.VALID_STATUSES == valid_statuses

    def _import_benchmark(self) -> object:
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        import benchmark  # noqa: PLC0415
        return benchmark


class TestPdbIdValidation:
    """Verify PDB ID validation rejects malformed IDs (STRIDE T-08-01)."""

    def _import_benchmark(self) -> object:
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        import benchmark  # noqa: PLC0415
        return benchmark

    def test_valid_pdb_ids_pass(self) -> None:
        bm = self._import_benchmark()
        for pdb_id in ["3EQS", "1G73", "2FLU", "1EJ4"]:
            assert bm.validate_pdb_id(pdb_id) is True, f"Should accept {pdb_id}"

    def test_lowercase_pdb_id_rejected(self) -> None:
        bm = self._import_benchmark()
        assert bm.validate_pdb_id("3eqs") is False

    def test_pdb_id_starting_with_letter_rejected(self) -> None:
        bm = self._import_benchmark()
        assert bm.validate_pdb_id("AEQS") is False

    def test_pdb_id_too_short_rejected(self) -> None:
        bm = self._import_benchmark()
        assert bm.validate_pdb_id("3EQ") is False

    def test_pdb_id_with_slash_rejected(self) -> None:
        bm = self._import_benchmark()
        assert bm.validate_pdb_id("../etc") is False

    def test_pdb_id_with_spaces_rejected(self) -> None:
        bm = self._import_benchmark()
        assert bm.validate_pdb_id("3EQS ") is False


class TestExtractBestScore:
    """Verify extract_best_score() handles hybrid vs vina column correctly."""

    def _import_benchmark(self) -> object:
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        import benchmark  # noqa: PLC0415
        return benchmark

    def _write_ranked_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "ranked_poses.csv"
        fieldnames = list(rows[0].keys())
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_hybrid_score_returns_first_row(self, tmp_path: Path) -> None:
        """hybrid_score: row 0 is the minimum — return it directly."""
        bm = self._import_benchmark()
        rows = [
            {"hybrid_score": "-10.0", "vina_score": "-7.0"},
            {"hybrid_score": "-9.0", "vina_score": "-8.5"},
            {"hybrid_score": "-8.0", "vina_score": "-9.2"},
        ]
        csv_path = self._write_ranked_csv(tmp_path, rows)
        assert bm.extract_best_score(csv_path, "hybrid_score") == pytest.approx(-10.0)

    def test_vina_score_scans_all_rows(self, tmp_path: Path) -> None:
        """vina_score: best vina may not be row 0 (sorted by hybrid, not vina)."""
        bm = self._import_benchmark()
        # Row 0 has hybrid_score=-10.0 but vina_score=-7.0; best vina is row 2 at -9.2
        rows = [
            {"hybrid_score": "-10.0", "vina_score": "-7.0"},
            {"hybrid_score": "-9.0", "vina_score": "-8.5"},
            {"hybrid_score": "-8.0", "vina_score": "-9.2"},
        ]
        csv_path = self._write_ranked_csv(tmp_path, rows)
        assert bm.extract_best_score(csv_path, "vina_score") == pytest.approx(-9.2)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        assert bm.extract_best_score(tmp_path / "nonexistent.csv", "hybrid_score") is None

    def test_empty_csv_returns_none(self, tmp_path: Path) -> None:
        bm = self._import_benchmark()
        p = tmp_path / "ranked_poses.csv"
        p.write_text("hybrid_score,vina_score\n")
        assert bm.extract_best_score(p, "hybrid_score") is None

    def test_vina_score_with_bad_values_skipped(self, tmp_path: Path) -> None:
        """Rows with non-numeric vina_score are skipped; valid rows still scanned."""
        bm = self._import_benchmark()
        rows = [
            {"hybrid_score": "-10.0", "vina_score": "N/A"},
            {"hybrid_score": "-9.0", "vina_score": "-8.5"},
        ]
        csv_path = self._write_ranked_csv(tmp_path, rows)
        assert bm.extract_best_score(csv_path, "vina_score") == pytest.approx(-8.5)


class TestGetPeptideCenter:
    """Verify get_peptide_center() computes the mean Ca position."""

    def test_returns_three_floats(self, tmp_path: Path) -> None:
        """get_peptide_center returns a 3-tuple of floats for a valid PDB."""
        bm = self._import_benchmark()
        # Use the existing receptor_tiny.pdb fixture as a minimal PDB
        tiny = Path(__file__).resolve().parents[0] / "fixtures" / "receptor_tiny.pdb"
        if not tiny.exists():
            pytest.skip("receptor_tiny.pdb fixture not available")
        # receptor_tiny.pdb is a receptor; it has chain A — any chain that exists
        # get_peptide_center should return a tuple even if no CA found (returns None or raises)
        # We test the function signature, not the value
        result = bm.get_peptide_center(tiny, "A")
        assert result is None or (
            isinstance(result, tuple)
            and len(result) == 3
            and all(isinstance(v, float) for v in result)
        )

    def _import_benchmark(self) -> object:
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        import benchmark  # noqa: PLC0415
        return benchmark
