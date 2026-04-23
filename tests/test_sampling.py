"""Tests for hybridock_pep.sampling — rapidock_runner and pose_io (SAMP-01)."""
from __future__ import annotations

import shutil
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestRapidockRunner:
    """Tests for run_sampling() in rapidock_runner.py (D-01 through D-11)."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        from hybridock_pep.models import DockConfig

        receptor = FIXTURES_DIR / "receptor_tiny.pdb"
        return DockConfig(
            peptide_sequence="ALA",
            receptor_path=receptor,
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=5,
        )

    def _make_mock_proc(self, returncode: int = 0):
        """Build a mock Popen process with sentinel-terminated readline."""
        proc = mock.MagicMock()
        proc.stdout.readline.side_effect = [b""]
        proc.stderr.readline.side_effect = [b""]
        proc.returncode = returncode
        proc.wait.return_value = returncode
        return proc

    def _setup_raw_poses(self, config, filenames: list[str]) -> None:
        """Create stub PDB files in the RAPiDock output directory."""
        raw_dir = config.output_dir / "poses_raw" / "poses_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (raw_dir / name).write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")

    # ------------------------------------------------------------------
    # task 4-01-01
    # ------------------------------------------------------------------

    def test_command_construction(self, config, tmp_path: Path, monkeypatch) -> None:
        """Command must be conda run ... python ...; all path args must be absolute."""
        from hybridock_pep.sampling.rapidock_runner import run_sampling

        self._setup_raw_poses(config, ["rank1.pdb"])

        proc = self._make_mock_proc(returncode=0)

        captured_cmd: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.append(cmd)
            return proc

        monkeypatch.setattr(
            "hybridock_pep.sampling.rapidock_runner.subprocess.Popen",
            fake_popen,
        )

        run_sampling(config)

        assert len(captured_cmd) == 1, "Popen should be called exactly once"
        cmd = captured_cmd[0]

        # Verify conda run prefix (D-04)
        assert cmd[0] == "conda"
        assert "run" in cmd
        assert "--no-capture-output" in cmd
        assert "-n" in cmd
        env_idx = cmd.index("-n")
        assert cmd[env_idx + 1] == "rapidock-env"
        assert "python" in cmd

        # Verify all path-like arguments are absolute (no relative segments)
        for arg in cmd:
            if arg.startswith("/") or arg == "conda" or arg == "run" or arg == "python":
                continue
            # Any argument that looks like a file path must be absolute
            p = Path(arg)
            if p.suffix in (".py", ".pdb", ".pdbqt", ".json") or "/" in arg:
                assert p.is_absolute(), f"Path argument must be absolute: {arg!r}"

    # ------------------------------------------------------------------
    # task 4-01-02
    # ------------------------------------------------------------------

    def test_nonzero_exit_raises(self, config, monkeypatch) -> None:
        """Non-zero subprocess exit code must raise RuntimeError containing the code."""
        from hybridock_pep.sampling.rapidock_runner import run_sampling

        proc = self._make_mock_proc(returncode=1)

        monkeypatch.setattr(
            "hybridock_pep.sampling.rapidock_runner.subprocess.Popen",
            lambda cmd, **kwargs: proc,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_sampling(config)

        assert "1" in str(exc_info.value), "RuntimeError message must contain exit code"

    # ------------------------------------------------------------------
    # task 4-01-03
    # ------------------------------------------------------------------

    def test_shortfall_warns(self, config, monkeypatch, caplog) -> None:
        """Fewer poses than requested → WARNING logged; list of available paths returned."""
        import logging

        from hybridock_pep.sampling.rapidock_runner import run_sampling

        # config.n_samples == 5, but we only put 3 poses
        self._setup_raw_poses(config, ["rank1.pdb", "rank2.pdb", "rank3.pdb"])

        proc = self._make_mock_proc(returncode=0)
        monkeypatch.setattr(
            "hybridock_pep.sampling.rapidock_runner.subprocess.Popen",
            lambda cmd, **kwargs: proc,
        )

        with caplog.at_level(logging.WARNING):
            # Ensure propagation so caplog captures the logger
            import hybridock_pep.sampling.rapidock_runner as rr_mod
            rr_mod_logger = rr_mod  # noqa — just trigger import
            result = run_sampling(config)

        # Must not raise
        assert isinstance(result, list)
        assert len(result) == 3

        # Must have logged a warning
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1, "Expected at least one WARNING log for pose shortfall"

    # ------------------------------------------------------------------
    # task 4-01-04
    # ------------------------------------------------------------------

    def test_zero_poses_raises(self, config, monkeypatch) -> None:
        """Zero output files → RuntimeError raised."""
        from hybridock_pep.sampling.rapidock_runner import run_sampling

        # Create the raw output dir but leave it empty
        raw_dir = config.output_dir / "poses_raw" / "poses_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        proc = self._make_mock_proc(returncode=0)
        monkeypatch.setattr(
            "hybridock_pep.sampling.rapidock_runner.subprocess.Popen",
            lambda cmd, **kwargs: proc,
        )

        with pytest.raises(RuntimeError):
            run_sampling(config)

    # ------------------------------------------------------------------
    # task 4-01-05
    # ------------------------------------------------------------------

    def test_file_rename(self, config, monkeypatch) -> None:
        """rank*.pdb files renamed to pose_N.pdb (sorted by rank); paths under output_dir/poses/."""
        from hybridock_pep.sampling.rapidock_runner import run_sampling

        self._setup_raw_poses(
            config,
            ["rank1_confidence.pdb", "rank2_confidence.pdb", "rank10_confidence.pdb"],
        )

        proc = self._make_mock_proc(returncode=0)
        monkeypatch.setattr(
            "hybridock_pep.sampling.rapidock_runner.subprocess.Popen",
            lambda cmd, **kwargs: proc,
        )

        result = run_sampling(config)

        assert len(result) == 3
        names = [p.name for p in result]
        assert "pose_0.pdb" in names, f"Expected pose_0.pdb in {names}"
        assert "pose_1.pdb" in names, f"Expected pose_1.pdb in {names}"
        assert "pose_2.pdb" in names, f"Expected pose_2.pdb in {names}"

        poses_dir = config.output_dir / "poses"
        for p in result:
            assert p.parent == poses_dir, f"Expected path under {poses_dir}, got {p.parent}"


class TestPoseIO:
    """Tests for parse_poses() in pose_io.py (D-12 through D-14)."""

    # ------------------------------------------------------------------
    # task 4-02-01
    # ------------------------------------------------------------------

    def test_parse_valid_pdb(self, tmp_path: Path) -> None:
        """Valid PDB with 3 CA atoms → 1 PoseRecord with shape (3, 3) float64 ca_coords."""
        from hybridock_pep.models import PoseRecord
        from hybridock_pep.sampling.pose_io import parse_poses

        src = FIXTURES_DIR / "pose_tiny.pdb"
        dst = tmp_path / "pose_0.pdb"
        shutil.copy(src, dst)

        records, failures = parse_poses(tmp_path)

        assert len(records) == 1
        assert isinstance(records[0], PoseRecord)
        assert records[0].ca_coords.shape == (3, 3)
        assert records[0].ca_coords.dtype == np.float64
        assert records[0].pose_idx == 0

    # ------------------------------------------------------------------
    # task 4-02-02
    # ------------------------------------------------------------------

    def test_parse_malformed_pdb(self, tmp_path: Path) -> None:
        """Malformed PDB → 1 PoseFailure(stage='parsing'); no records."""
        from hybridock_pep.models import PoseFailure
        from hybridock_pep.sampling.pose_io import parse_poses

        bad = tmp_path / "pose_0.pdb"
        bad.write_text("NOT A VALID PDB\n")

        records, failures = parse_poses(tmp_path)

        assert len(failures) == 1
        assert failures[0].stage == "parsing"
        assert len(records) == 0

    # ------------------------------------------------------------------
    # task 4-02-03
    # ------------------------------------------------------------------

    def test_batch_invariant(self, tmp_path: Path) -> None:
        """2 valid + 1 malformed → len(records)+len(failures)==3; no exception raised."""
        from hybridock_pep.sampling.pose_io import parse_poses

        src = FIXTURES_DIR / "pose_tiny.pdb"
        shutil.copy(src, tmp_path / "pose_0.pdb")
        shutil.copy(src, tmp_path / "pose_1.pdb")
        (tmp_path / "pose_2.pdb").write_text("GARBAGE\n")

        records, failures = parse_poses(tmp_path)

        assert len(records) + len(failures) == 3

    # ------------------------------------------------------------------
    # task 4-02-04 — D-14 SEQRES-first (BLOCKER 1)
    # ------------------------------------------------------------------

    def test_parse_seqres_preferred(self, tmp_path: Path) -> None:
        """SEQRES records take priority over ATOM residue iteration for sequence."""
        from hybridock_pep.sampling.pose_io import parse_poses

        pdb_text = (
            "SEQRES   1 A    2  ALA GLY\n"
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.522   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  N   GLY A   2       3.000   1.000   0.000  1.00  0.00           N\n"
            "ATOM      4  CA  GLY A   2       4.000   1.500   0.000  1.00  0.00           C\n"
            "END\n"
        )
        pose = tmp_path / "pose_0.pdb"
        pose.write_text(pdb_text)

        records, failures = parse_poses(tmp_path)

        assert len(records) == 1, f"Expected 1 record, got {len(records)} records and {len(failures)} failures"
        assert records[0].sequence == "AG", (
            f"Expected SEQRES-derived sequence 'AG', got {records[0].sequence!r}"
        )

    # ------------------------------------------------------------------
    # task 4-02-05 — D-14 ATOM fallback (BLOCKER 1)
    # ------------------------------------------------------------------

    def test_parse_atom_fallback(self, tmp_path: Path) -> None:
        """When no SEQRES records are present, sequence falls back to ATOM residue iteration."""
        from hybridock_pep.sampling.pose_io import parse_poses

        src = FIXTURES_DIR / "pose_tiny.pdb"  # 3 ALA residues, no SEQRES
        shutil.copy(src, tmp_path / "pose_0.pdb")

        records, failures = parse_poses(tmp_path)

        assert len(records) == 1, f"Expected 1 record, got {len(records)} records and {len(failures)} failures"
        assert records[0].sequence == "AAA", (
            f"Expected ATOM-fallback sequence 'AAA', got {records[0].sequence!r}"
        )
