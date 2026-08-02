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

        # On a CUDA host, backend detection shells out to `nvidia-smi` through
        # the same Popen, so counting every call asserts "this machine has no
        # GPU" rather than anything about the command. Select the sampling
        # invocation instead.
        sampling_cmds = [c for c in captured_cmd if any(str(a).endswith("run_rapidock.py") for a in c)]
        assert len(sampling_cmds) == 1, (
            f"expected exactly one run_rapidock.py invocation, got {captured_cmd}"
        )
        cmd = sampling_cmds[0]

        # Verify direct python3 invocation (no conda run — see module docstring)
        # cmd[0] must be the rapidock env's python3 (absolute path ending in python3)
        assert cmd[0].endswith("python3"), f"Expected python3 as first arg, got {cmd[0]!r}"
        assert Path(cmd[0]).is_absolute(), f"python3 path must be absolute: {cmd[0]!r}"
        # cmd[1] must be the run_rapidock.py shim (absolute path)
        assert cmd[1].endswith("run_rapidock.py"), f"Expected run_rapidock.py as second arg, got {cmd[1]!r}"

        # Verify all path-like arguments are absolute (no relative segments)
        for arg in cmd:
            p = Path(arg)
            if p.suffix in (".py", ".pdb", ".pdbqt", ".json") or (arg.startswith("/") and "/" in arg):
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


# ---------------------------------------------------------------------------
# Cross-platform: _detect_device_platform + _seed_everything macOS paths
# ---------------------------------------------------------------------------

class TestCrossPlatformDetection:
    """Tests for macOS/Linux/WSL2 device detection and seed safety."""

    def test_detect_device_linux_with_cuda(self) -> None:
        """On Linux with nvidia-smi present and returning exit 0, return CUDA label."""
        from hybridock_pep.sampling.rapidock_runner import _detect_device_platform
        import subprocess as _sp

        # which() returns a path; subprocess.run() exits 0 → CUDA detected
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        with mock.patch("sys.platform", "linux"):
            with mock.patch("hybridock_pep.sampling.rapidock_runner.shutil.which",
                            return_value="/usr/bin/nvidia-smi"):
                with mock.patch("subprocess.run", return_value=mock_result):
                    label = _detect_device_platform()
        assert "CUDA" in label

    def test_detect_device_linux_no_gpu(self) -> None:
        """On Linux without nvidia-smi and no WSL2 cuda libs, return CPU label."""
        from hybridock_pep.sampling.rapidock_runner import _detect_device_platform

        with mock.patch("sys.platform", "linux"):
            with mock.patch("hybridock_pep.sampling.rapidock_runner.shutil.which",
                            return_value=None):
                # Also gate out WSL2 paths + AMD/Intel detection so we genuinely
                # land on the CPU fallback (the test machine may have those
                # libraries from a real WSL2 install).
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("pathlib.Path.exists", return_value=False):
                        label = _detect_device_platform()
        assert "CPU" in label and "Linux" in label

    def test_detect_device_macos_arm64(self) -> None:
        """On macOS arm64, should return MPS label."""
        from hybridock_pep.sampling.rapidock_runner import _detect_device_platform
        import platform as _platform

        with mock.patch("sys.platform", "darwin"):
            with mock.patch.object(_platform, "machine", return_value="arm64"):
                label = _detect_device_platform()
        assert "MPS" in label and "Apple Silicon" in label

    def test_detect_device_macos_intel(self) -> None:
        """On macOS x86_64, should return CPU label for Intel."""
        from hybridock_pep.sampling.rapidock_runner import _detect_device_platform
        import platform as _platform

        with mock.patch("sys.platform", "darwin"):
            with mock.patch.object(_platform, "machine", return_value="x86_64"):
                label = _detect_device_platform()
        assert "CPU" in label and "Intel" in label

    def test_seed_everything_no_cuda(self) -> None:
        """_seed_everything must not call cuda.manual_seed_all when CUDA unavailable (macOS)."""
        import sys
        import types

        # run_rapidock.py lives in the rapidock env (Python 3.10) — torch is not in score-env.
        # Inject a fake torch module so we can import the shim and test _seed_everything.
        mock_torch = types.ModuleType("torch")
        mock_cuda_mod = types.ModuleType("torch.cuda")
        mock_cuda_mod.is_available = mock.MagicMock(return_value=False)
        mock_cuda_mod.manual_seed_all = mock.MagicMock()
        mock_torch.cuda = mock_cuda_mod
        mock_torch.manual_seed = mock.MagicMock()
        mock_torch.backends = types.ModuleType("torch.backends")
        mock_torch.backends.mps = types.SimpleNamespace(is_available=lambda: False)

        mock_np = types.ModuleType("numpy")
        mock_np_random = types.ModuleType("numpy.random")
        mock_np_random.seed = mock.MagicMock()
        mock_np.random = mock_np_random

        shim_path = Path(__file__).parent.parent / "src" / "hybridock_pep" / "sampling"
        sys.path.insert(0, str(shim_path))
        orig_torch = sys.modules.pop("torch", None)
        orig_np = sys.modules.pop("numpy", None)
        orig_rr = sys.modules.pop("run_rapidock", None)

        sys.modules["torch"] = mock_torch
        sys.modules["numpy"] = mock_np
        sys.modules["numpy.random"] = mock_np_random
        try:
            import run_rapidock as rr  # noqa: PLC0415
            import importlib; importlib.reload(rr)
            rr._seed_everything(42)
        except ImportError:
            pytest.skip("run_rapidock not importable — OK in isolation")
            return
        finally:
            sys.path.pop(0)
            if orig_torch is not None:
                sys.modules["torch"] = orig_torch
            else:
                sys.modules.pop("torch", None)
            if orig_np is not None:
                sys.modules["numpy"] = orig_np
            else:
                sys.modules.pop("numpy", None)
            if orig_rr is not None:
                sys.modules["run_rapidock"] = orig_rr
            else:
                sys.modules.pop("run_rapidock", None)

        mock_torch.manual_seed.assert_called_once_with(42)
        mock_cuda_mod.manual_seed_all.assert_not_called()

    def test_seed_everything_with_cuda(self) -> None:
        """_seed_everything calls cuda.manual_seed_all when CUDA is available."""
        import sys
        import types

        mock_torch = types.ModuleType("torch")
        mock_cuda_mod = types.ModuleType("torch.cuda")
        mock_cuda_mod.is_available = mock.MagicMock(return_value=True)
        mock_cuda_mod.manual_seed_all = mock.MagicMock()
        mock_torch.cuda = mock_cuda_mod
        mock_torch.manual_seed = mock.MagicMock()
        mock_torch.backends = types.ModuleType("torch.backends")
        mock_torch.backends.mps = types.SimpleNamespace(is_available=lambda: False)

        mock_np = types.ModuleType("numpy")
        mock_np_random = types.ModuleType("numpy.random")
        mock_np_random.seed = mock.MagicMock()
        mock_np.random = mock_np_random

        shim_path = Path(__file__).parent.parent / "src" / "hybridock_pep" / "sampling"
        sys.path.insert(0, str(shim_path))
        orig_torch = sys.modules.pop("torch", None)
        orig_np = sys.modules.pop("numpy", None)
        orig_rr = sys.modules.pop("run_rapidock", None)

        sys.modules["torch"] = mock_torch
        sys.modules["numpy"] = mock_np
        sys.modules["numpy.random"] = mock_np_random
        try:
            import run_rapidock as rr  # noqa: PLC0415
            import importlib; importlib.reload(rr)
            rr._seed_everything(99)
        except ImportError:
            pytest.skip("run_rapidock not importable — OK in isolation")
            return
        finally:
            sys.path.pop(0)
            if orig_torch is not None:
                sys.modules["torch"] = orig_torch
            else:
                sys.modules.pop("torch", None)
            if orig_np is not None:
                sys.modules["numpy"] = orig_np
            else:
                sys.modules.pop("numpy", None)
            if orig_rr is not None:
                sys.modules["run_rapidock"] = orig_rr
            else:
                sys.modules.pop("run_rapidock", None)

        mock_torch.manual_seed.assert_called_once_with(99)
        mock_cuda_mod.manual_seed_all.assert_called_once_with(99)


class TestComputeBatchSize:
    """_compute_batch_size — regression coverage for the MPS batching fix.

    Diffusion sampling batches all requested poses into one (or a few)
    forward passes per denoising step. This wrapper used to hardcode
    batch_size=4 regardless of --n-samples, forcing e.g. a 10-pose run into
    3 separate MPS forward passes per step instead of 1 — profiled on an M3
    (8-core GPU) at a 1.84x wall-clock cost (MPS kernel dispatch carries
    real fixed overhead per call: https://github.com/pytorch/pytorch/issues/122123).
    The cap is a *memory* limit, not a throughput one, and it is derived
    from physical RAM rather than fixed. A fixed cap of 32 was measured at
    398s for --n-samples 100 on a 16 GB M3 versus 62s at batch 16 — 6.4x
    slower, purely from swapping (+4.8 GB), and worse than the batch of 4
    this wrapper originally hardcoded. Batching past the RAM budget does not
    degrade gracefully; it falls off a cliff.
    """

    def _import_run_rapidock(self):
        import sys

        shim_path = Path(__file__).parent.parent / "src" / "hybridock_pep" / "sampling"
        sys.path.insert(0, str(shim_path))
        orig_rr = sys.modules.pop("run_rapidock", None)
        try:
            import run_rapidock as rr  # noqa: PLC0415
            import importlib
            importlib.reload(rr)
            return rr
        finally:
            sys.path.pop(0)
            if orig_rr is not None:
                import sys as _sys
                _sys.modules["run_rapidock"] = orig_rr

    def test_small_n_uses_single_batch(self) -> None:
        """A typical exploratory run gets one batch — the original speedup
        fix: previously this would have been split into ceil(n/4) separate
        MPS forward passes per diffusion step."""
        rr = self._import_run_rapidock()
        assert rr._compute_batch_size(1) == 1
        assert rr._compute_batch_size(10) == 10

    def test_large_n_is_capped(self) -> None:
        """A large production run is capped, not batched unboundedly."""
        rr = self._import_run_rapidock()
        cap = rr._default_batch_cap()
        assert rr._compute_batch_size(100) == cap
        assert rr._compute_batch_size(1000) == cap
        assert cap < 100

    def test_cap_is_derived_from_physical_ram(self) -> None:
        """The cap tracks RAM and deliberately sits just below the measured
        optimum, because overshooting costs 640% and undershooting ~15%."""
        rr = self._import_run_rapidock()
        orig = rr._total_ram_gb
        try:
            rr._total_ram_gb = lambda: 16.0
            assert rr._default_batch_cap() == 14
            rr._total_ram_gb = lambda: 8.0
            assert rr._default_batch_cap() == 7
            rr._total_ram_gb = lambda: 128.0
            assert rr._default_batch_cap() == 32   # absolute ceiling
            rr._total_ram_gb = lambda: 1.0
            assert rr._default_batch_cap() == 2    # floor, never 0
        finally:
            rr._total_ram_gb = orig

    def test_env_override_wins(self) -> None:
        """HYBRIDOCK_RAPIDOCK_BATCH is the documented escape hatch."""
        import os

        rr = self._import_run_rapidock()
        orig = os.environ.get("HYBRIDOCK_RAPIDOCK_BATCH")
        try:
            os.environ["HYBRIDOCK_RAPIDOCK_BATCH"] = "8"
            assert rr._compute_batch_size(100) == 8
            assert rr._compute_batch_size(4) == 4      # still bounded by n
            os.environ["HYBRIDOCK_RAPIDOCK_BATCH"] = "garbage"
            assert rr._compute_batch_size(100) == rr._default_batch_cap()
            os.environ["HYBRIDOCK_RAPIDOCK_BATCH"] = "0"
            assert rr._compute_batch_size(100) == rr._default_batch_cap()
        finally:
            if orig is None:
                os.environ.pop("HYBRIDOCK_RAPIDOCK_BATCH", None)
            else:
                os.environ["HYBRIDOCK_RAPIDOCK_BATCH"] = orig

    def test_custom_cap_respected(self) -> None:
        rr = self._import_run_rapidock()
        assert rr._compute_batch_size(50, cap=16) == 16
        assert rr._compute_batch_size(10, cap=16) == 10
