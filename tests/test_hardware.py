"""Unit tests for the centralized hardware/accelerator selection (hybridock_pep.hardware)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from hybridock_pep.hardware import cpu_threads, openmm_platform


def test_cpu_threads_positive() -> None:
    assert cpu_threads() >= 1


def test_cpu_threads_honors_env() -> None:
    with patch.dict("os.environ", {"OPENMM_CPU_THREADS": "3"}):
        assert cpu_threads() == 3


def test_cpu_threads_ignores_bad_env() -> None:
    with patch.dict("os.environ", {"OPENMM_CPU_THREADS": "garbage"}):
        assert cpu_threads() >= 1


def test_platform_priority_cuda_first() -> None:
    """CUDA is tried first and returned with mixed precision."""
    mock_openmm = MagicMock()
    cuda = MagicMock()
    mock_openmm.Platform.getPlatformByName.return_value = cuda
    with patch.dict("sys.modules", {"openmm": mock_openmm}):
        platform, props = openmm_platform(force_cpu=False)
    assert mock_openmm.Platform.getPlatformByName.call_args_list[0][0][0] == "CUDA"
    assert props["Precision"] == "mixed"


def test_platform_hip_before_opencl() -> None:
    """AMD HIP is tried before OpenCL (HIP is faster than OpenCL on AMD)."""
    mock_openmm = MagicMock()
    hip = MagicMock()
    hip.getName.return_value = "HIP"

    def side_effect(name):
        if name == "CUDA":
            raise RuntimeError("no CUDA")
        if name == "HIP":
            return hip
        raise RuntimeError("should not reach OpenCL")

    mock_openmm.Platform.getPlatformByName.side_effect = side_effect
    with patch.dict("sys.modules", {"openmm": mock_openmm}):
        platform, props = openmm_platform(force_cpu=False)
    assert platform is hip
    assert props["Precision"] == "mixed"


def test_force_cpu_returns_threaded_cpu() -> None:
    mock_openmm = MagicMock()
    cpu = MagicMock()
    mock_openmm.Platform.getPlatformByName.return_value = cpu
    with patch.dict("sys.modules", {"openmm": mock_openmm}):
        platform, props = openmm_platform(force_cpu=True)
    mock_openmm.Platform.getPlatformByName.assert_called_once_with("CPU")
    assert "Threads" in props and int(props["Threads"]) >= 1


def test_cpu_fallback_when_no_gpu() -> None:
    mock_openmm = MagicMock()
    cpu = MagicMock()
    cpu.getName.return_value = "CPU"

    def side_effect(name):
        if name in ("CUDA", "HIP", "OpenCL"):
            raise RuntimeError(f"no {name}")
        return cpu

    mock_openmm.Platform.getPlatformByName.side_effect = side_effect
    with patch.dict("sys.modules", {"openmm": mock_openmm}):
        platform, props = openmm_platform(force_cpu=False)
    assert platform is cpu
    assert "Threads" in props
