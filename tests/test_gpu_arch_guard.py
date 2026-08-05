"""Stage 1 must run on a GPU newer than the PyTorch wheel that was installed.

Found on a DGX Spark: the GB10 is sm_121, while torch 2.7.0+cu128 reports
`get_arch_list() == ['sm_90', 'sm_100', 'sm_120']`. Everything compiled ahead of
time (cuBLAS, cuDNN, eager kernels) works, but the TorchScript TensorExpr fuser
calls NVRTC at runtime with `-arch=sm_121`, which the shipped toolkit rejects:

    nvrtc: error: invalid value for --gpu-architecture (-arch)

RAPiDock swallowed that into "produced 0 poses", so docking failed on a machine
whose install and smoke test both passed. Disabling GPU fusion fixes it, and
must happen *only* on hardware the wheel cannot target — otherwise every other
machine loses the fuser for nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SHIM = REPO / "src" / "hybridock_pep" / "sampling" / "run_rapidock.py"


def _load_shim():
    """Import run_rapidock.py by path.

    It is written for the rapidock env's interpreter and imports torch lazily
    inside functions, so it loads fine in score-env.
    """
    spec = importlib.util.spec_from_file_location("_run_rapidock_under_test", SHIM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeC:
    def __init__(self):
        self.calls = {}

    def _jit_set_texpr_fuser_enabled(self, v):
        self.calls["texpr"] = v

    def _jit_override_can_fuse_on_gpu(self, v):
        self.calls["gpu_fuse"] = v

    def _jit_set_nvfuser_enabled(self, v):
        self.calls["nvfuser"] = v

    def _jit_set_profiling_executor(self, v):
        self.calls["profiling"] = v


class _FakeTorch:
    def __init__(self, capability, arch_list):
        self._cap = capability
        self._arch = arch_list
        self._C = _FakeC()

        outer = self

        class _Cuda:
            @staticmethod
            def get_device_capability(idx):
                return outer._cap

            @staticmethod
            def get_arch_list():
                return outer._arch

        self.cuda = _Cuda()


class TestArchGuard:
    def test_disables_fusion_when_gpu_is_newer_than_the_wheel(self, capsys):
        """The GB10 case, exactly as measured."""
        mod = _load_shim()
        torch = _FakeTorch((12, 1), ["sm_90", "sm_100", "sm_120"])

        assert mod._disable_jit_fusion_if_arch_unsupported(torch) is True
        assert torch._C.calls["texpr"] is False
        assert torch._C.calls["gpu_fuse"] is False
        # The user must be told why sampling is slower than expected.
        err = capsys.readouterr().err
        assert "sm_121" in err and "fusion" in err.lower()

    def test_leaves_fusion_alone_on_a_supported_gpu(self):
        """An RTX 5070 (sm_120) against the same wheel must keep the fuser."""
        mod = _load_shim()
        torch = _FakeTorch((12, 0), ["sm_90", "sm_100", "sm_120"])

        assert mod._disable_jit_fusion_if_arch_unsupported(torch) is False
        assert torch._C.calls == {}

    def test_ignores_rocm(self):
        """ROCm reports gfx* targets and has no NVRTC; nothing to guard against."""
        mod = _load_shim()
        torch = _FakeTorch((9, 4), ["gfx90a", "gfx942"])

        assert mod._disable_jit_fusion_if_arch_unsupported(torch) is False
        assert torch._C.calls == {}

    def test_survives_a_torch_build_without_the_query(self):
        """Never let a probe break sampling on an unusual build."""
        mod = _load_shim()

        class _Broken:
            class cuda:
                @staticmethod
                def get_device_capability(idx):
                    raise RuntimeError("no CUDA context")

                @staticmethod
                def get_arch_list():
                    return []

        assert mod._disable_jit_fusion_if_arch_unsupported(_Broken) is False


class TestMetalE3nnHook:
    """metal-e3nn (github.com/Tasty-Ramen2010/metal-e3nn) is a separate,
    standalone project — never a hard dependency here. This is purely
    additive: absent, disabled, or broken must all fall through to stock
    e3nn silently, and only an actually-successful patch_() call earns the
    "+ metal-e3nn" label suffix on the MPS backend line."""

    def test_absent_is_a_silent_noop(self, monkeypatch):
        import sys

        mod = _load_shim()
        monkeypatch.delenv("METAL_E3NN_DISABLE", raising=False)
        monkeypatch.setitem(sys.modules, "metal_e3nn", None)  # simulate ImportError
        assert mod._try_patch_metal_e3nn() == ""

    def test_disabled_via_env_var_even_if_installed(self, monkeypatch):
        import sys

        mod = _load_shim()

        class _FakeMetalE3nn:
            patched = False

            def patch_(self):
                self.patched = True

        fake = _FakeMetalE3nn()
        monkeypatch.setitem(sys.modules, "metal_e3nn", fake)
        monkeypatch.setenv("METAL_E3NN_DISABLE", "1")
        assert mod._try_patch_metal_e3nn() == ""
        assert fake.patched is False

    def test_applies_and_labels_when_installed(self, monkeypatch):
        import sys

        mod = _load_shim()
        monkeypatch.delenv("METAL_E3NN_DISABLE", raising=False)

        class _FakeMetalE3nn:
            patched = False

            def patch_(self):
                self.patched = True

        fake = _FakeMetalE3nn()
        monkeypatch.setitem(sys.modules, "metal_e3nn", fake)
        assert mod._try_patch_metal_e3nn() == " + metal-e3nn"
        assert fake.patched is True

    def test_a_broken_patch_call_falls_through_silently(self, monkeypatch):
        import sys

        mod = _load_shim()
        monkeypatch.delenv("METAL_E3NN_DISABLE", raising=False)

        class _BrokenMetalE3nn:
            def patch_(self):
                raise RuntimeError("incompatible e3nn version")

        monkeypatch.setitem(sys.modules, "metal_e3nn", _BrokenMetalE3nn())
        assert mod._try_patch_metal_e3nn() == ""


class TestZeroPoseErrorQuotesStderr:
    """"Check stderr logs above" was useless: stderr is logged at DEBUG, so at
    the default level there was nothing above to check."""

    def test_stream_stderr_fills_the_sink(self):
        from collections import deque
        from hybridock_pep.sampling import rapidock_runner as rr

        class _Pipe:
            def __init__(self, lines):
                self._it = iter(lines)

            def readline(self):
                return next(self._it, b"")

        sink: deque = deque(maxlen=rr._STDERR_TAIL_LINES)
        rr._stream_stderr(_Pipe([b"nvrtc: error: invalid value\n", b"\n", b"boom\n"]), sink)

        assert list(sink) == ["nvrtc: error: invalid value", "boom"]

    def test_sink_is_bounded(self):
        from collections import deque
        from hybridock_pep.sampling import rapidock_runner as rr

        class _Pipe:
            def __init__(self, n):
                self._it = iter([b"line %d\n" % i for i in range(n)])

            def readline(self):
                return next(self._it, b"")

        sink: deque = deque(maxlen=rr._STDERR_TAIL_LINES)
        rr._stream_stderr(_Pipe(5000), sink)

        assert len(sink) == rr._STDERR_TAIL_LINES
        assert sink[-1] == "line 4999", "must keep the *tail*, where the traceback is"
