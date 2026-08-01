"""Tests for hybridock_pep.output.progress — suppress_native_stdout + PipelineProgress.heartbeat.

These back the default-mode output cleanup (WARNING-only logging + PipelineProgress as the
sole default UI): Vina's native C++ stdout prints ("Computing Vina grid ... done.") ignore
Python log levels entirely, so suppress_native_stdout is the only way to silence them; and
heartbeat() gives the long, silent RAPiDock-sampling stage a visible "still working" tick
instead of looking frozen.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time

import pytest

from hybridock_pep.output.progress import PipelineProgress, suppress_native_stdout


class _FakeTTYStream(io.StringIO):
    """StringIO reports isatty()=False by default; PipelineProgress checks it."""

    def isatty(self) -> bool:  # noqa: D102
        return True


class TestSuppressNativeStdout:
    # These two run the redirect in a real subprocess rather than under
    # pytest's own capfd fixture: capfd does its own fd-1 dup2 bookkeeping,
    # and nesting another low-level os.dup2 inside it produces false
    # failures that don't reproduce outside pytest (verified manually — the
    # exact same code correctly hides the write in a plain `python3 -c`
    # invocation). A subprocess sidesteps that interaction and is also a
    # closer match for the real scenario: a C extension (Vina) writing to
    # fd 1 from inside the process.
    def test_suppresses_raw_fd_write_when_enabled(self) -> None:
        """A raw os.write(1, ...) — the same path a C extension uses — must not
        reach the real stdout while the context is active."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import os\n"
                "from hybridock_pep.output.progress import suppress_native_stdout\n"
                "with suppress_native_stdout(enabled=True):\n"
                "    os.write(1, b'should not appear\\n')\n",
            ],
            capture_output=True, text=True,
        )
        assert "should not appear" not in result.stdout

    def test_stdout_restored_after_block(self) -> None:
        """Writes before/after the block are unaffected; fd 1 must be restored,
        not left pointed at devnull."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import os\n"
                "from hybridock_pep.output.progress import suppress_native_stdout\n"
                "with suppress_native_stdout(enabled=True):\n"
                "    os.write(1, b'hidden\\n')\n"
                "os.write(1, b'visible\\n')\n",
            ],
            capture_output=True, text=True,
        )
        assert "hidden" not in result.stdout
        assert "visible" in result.stdout

    def test_disabled_is_passthrough(self, capfd) -> None:
        with suppress_native_stdout(enabled=False):
            os.write(1, b"shown\n")
        out, _err = capfd.readouterr()
        assert "shown" in out

    def test_never_raises_without_real_fileno(self, monkeypatch) -> None:
        """Piped/captured stdout without a real file descriptor (some CI/pytest
        capture modes) must degrade to a no-op, not crash the caller."""

        class NoFilenoStream:
            def write(self, s):  # noqa: D102
                pass

            def flush(self):  # noqa: D102
                pass

            def fileno(self):  # noqa: D102
                raise AttributeError("no fileno")

        monkeypatch.setattr(sys, "stdout", NoFilenoStream())
        with suppress_native_stdout(enabled=True):
            pass  # must not raise

    def test_exception_inside_block_propagates_and_restores_stdout(self, capfd) -> None:
        with pytest.raises(ValueError, match="boom"):
            with suppress_native_stdout(enabled=True):
                raise ValueError("boom")
        os.write(1, b"still works\n")
        out, _err = capfd.readouterr()
        assert "still works" in out


class TestHeartbeat:
    def test_disabled_is_immediate_passthrough(self) -> None:
        prog = PipelineProgress(enabled=False, total=1, stream=_FakeTTYStream())
        with prog.heartbeat(interval_s=0.01):
            pass  # must not raise, must not hang

    def test_noop_when_stream_not_a_tty(self) -> None:
        prog = PipelineProgress(enabled=True, total=1, stream=io.StringIO())
        assert prog.tty is False
        with prog.heartbeat(interval_s=0.01):
            pass  # must not raise, must not hang

    def test_enabled_tty_runs_and_cleans_up(self) -> None:
        stream = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=stream)
        start = time.time()
        with prog.heartbeat(interval_s=0.02):
            time.sleep(0.06)  # long enough for at least one tick
        elapsed = time.time() - start
        assert elapsed < 2.0  # thread must not hang the exit past its join timeout

    def test_exception_inside_heartbeat_propagates(self) -> None:
        stream = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=stream)
        with pytest.raises(RuntimeError, match="boom"):
            with prog.heartbeat(interval_s=0.01):
                raise RuntimeError("boom")
