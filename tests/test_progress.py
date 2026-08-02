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


# ---------------------------------------------------------------------------
# ASCII bars and the ambient progress sink
# ---------------------------------------------------------------------------
# `bar()` used to be implemented on top of tqdm, which is NOT a dependency of
# score-env. So on every standard install it silently fell through to plain
# iteration and drew nothing: a progress system that reported no progress,
# which is exactly the "is it frozen?" experience it exists to prevent.
# Nothing caught that, because nothing tested it. These do.

from hybridock_pep.output import progress as P  # noqa: E402


class _FakePipeStream(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_ambient_leak():
    """The sink is module-global; never let one test's reporter reach another."""
    P.set_active(None)
    yield
    P.set_active(None)


class TestBarDrawsWithoutTqdm:
    def test_bar_writes_progress_on_a_tty(self) -> None:
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("Scoring")
        assert list(prog.bar(range(5), "poses")) == [0, 1, 2, 3, 4]
        text = out.getvalue()
        assert "%" in text and "poses" in text
        assert "5/5" in text, "the bar never reached completion"

    def test_bar_is_silent_off_tty(self) -> None:
        out = _FakePipeStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("Scoring")
        before = out.getvalue()
        assert list(prog.bar(range(5), "poses")) == [0, 1, 2, 3, 4]
        assert out.getvalue() == before, "a piped log must not receive bar output"

    def test_bar_yields_everything_for_a_sizeless_iterable(self) -> None:
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)

        def gen():
            yield from range(4)

        assert list(prog.bar(gen(), "items")) == [0, 1, 2, 3]

    def test_render_bar_clamps_out_of_range_counts(self) -> None:
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("x")
        prog.render_bar(999, 10, "poses")
        prog.render_bar(-5, 10, "poses")
        text = out.getvalue()
        assert "100.0%" in text and "0.0%" in text

    def test_zero_total_does_not_divide_by_zero(self) -> None:
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("x")
        prog.render_bar(0, 0, "poses")


class TestAmbientSink:
    def test_tick_is_a_noop_with_no_reporter(self) -> None:
        P.tick(1, 10, "poses")
        P.clear()

    def test_tick_draws_through_the_active_reporter(self) -> None:
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("Scoring")
        P.set_active(prog)
        P.tick(3, 10, "poses refined")
        assert "poses refined" in out.getvalue()
        assert "30.0%" in out.getvalue()

    def test_track_iterates_identically_with_or_without_a_reporter(self) -> None:
        assert list(P.track(range(4), "x")) == [0, 1, 2, 3]
        out = _FakeTTYStream()
        prog = PipelineProgress(enabled=True, total=1, stream=out)
        prog.step("s")
        P.set_active(prog)
        assert list(P.track(range(4), "x")) == [0, 1, 2, 3]

    def test_a_broken_reporter_cannot_break_the_run(self) -> None:
        """Progress reporting must never be able to fail real work."""
        class Exploding:
            def render_bar(self, *a, **kw):
                raise RuntimeError("boom")

            def bar(self, iterable, label, total=None):
                raise RuntimeError("boom")

            def clear_line(self):
                raise RuntimeError("boom")

        P.set_active(Exploding())
        P.tick(1, 2, "x")
        P.clear()
        assert list(P.track(range(3), "x")) == [0, 1, 2]


class TestBanner:
    def test_banner_writes_plain_ascii_art(self) -> None:
        out = io.StringIO()
        P.banner(stream=out)
        text = out.getvalue()
        assert len(text.splitlines()) > 3
        assert text.isascii(), "banner must be plain ASCII so it renders anywhere"

    def test_banner_never_raises_on_a_broken_stream(self) -> None:
        class Broken:
            def write(self, s):
                raise OSError("no")

            def flush(self):
                pass

        P.banner(stream=Broken())


class TestSamplingProgressProtocol:
    """The RAPiDock shim and the parent must agree on the wire format."""

    def test_parent_and_shim_share_one_prefix_constant(self) -> None:
        from hybridock_pep.sampling.run_rapidock import PROGRESS_PREFIX
        assert PROGRESS_PREFIX.startswith("[[") and PROGRESS_PREFIX.endswith("]]")

    def test_progress_line_round_trips(self) -> None:
        from hybridock_pep.sampling.run_rapidock import PROGRESS_PREFIX
        line = "%s %d %d" % (PROGRESS_PREFIX, 7, 112)
        assert line.startswith(PROGRESS_PREFIX)
        done_s, total_s = line[len(PROGRESS_PREFIX):].split()
        assert (int(done_s), int(total_s)) == (7, 112)
