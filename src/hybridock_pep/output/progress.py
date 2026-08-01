"""Lightweight, user-facing pipeline progress reporter.

Prints clean, plain-language stage banners ("▶ Generating poses…") so a `hybridock-pep dock` run reads like a
pipeline instead of a wall of DEBUG logs. TTY-aware: live ✓/elapsed lines on a terminal, plain one-line-per-stage
when piped to a file. tqdm drives per-item bars for long loops when available and on a TTY; otherwise it degrades
to a silent pass-through. Never raises — progress reporting must not break a run.

Sequential API (fits the driver's linear stages with minimal edits):
    prog = PipelineProgress(enabled=..., total=6)
    prog.step("Generating poses"); ...work...
    prog.step("Scoring poses");    ...work...
    prog.finish()
Each `step()` closes the previous stage with a ✓ and elapsed time.
"""
from __future__ import annotations
import contextlib
import os
import sys
import threading
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


@contextlib.contextmanager
def suppress_native_stdout(enabled: bool = True):
    """Silence unconditional C/C++-extension stdout writes (fd 1) for the block.

    AutoDock Vina's Python bindings print phase banners ("Computing Vina grid
    ... done.", "Performing local search ... done.") directly from the C++
    layer regardless of the `verbosity=` constructor arg — that's a real
    limitation of the bindings, not something a Python-level log level or
    ``verbosity=0`` can reach. Redirecting the raw file descriptor is the only
    way to catch it.

    Only touches fd 1 (stdout); a raised exception's traceback still goes to
    fd 2 (stderr) and is visible. No-ops safely if stdout has no real file
    descriptor (piped/captured stdout under pytest, some CI setups) — matches
    this module's "progress reporting must not break a run" rule.
    """
    if not enabled:
        yield
        return
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    try:
        saved_fd = os.dup(stdout_fd)
    except OSError:
        yield
        return
    try:
        with open(os.devnull, "w") as devnull:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            os.dup2(devnull.fileno(), stdout_fd)
            try:
                yield
            finally:
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                os.dup2(saved_fd, stdout_fd)
    finally:
        os.close(saved_fd)


class PipelineProgress:
    def __init__(self, enabled: bool = True, total: int = 0, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled and self.stream is not None
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._total = total
        self._n = 0
        self._open: str | None = None
        self._t: float = 0.0

    def _write(self, s: str) -> None:
        try:
            self.stream.write(s)
            self.stream.flush()
        except Exception:
            pass

    def _close(self, ok: bool = True) -> None:
        if self._open is not None:
            mark = "✓" if ok else "✗"
            self._write(f"  {mark} {self._open}  ({time.time() - self._t:.0f}s)\n")
            self._open = None

    def step(self, label: str) -> None:
        """Announce a new stage (closing the previous one with a ✓)."""
        if not self.enabled:
            return
        self._close(ok=True)
        self._n += 1
        tag = f"[{self._n}/{self._total}] " if self._total else ""
        self._write(f"▶ {tag}{label}…\n")
        self._open = label
        self._t = time.time()

    def note(self, msg: str) -> None:
        """A sub-line under the current stage (e.g. a detail the user should see)."""
        if self.enabled:
            self._write(f"   • {msg}\n")

    def bar(self, iterable: Iterable[T], label: str, total: int | None = None) -> Iterator[T]:
        """Per-item progress for a loop. Falls back to plain iteration off-TTY or if tqdm is missing."""
        if not self.enabled or not self.tty:
            yield from iterable
            return
        try:
            from tqdm import tqdm  # noqa: PLC0415
            yield from tqdm(iterable, desc=f"   {label}", total=total, leave=False,
                            file=self.stream, ncols=72)
        except Exception:
            yield from iterable

    @contextlib.contextmanager
    def heartbeat(self, interval_s: float = 15.0):
        """Tick an in-place elapsed-time indicator while a long, silent call
        runs under the current stage (e.g. the RAPiDock sampling subprocess,
        which can run 2+ minutes with no output at default verbosity). TTY-only
        so it never spams a log file; a plain step()/note() line is enough
        there. No-op if disabled. Never lets a heartbeat-thread problem break
        the wrapped call — same "must not break a run" rule as the rest of
        this module.
        """
        if not self.enabled or not self.tty:
            yield
            return
        stop = threading.Event()
        start = time.time()

        def _tick() -> None:
            while not stop.wait(interval_s):
                elapsed = time.time() - start
                self._write(f"\r   … still working ({elapsed:.0f}s elapsed)")

        try:
            t = threading.Thread(target=_tick, daemon=True)
            t.start()
        except Exception:
            yield
            return
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=1.0)
            # Clear the heartbeat line so it doesn't mangle the next ✓/▶ line.
            self._write("\r" + " " * 40 + "\r")

    def finish(self) -> None:
        self._close(ok=True)


# Simplified plain-language stage labels — the vocabulary the user sees.
LABELS = {
    "sample": "Generating poses",
    "prep": "Preparing receptor & ligands",
    "score": "Scoring poses",
    "cluster": "Clustering poses",
    "refine": "Refining top poses (MM-GBSA + entropy)",
    "charged": "Charged-residue correction",
    "rank": "Final ranking & ΔG",
    "write": "Writing results",
}
