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

    def render_bar(self, done: int, total: int, label: str, *, width: int = 28) -> None:
        """Draw/redraw one in-place ASCII progress bar. TTY-only.

        Deliberately hand-rolled rather than tqdm: tqdm is not a dependency of
        score-env, so the previous tqdm-backed implementation silently degraded
        to no progress output at all on a standard install — which is exactly
        the "is it frozen?" experience this module exists to prevent. Plain
        ASCII also renders identically everywhere (SSH, Terminal.app, tmux,
        screen recordings) with no font or Unicode-width surprises.
        """
        if not self.enabled or not self.tty:
            return
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        pct = 100.0 * done / total
        elapsed = time.time() - self._t if self._t else 0.0
        eta = ""
        if done and done < total and elapsed > 1.0:
            eta = f"  ETA {self._fmt_time(elapsed / done * (total - done))}"
        self._write(f"\r   [{bar}] {pct:5.1f}%  {done}/{total} {label}{eta}   ")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(max(0, seconds))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

    def clear_line(self) -> None:
        """Erase an in-place bar so the next ✓/▶ line starts clean."""
        if self.enabled and self.tty:
            self._write("\r" + " " * 78 + "\r")

    def bar(self, iterable: Iterable[T], label: str, total: int | None = None) -> Iterator[T]:
        """Wrap a loop in an in-place ASCII progress bar.

        Degrades to plain iteration off-TTY (piped output, CI, log files) so a
        captured log never fills with control characters. Never raises: a
        progress bar must not be able to break a run.
        """
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total = None
        if not self.enabled or not self.tty or not total:
            yield from iterable
            return
        done = 0
        self.render_bar(0, total, label)
        try:
            for item in iterable:
                yield item
                done += 1
                self.render_bar(done, total, label)
        finally:
            self.clear_line()

    @contextlib.contextmanager
    def heartbeat(self, interval_s: float = 15.0, art_interval_s: float = 30.0):
        """Tick an in-place elapsed-time indicator while a long, silent call
        runs under the current stage (e.g. the RAPiDock sampling subprocess,
        which can run 2+ minutes with no output at default verbosity). TTY-only
        so it never spams a log file; a plain step()/note() line is enough
        there. No-op if disabled. Never lets a heartbeat-thread problem break
        the wrapped call — same "must not break a run" rule as the rest of
        this module.

        Every ``art_interval_s`` seconds the in-place ticker is joined by one
        piece from :mod:`hybridock_pep.output.art`'s gallery, printed on its
        own lines so a long real wait has something to look at beyond a
        growing number. Purely cosmetic — set ``HYBRIDOCK_NO_ART=1`` to get
        the plain elapsed-time ticker back.
        """
        if not self.enabled or not self.tty:
            yield
            return
        from hybridock_pep.output import art as _art  # noqa: PLC0415

        art_on = _art.art_enabled(self.stream)
        stop = threading.Event()
        start = time.time()
        shown = 0

        def _tick() -> None:
            nonlocal shown
            while not stop.wait(interval_s):
                elapsed = time.time() - start
                if art_on and elapsed >= (shown + 1) * art_interval_s:
                    self._write("\r" + " " * 40 + "\r")
                    _art.write_gallery_piece(self.stream, shown)
                    shown += 1
                spin = _art.spin_frame(elapsed) if art_on else ""
                suffix = f"  {spin}" if spin else ""
                self._write(f"\r   … still working ({elapsed:.0f}s elapsed){suffix}")

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


# ---------------------------------------------------------------------------
# Ambient progress sink
# ---------------------------------------------------------------------------
# The expensive loops (ligand prep, Vina scoring, minimization, MM-GBSA) live
# inside batch helpers several call-frames below the driver, and most of them
# are also called directly by tests and scripts. Threading a `prog` argument
# through every one of those signatures would be broad, churn-heavy, and would
# force every caller to care about progress reporting.
#
# Instead the driver publishes its reporter here once, and any module that
# wants to draw a bar asks for it. Unset (tests, library use) it is None and
# every call site is a no-op, so nothing has to be mocked.

_ACTIVE: "PipelineProgress | None" = None


def set_active(progress: "PipelineProgress | None") -> None:
    """Publish the reporter that batch helpers should draw on."""
    global _ACTIVE
    _ACTIVE = progress


def active() -> "PipelineProgress | None":
    """The current reporter, or None when progress reporting is off."""
    return _ACTIVE


def track(iterable: Iterable[T], label: str, total: int | None = None) -> Iterator[T]:
    """Draw a bar over `iterable` using the ambient reporter, if there is one.

    Safe everywhere: with no active reporter, off-TTY, or on any internal
    failure this is exactly plain iteration.
    """
    prog = active()
    if prog is None:
        yield from iterable
        return
    try:
        yield from prog.bar(iterable, label, total=total)
    except Exception:
        yield from iterable


def tick(done: int, total: int, label: str) -> None:
    """Redraw the ambient bar for code that owns its own loop counter."""
    prog = active()
    if prog is not None:
        try:
            prog.render_bar(done, total, label)
        except Exception:
            pass


def clear() -> None:
    """Erase the ambient bar. Call when a hand-driven `tick` loop finishes, so
    the trailing bar does not collide with the next ✓ or ▶ line."""
    prog = active()
    if prog is not None:
        try:
            prog.clear_line()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ASCII art
# ---------------------------------------------------------------------------

BANNER = r"""
  _   _       _          _ ____             _      ____
 | | | |_   _| |__  _ __(_)  _ \  ___   ___| | __ |  _ \ ___ _ __
 | |_| | | | | '_ \| '__| | | | |/ _ \ / __| |/ / | |_) / _ \ '_ \
 |  _  | |_| | |_) | |  | | |_| | (_) | (__|   <  |  __/  __/ |_) |
 |_| |_|\__, |_.__/|_|  |_|____/ \___/ \___|_|\_\ |_|   \___| .__/
        |___/                                               |_|
"""

BANNER_COMPACT = r"""
 +-+-+-+-+-+-+-+-+-+-+-+-+-+
  H y b r i D o c k - P e p
 +-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

TAGLINE = "peptide -> AI poses -> physics rescoring -> binding energy"


def banner(stream=None, subtitle: str = TAGLINE) -> None:
    """Return the wordmark sized to the terminal, and write it. Never raises.

    Uses ``shutil.get_terminal_size`` (not ``os.get_terminal_size``) to match
    the rest of the CLI: it honours the COLUMNS environment variable and has a
    fallback, so tests and piped output get a deterministic width instead of an
    OSError path.
    """
    import shutil  # noqa: PLC0415

    out = stream if stream is not None else sys.stderr
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    art = BANNER if cols >= 72 else BANNER_COMPACT
    try:
        out.write(art)
        if subtitle:
            out.write(f"  {subtitle}\n")
        out.write("\n")
        out.flush()
    except Exception:
        pass


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
