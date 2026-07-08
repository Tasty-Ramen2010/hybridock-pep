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
import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


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
