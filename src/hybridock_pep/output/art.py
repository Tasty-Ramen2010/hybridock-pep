"""Ambient ASCII art for long, silent waits — small, optional, never load-bearing.

Stage 1 (RAPiDock sampling) is the one part of a `dock` run that can sit
silent for 2+ minutes with no output at default verbosity — exactly the
"is it frozen?" gap :mod:`hybridock_pep.output.progress` was written to
close. This module gives that wait something to look at: a short rotating
gallery of science-themed ASCII art (most of it lightly animated — the DNA
strand rotates, the alpha helix turns, the diffusion piece actually shows
noise resolving into a pose), plus a couple of easter eggs.

Every function here is decorative only:

- Nothing in this module may influence scoring, timing-sensitive control
  flow, or anything written to ``run_metadata.json`` — it only ever writes
  bytes to a stream.
- Every entry point degrades to a no-op off a TTY, under ``HYBRIDOCK_NO_ART``,
  or on any internal error — the same "must not break a run" rule
  :mod:`hybridock_pep.output.progress` follows.
- Frame data is generated once at import time from plain math (a fixed
  random seed for the diffusion piece), not hand-typed per frame, so every
  frame of an animation shares the same line count and column alignment.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import TextIO

# ---------------------------------------------------------------------------
# Opt-out / TTY gate
# ---------------------------------------------------------------------------


def art_enabled(stream: TextIO | None = None) -> bool:
    """True when ambient art should render.

    Requires a real TTY (matches every other decorative element in this
    package) and respects ``HYBRIDOCK_NO_ART=1`` for anyone who'd rather
    just watch the elapsed-time counter.
    """
    if os.environ.get("HYBRIDOCK_NO_ART"):
        return False
    s = stream if stream is not None else sys.stderr
    try:
        return bool(s.isatty())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gallery — rotates during a long wait (Stage 1 sampling)
# ---------------------------------------------------------------------------

_DNA_BASE = (
    "        A\u2508\u2508\u2508\u2508\u2508T\n"
    "         \\    /\n"
    "          \\  /\n"
    "    G\u2508\u2508\u2508\u2508\u2508\u2508\u2573\u2508\u2508\u2508\u2508\u2508\u2508\u2508C\n"
    "          /  \\\n"
    "         /    \\\n"
    "        C      G\n"
    "         \\    /\n"
    "          \\  /\n"
    "    T\u2508\u2508\u2508\u2508\u2508\u2508\u2573\u2508\u2508\u2508\u2508\u2508\u2508\u2508A\n"
    "          /  \\\n"
    "         /    \\\n"
    "        G      C"
)


def _dna_frames() -> tuple[str, ...]:
    """A believable spin: base-pair view -> edge-on -> 180 deg -> edge-on."""
    edge_on = "".join("\u2022" if c in "ATGC" else c for c in _DNA_BASE)
    swapped = _DNA_BASE.translate(str.maketrans({"A": "T", "T": "A", "G": "C", "C": "G"}))
    return (_DNA_BASE, edge_on, swapped, edge_on)


def _alpha_helix_frames() -> tuple[str, ...]:
    """One coil, phase-shifted a quarter turn per frame — a rotating sine wave."""
    frames = []
    width = 10
    for phase in (0.0, 1.1, 2.2, 3.3):
        lines = ["   N-term"]
        for i in range(14):
            x = int(round(width / 2 * (1 + math.sin(i * 0.9 + phase))))
            ch = "@" if i % 2 == 0 else "o"
            lines.append(" " * x + ch)
        lines.append("           C-term")
        frames.append("\n".join(lines))
    return tuple(frames)


_BINDING_POCKET = (
    " ________________________\n"
    "/                        \\\n"
    "|                        |\n"
    "|  key  \u21e2   \u2b21   \u21e0  lock  |\n"
    "|                        |\n"
    "|   peptide  \u21e2  pocket   |\n"
    "|   receptor \u21e2  pocket   |\n"
    "|                        |\n"
    " \\________________________/"
)


def _diffusion_frames() -> tuple[str, ...]:
    """Noise literally resolving into a pose — points move from scattered
    starting positions to a target ring shape over four interpolation steps,
    which is what a diffusion model (RAPiDock's Stage 1) actually does.
    Fixed seed: deterministic, testable, identical on every machine.
    """
    cols, rows, n = 23, 7, 14
    cx, cy = cols / 2, rows / 2
    target = [
        (cx + 7.5 * math.cos(2 * math.pi * i / n), cy + 2.4 * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    rng = random.Random(7)
    noise = [(rng.uniform(1, cols - 2), rng.uniform(1, rows - 2)) for _ in range(n)]

    def render(t: float, glyph: str) -> str:
        grid = [[" "] * cols for _ in range(rows)]
        for (nx, ny), (tx, ty) in zip(noise, target):
            x = nx + (tx - nx) * t
            y = ny + (ty - ny) * t
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rows and 0 <= xi < cols:
                grid[yi][xi] = glyph
        return "\n".join("".join(row) for row in grid)

    steps = [(0.0, "\u00b7"), (0.35, "\u00b7"), (0.7, "o"), (1.0, "\u25cf")]
    caption = "      noise " + "".join("\u00b7" for _ in range(3))
    frames = [render(t, g) + f"\n{caption}" for t, g in steps]
    frames[-1] = frames[-1].rsplit("\n", 1)[0] + "\n      noise  ->  pose"
    return tuple(frames)


#: (caption, frames) pairs, rotated in order during a long wait. Single-frame
#: entries (a tuple of length 1) just render once — not every piece needs to
#: animate to earn a place in the gallery.
GALLERY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DNA — double helix", _dna_frames()),
    ("Peptide backbone — one alpha-helix turn", _alpha_helix_frames()),
    ("Induced fit — key, lock, pocket", (_BINDING_POCKET,)),
    ("Stage 1 — noise resolving into a pose", _diffusion_frames()),
)


def gallery_piece(index: int) -> tuple[str, tuple[str, ...]]:
    """The ``index``-th gallery piece (title, frames), cycling."""
    return GALLERY[index % len(GALLERY)]


def animate_gallery_piece(
    stream: TextIO, index: int, indent: str = "   ", frame_delay: float = 0.28
) -> None:
    """Write one gallery piece to ``stream``, animating in place if it has
    more than one frame, then leave the final frame on screen.

    TTY-only in spirit (the caller already gates on that — see
    :meth:`hybridock_pep.output.progress.PipelineProgress.heartbeat`): uses
    plain ANSI cursor-up + line-clear to redraw, which only makes sense on a
    real terminal. Never raises — a broken animation must not break a run.
    """
    try:
        title, frames = gallery_piece(index)
        first = frames[0].splitlines()
        stream.write(f"\n{indent}\u00b7 {title}\n")
        for line in first:
            stream.write(f"{indent}{line}\n")
        stream.flush()
        n = len(first)
        for frame in frames[1:]:
            time.sleep(frame_delay)
            lines = frame.splitlines()
            stream.write(f"\x1b[{n}A")
            for line in lines:
                stream.write(f"\x1b[2K{indent}{line}\n")
            stream.flush()
            n = len(lines)
        stream.write("\n")
        stream.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ambient spinner — a tiny chain-forming animation for the in-place ticker
# ---------------------------------------------------------------------------

_SPIN_CHAIN = ["o", "o-o", "o-o-o", "o-o-o-o", "o-o-o", "o-o", "o"]


def spin_frame(t: float | None = None) -> str:
    """One frame of a small ambient 'peptide chain forming' animation.

    Deterministic in ``t`` (seconds) so it's testable without a clock; the
    caller passes ``time.time()`` for real use.
    """
    now = t if t is not None else time.time()
    return _SPIN_CHAIN[int(now * 3) % len(_SPIN_CHAIN)]


# ---------------------------------------------------------------------------
# Easter eggs
# ---------------------------------------------------------------------------

#: The parent iGEM project's real target peptide (see CLAUDE.md / README) —
#: a malaria rapid-diagnostic candidate against PfLDH.
PARENT_PROJECT_PEPTIDE = "LISDAELEAIFEADC"

_MOSQUITO = r"""
              __
             /  \
        ,---(    )---.
       /     \__/     \
      (  o          o  )
       \      /\      /
        \    /  \    /
    -----)--(    )--(-----
        /    \  /    \
       (      \/      )
        \_____/\_____/
      PfLDH · Plasmodium falciparum
"""

_IGEM_EGG = r"""
        .---.
       /     \
      | () () |      i . G . E . M
       \  ~  /       synthetic biology,
        |||||        one plasmid at a time
       /_____\
    ~~~~~~~~~~~~~
      Denmark High School . Dry Lab
"""

_SHINY = r"""
      *  .  *    ·  .  *
   .    *   you found a shiny pose!   .
      ·         (this is rare)        *
   *  .    keep this run's seed     .
"""


def easter_egg_for_peptide(sequence: str) -> str | None:
    """A special art block for a small, fixed set of trigger sequences.

    Case-insensitive, whitespace-stripped. Two triggers, both harmless if
    you don't know they exist:

    - The parent iGEM project's real peptide (a nod, not a feature).
    - ``IGEM`` — which is also, conveniently, a real valid tetrapeptide
      (Ile-Gly-Glu-Met), so it passes normal sequence validation on its
      own merits and isn't a special-cased input.

    Returns ``None`` for everything else.
    """
    if not sequence:
        return None
    seq = sequence.strip().upper()
    if seq == PARENT_PROJECT_PEPTIDE:
        return _MOSQUITO
    if seq == "IGEM":
        return _IGEM_EGG
    return None


def maybe_rare(rng: random.Random | None = None, chance: float = 0.05) -> str | None:
    """The rare bonus art with probability ``chance`` (default 1-in-20), else ``None``."""
    r = rng if rng is not None else random
    return _SHINY if r.random() < chance else None
