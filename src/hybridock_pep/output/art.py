"""Ambient ASCII art for long, silent waits — small, optional, never load-bearing.

Stage 1 (RAPiDock sampling) is the one part of a `dock` run that can sit
silent for 2+ minutes with no output at default verbosity — exactly the
"is it frozen?" gap :mod:`hybridock_pep.output.progress` was written to
close. This module gives that wait something to look at: a short rotating
gallery of science-themed ASCII art, plus a couple of easter eggs.

Every function here is decorative only:

- Nothing in this module may influence scoring, timing-sensitive control
  flow, or anything written to ``run_metadata.json`` — it only ever writes
  bytes to a stream.
- Every entry point degrades to a no-op off a TTY, under ``HYBRIDOCK_NO_ART``,
  or on any internal error — the same "must not break a run" rule
  :mod:`hybridock_pep.output.progress` follows.
"""
from __future__ import annotations

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

_DNA_HELIX = r"""
        A┈┈┈┈┈T
         \    /
          \  /
    G┈┈┈┈┈┈╳┈┈┈┈┈┈C
          /  \
         /    \
        C      G
         \    /
          \  /
    T┈┈┈┈┈┈╳┈┈┈┈┈┈A
          /  \
         /    \
        G      C
"""

_ALPHA_HELIX = r"""
   N-term
     @
         o
          @
       o
   @
o
 @
     o
         @
          o
       @
   o
@
 o
           C-term
   (one turn ~= 3.6 residues)
"""

_BINDING_POCKET = r"""
 ________________________
/                        \
|                        |
|  key  ⇢   ⬡   ⇠  lock  |
|                        |
|   peptide  ⇢  pocket   |
|   receptor ⇢  pocket   |
|                        |
 \________________________/
"""

_DIFFUSION = r"""
        .   .    .
      .   ·   ·    .
    .   ·  ·  ·   ·   .
      ·  ·  ○  ·  ·
    .   ·  ·  ·   ·   .
      .   ·   ·    .
        .   .    .
      noise  ->  pose
"""

#: (caption, art) pairs, rotated in order during a long wait.
GALLERY: tuple[tuple[str, str], ...] = (
    ("DNA — double helix", _DNA_HELIX),
    ("Peptide backbone — one alpha-helix turn", _ALPHA_HELIX),
    ("Induced fit — key, lock, pocket", _BINDING_POCKET),
    ("Stage 1 — noise to pose", _DIFFUSION),
)


def gallery_piece(index: int) -> tuple[str, str]:
    """The ``index``-th gallery piece, cycling."""
    return GALLERY[index % len(GALLERY)]


def write_gallery_piece(stream: TextIO, index: int, indent: str = "   ") -> None:
    """Write one indented gallery piece to ``stream``. Never raises."""
    try:
        title, art = gallery_piece(index)
        stream.write(f"\n{indent}· {title}\n")
        for line in art.strip("\n").splitlines():
            stream.write(f"{indent}{line}\n")
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
