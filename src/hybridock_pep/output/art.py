"""Ambient ASCII art for long, silent waits — small, optional, never load-bearing.

Stage 1 (RAPiDock sampling) is the one part of a `dock` run that can sit
silent for 2+ minutes with no output at default verbosity — exactly the
"is it frozen?" gap :mod:`hybridock_pep.output.progress` was written to
close. This module gives that wait something to look at:

- A rotating gallery of science-themed ASCII animations, each one a real
  parametric model (a rotating double helix, a coiling alpha-helix ribbon,
  a peptide docking into a pocket, diffusion noise resolving into a pose) —
  not hand-typed frames swapped in and out. Every frame of a piece is
  generated from the same formula at a different phase/time step, which is
  what makes the motion read as continuous rotation/motion instead of a
  slideshow.
- A cycling "verb" spinner (``Diffusing…``, ``Helixing…``, ``Docking…`` — the
  same idea as a chat assistant's own "Thinking…"/"Pondering…" ticker,
  themed to this pipeline's wet-lab/dry-lab vocabulary) for the plain
  in-place heartbeat line.
- A couple of easter eggs.

Every function here is decorative only:

- Nothing in this module may influence scoring, timing-sensitive control
  flow, or anything written to ``run_metadata.json`` — it only ever writes
  bytes to a stream.
- Every entry point degrades to a no-op off a TTY, under ``HYBRIDOCK_NO_ART``,
  or on any internal error — the same "must not break a run" rule
  :mod:`hybridock_pep.output.progress` follows.
- Frame data is generated once at import time from plain math (fixed random
  seeds where randomness is involved), not hand-typed per frame, so every
  frame of an animation shares the same line count — required for the
  in-place cursor-redraw trick both the CLI and the TUI use to animate it.
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

#: The parent iGEM project's real, ready-to-use target peptide (see
#: CLAUDE.md / README) — a malaria rapid-diagnostic candidate against PfLDH.
#: Also the sequence the peptide-backbone gallery piece actually draws (see
#: ``_peptide_frames`` below) — the real molecule, not a generic placeholder.
#: (Superseded LISDAELEAIFEADC as the project's main/default sequence.)
PARENT_PROJECT_PEPTIDE = "LISAAALAAIFAAALAC"


def _dna_frames(n_frames: int = 8) -> tuple[str, ...]:
    """A double helix actually rotating around its own axis.

    One parametric model, not swapped characters: each of the two backbone
    strands traces ``x = center + amplitude * cos(angle)`` down the length of
    the molecule, phase-offset by pi from each other. ``angle``'s sign of
    ``sin`` gives which strand is nearer the viewer at that row, so the near
    strand draws solid (``●``) and the far strand draws faint (``·``) —
    that's what actually sells "rotating in depth" instead of "wiggling in
    place". Rungs (base pairs) fill the gap between the two strands on every
    row. Advancing ``phase`` by a fixed step per frame is the entire
    animation — no per-frame special-casing.
    """
    rows = 15
    turns = 1.55
    amp = 9.0
    center = 10
    width = center * 2 + 3
    frames = []
    for f in range(n_frames):
        phase = f * (2 * math.pi / n_frames)
        lines = []
        for r in range(rows):
            angle = phase + r * (turns * 2 * math.pi / rows)
            x1 = center + amp * math.cos(angle)
            x2 = center + amp * math.cos(angle + math.pi)
            depth1 = math.sin(angle)
            i1, i2 = int(round(x1)), int(round(x2))
            row = [" "] * width
            lo, hi = sorted((i1, i2))
            if hi - lo > 2:
                for x in range(lo + 1, hi):
                    row[x] = "\u2500"  # ─
            near, far = ("\u25cf", "\u00b7") if depth1 >= 0 else ("\u00b7", "\u25cf")
            row[i1], row[i2] = near, far
            lines.append("".join(row))
        frames.append("\n".join(lines))
    return tuple(frames)


# ---------------------------------------------------------------------------
# Peptide backbone — the actual parent-project sequence, one continuous
# chain, with a side chain per residue drawn from its real chemical class.
#
# Earlier versions of this piece tried an arbitrary generic helix, then a
# fictional 3-helix tertiary bundle. Both were wrong for the same reason: a
# short linear peptide with no crosslinks does not fold into a compact
# tertiary bundle — that's not what RAPiDock is actually generating. The
# honest picture is simpler: the real sequence, PARENT_PROJECT_PEPTIDE, as
# one helical chain, each side chain shaped by what that residue actually is.
# ---------------------------------------------------------------------------

#: Real amino-acid classes, keyed by one-letter code — decide the *shape* of
#: the drawn side chain, not just its presence. Every one of the 20 standard
#: residues falls into exactly one bucket; anything not listed (a stray
#: symbol) falls through to the SMALL default rather than raising.
_RING = set("FWYH")        # aromatic — drawn as a small hexagon
_BRANCHED = set("LIVTM")   # bulky/branched aliphatic — drawn as a two-pronged fork
_CHARGED = set("DEKRQN")   # acidic/basic/amide — drawn as a longer, double-forked chain


def _sidechain_kit(
    aa: str, backbone_pt: tuple[float, float, float], u: tuple[float, float], v: tuple[float, float]
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    """A small local side-chain "graph" (points + edges between them) whose
    shape depends on ``aa``'s chemical class. ``u`` is the unit direction
    radially outward from the helix axis at this residue, ``v`` the
    in-plane perpendicular to ``u`` — together they place fork/ring points
    without needing a full local coordinate frame.

    Point 0 is always the backbone atom itself (edge (0, 1) is the bond into
    the side chain); every other point gets a drawn "bead" by the caller.
    """
    p = backbone_pt

    def at(scale_u: float, scale_v: float = 0.0, scale_y: float = 0.0) -> tuple[float, float, float]:
        return (p[0] + u[0] * scale_u + v[0] * scale_v,
                p[1] + scale_y,
                p[2] + u[1] * scale_u + v[1] * scale_v)

    if aa in _RING:
        cb = at(1.3)
        center = at(2.1)
        ring = [
            (center[0] + v[0] * 0.85 * math.cos(k * math.pi / 3),
             center[1] + 0.55 * math.sin(k * math.pi / 3),
             center[2] + v[1] * 0.85 * math.cos(k * math.pi / 3))
            for k in range(6)
        ]
        points = [p, cb, *ring]
        edges = [(0, 1), (1, 2), *((2 + k, 2 + (k + 1) % 6) for k in range(6))]
        return points, edges

    if aa in _BRANCHED:
        cb = at(1.3)
        tip1 = at(0.7, 0.9)
        tip2 = at(0.7, -0.9)
        return [p, cb, tip1, tip2], [(0, 1), (1, 2), (1, 3)]

    if aa in _CHARGED:
        cb = at(1.3)
        cg = (cb[0] + u[0] * 1.0, cb[1], cb[2] + u[1] * 1.0)
        tip1 = (cg[0] + v[0] * 0.7, cg[1] + 0.15, cg[2] + v[1] * 0.7)
        tip2 = (cg[0] - v[0] * 0.7, cg[1] + 0.15, cg[2] - v[1] * 0.7)
        return [p, cb, cg, tip1, tip2], [(0, 1), (1, 2), (2, 3), (2, 4)]

    # SMALL (A, G, S, C, P) and anything unclassified — a short stub, plain.
    return [p, at(1.05)], [(0, 1)]


def peptide_backbone_model(
    sequence: str, radius: float = 2.3, rise: float = 1.55, twist_deg: float = 100.0
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[list[tuple[float, float, float]], list[tuple[int, int]]], ...],
]:
    """Backbone xyz coordinates for ``sequence`` on a right-handed alpha
    helix (textbook radius / rise-per-residue / twist-per-residue), plus one
    :func:`_sidechain_kit` graph per residue built from its real letter.
    """
    twist = math.radians(twist_deg)
    n = len(sequence)
    backbone, thetas = [], []
    for i in range(n):
        theta = i * twist
        thetas.append(theta)
        backbone.append((radius * math.cos(theta), i * rise, radius * math.sin(theta)))
    y_mid = backbone[n // 2][1]
    backbone = tuple((x, y - y_mid, z) for x, y, z in backbone)

    decorations = []
    for i, aa in enumerate(sequence):
        theta = thetas[i]
        u = (math.cos(theta), math.sin(theta))
        v = (-math.sin(theta), math.cos(theta))
        decorations.append(_sidechain_kit(aa, backbone[i], u, v))
    return backbone, tuple(decorations)


def _rotate_yz(pt: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float, float]:
    """Yaw (around the vertical axis) then pitch (around the horizontal axis)."""
    x, y, z = pt
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1, z1 = x * cy + z * sy, -x * sy + z * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2, z2 = y * cp - z1 * sp, y * sp + z1 * cp
    return x1, y2, z2


_RIBBON_NEAR = ["░", "▓", "█", "▓", "░"]
_RIBBON_MID = ["·", "▒", "▒", "▒", "·"]
_RIBBON_FAR = [" ", "·", "·", "·", " "]
_RIBBON_OFFSETS = [-1.1, -0.55, 0.0, 0.55, 1.1]


def render_peptide_frame(
    backbone: tuple[tuple[float, float, float], ...],
    decorations: tuple[tuple[list[tuple[float, float, float]], list[tuple[int, int]]], ...],
    yaw: float,
    pitch: float,
    width: int = 48,
    height: int = 24,
    scale: float = 2.15,
) -> str:
    """One ASCII frame: the backbone as a thick shaded ribbon (a real-width
    perpendicular sweep with a 5-tone block profile, so it reads as a lit
    round tube rather than a wire), each residue's side chain drawn as thin
    bonded strokes ending in small beads — a hexagon for a ring, a fork for
    branched/charged residues, a single bead for a small one. Every stroke
    and bead across the whole molecule is collected and painted in one
    depth-sorted (far-to-near) pass so nearer geometry always visibly wins
    an overlap, then backbone atoms are stamped on top of everything else
    (a residue marker hidden behind a bond reads as a broken chain).
    """

    def project(pt: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = _rotate_yz(pt, yaw, pitch)
        col = width / 2 + x * scale
        row = height / 2 - y * scale * 0.5  # terminal cells are ~2x taller than wide
        return col, row, z

    pb = [project(p) for p in backbone]
    proj_deco = [([project(p) for p in pts], edges) for pts, edges in decorations]

    zs = [p[2] for p in pb] + [p[2] for pts, _ in proj_deco for p in pts]
    zmin, zspan = min(zs), max(max(zs) - min(zs), 1e-6)

    def ribbon_profile(z: float) -> list[str]:
        f = (z - zmin) / zspan
        return _RIBBON_NEAR if f > 0.62 else (_RIBBON_MID if f > 0.3 else _RIBBON_FAR)

    def bucket(z: float, near: str, mid: str, far: str) -> str:
        f = (z - zmin) / zspan
        return near if f > 0.62 else (mid if f > 0.3 else far)

    draws: list[tuple[float, float, float, str]] = []

    # Backbone: a real-width ribbon, not a single-cell line — at every point
    # along each segment, sweep a short perpendicular span filled with a
    # 5-tone profile so it reads as a lit tube.
    for i in range(len(pb) - 1):
        c0, r0, z0 = pb[i]
        c1, r1, z1 = pb[i + 1]
        dc, dr = c1 - c0, r1 - r0
        seg_len = max((dc**2 + dr**2) ** 0.5, 1e-6)
        tx, ty = dc / seg_len, dr / seg_len
        px, py = -ty, tx  # perpendicular unit vector
        steps = max(int(round(seg_len)), 1)
        for s in range(steps + 1):
            t = s / steps
            cx, cy = c0 + dc * t, r0 + dr * t
            z = z0 + (z1 - z0) * t
            for off, ch in zip(_RIBBON_OFFSETS, ribbon_profile(z)):
                if ch == " ":
                    continue
                draws.append((z, cx + px * off, cy + py * off * 0.6, ch))

    # Side chains: thin bonded strokes + beads, shaped by _sidechain_kit().
    for pts, edges in proj_deco:
        for a, b in edges:
            c0, r0, z0 = pts[a]
            c1, r1, z1 = pts[b]
            zmid = (z0 + z1) / 2
            ch = bucket(zmid, "#", "+", ".")
            seg_len = max(((c1 - c0) ** 2 + (r1 - r0) ** 2) ** 0.5, 1e-6)
            steps = max(int(round(seg_len)), 1)
            for s in range(steps + 1):
                t = s / steps
                draws.append((zmid, c0 + (c1 - c0) * t, r0 + (r1 - r0) * t, ch))
        for c, r, z in pts[1:]:  # point 0 is the backbone atom, drawn separately below
            draws.append((z + 1e-3, c, r, bucket(z, "@", "o", ".")))

    grid = [[" "] * width for _ in range(height)]

    def put(col: float, row: float, ch: str) -> None:
        ci, ri = int(round(col)), int(round(row))
        if 0 <= ri < height and 0 <= ci < width:
            grid[ri][ci] = ch

    for _z, col, row, ch in sorted(draws, key=lambda d: d[0]):
        put(col, row, ch)
    for col, row, z in sorted(pb, key=lambda p: p[2]):
        put(col, row, bucket(z, "●", "o", "·"))

    return "\n".join("".join(row) for row in grid)


def _peptide_frames(n_frames: int = 16) -> tuple[str, ...]:
    """The gallery piece: the real parent-project peptide (see
    ``PARENT_PROJECT_PEPTIDE`` below), spun through a full turn at a fixed
    downward tilt so it reads as a 3D object rotating in space.

    Sized smaller than :func:`render_peptide_frame`'s own default: at this
    many real residues this piece is inherently taller than the other three
    gallery pieces, and the terminal UI's output panel is a compact, fixed
    box — a piece too tall for it just scrolls its own title out of view
    before the animation finishes. 19 rows keeps it in the same ballpark as
    the rest of the gallery while still reading clearly as a full molecule.
    """
    backbone, decorations = peptide_backbone_model(PARENT_PROJECT_PEPTIDE)
    pitch = 0.38
    return tuple(
        render_peptide_frame(backbone, decorations, f * (2 * math.pi / n_frames), pitch,
                              width=44, height=19, scale=1.7)
        for f in range(n_frames)
    )


# ---------------------------------------------------------------------------
# Docking — a small 2D scene: a rounded pocket and a bent tripeptide ligand
# swinging into it along a curved path, turning to face the pocket as it
# settles (the same "key turning into a lock" idea as induced fit).
# ---------------------------------------------------------------------------

_POCKET = (
    "     .------------------.",
    "    /                    \\",
    "   |                      |",
    "   |                      |",
    "    \\                    /",
    "     '------------------'",
)
_POCKET_BOUND = (
    "     +==================+",
    "    /                    \\",
    "   #                      #",
    "   #                      #",
    "    \\                    /",
    "     +==================+",
)


def _bezier(t: float, p0, p1, p2) -> tuple[float, float]:
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return x, y


def _rotate2d(pt: tuple[float, float], angle: float) -> tuple[float, float]:
    x, y = pt
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c


def _docking_frames(n_frames: int = 7) -> tuple[str, ...]:
    width = max(len(line) for line in _POCKET) + 2
    pocket_top_row = 3
    height = pocket_top_row + len(_POCKET)
    cup_center = (13.0, float(pocket_top_row) + 2.5)
    start = (3.0, 0.0)
    control = (11.0, 0.0)
    # A shallow bent tripeptide (three backbone points) plus one visible
    # side-chain stub off the middle residue — small enough to read clearly
    # at this scale, but unmistakably a chain, not a blob.
    ligand_local = ((-2.3, -0.7), (0.0, 0.4), (2.3, -0.7))
    sidechain_local = (0.0, 1.7)

    frames = []
    for f in range(n_frames):
        t = (f / (n_frames - 1)) ** 0.65
        bound = f == n_frames - 1
        cx, cy = _bezier(t, start, control, cup_center)
        angle = math.radians(70.0 * (1.0 - t))  # turns to face the pocket as it settles

        grid = [[" "] * width for _ in range(height)]
        for r, line in enumerate(_POCKET_BOUND if bound else _POCKET):
            for c, ch in enumerate(line):
                grid[pocket_top_row + r][c] = ch

        def put(x: float, y: float, ch: str) -> None:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < height and 0 <= xi < width:
                grid[yi][xi] = ch

        pts = [(cx + dx, cy + dy) for dx, dy in (_rotate2d(p, angle) for p in ligand_local)]
        side = (cx + _rotate2d(sidechain_local, angle)[0], cy + _rotate2d(sidechain_local, angle)[1])

        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            for s in range(4):
                tt = s / 3
                put(x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt, "-")
        mx, my = pts[1]
        for s in range(3):
            tt = s / 2
            put(mx + (side[0] - mx) * tt, my + (side[1] - my) * tt, ":")
        for x, y in pts:
            put(x, y, "●")

        lines = ["".join(row) for row in grid]
        lines.append("      * induced fit -- bound" if bound else " ")
        frames.append("\n".join(lines))
    return tuple(frames)


def _diffusion_frames() -> tuple[str, ...]:
    """What Stage 1 (RAPiDock's diffusion model) is actually doing: a cloud
    of scattered points converging onto a target backbone shape over several
    denoising steps, and — on the final step — those points connecting into
    a closed loop, because a real diffusion model's output isn't just dots in
    the right place, it's a bonded chain.

    Each point converges at its own slightly randomized rate (a fixed-seed
    per-point speed jitter) rather than all in perfect lockstep — closer to
    how an actual reverse-diffusion trajectory looks, and considerably less
    mechanical-looking than a uniform linear interpolation. Fixed seed:
    deterministic, testable, identical on every machine.
    """
    cols, rows, n = 29, 10, 18
    cx, cy = cols / 2, rows / 2
    target = [
        (cx + 10.0 * math.cos(2 * math.pi * i / n), cy + 3.2 * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    rng = random.Random(7)
    noise = [(rng.uniform(1, cols - 2), rng.uniform(1, rows - 2)) for _ in range(n)]
    speed = [rng.uniform(0.75, 1.35) for _ in range(n)]  # per-point convergence-rate jitter
    steps = [
        (0.0, "pure noise"),
        (0.15, "denoising."),
        (0.35, "denoising.."),
        (0.55, "denoising..."),
        (0.75, "resolving...."),
        (0.92, "resolving....."),
        (1.0, "pose (backbone closed)"),
    ]

    def render(t: float, label: str) -> str:
        grid = [[" "] * cols for _ in range(rows)]
        pts = []
        for (nx, ny), (tx, ty), sp in zip(noise, target, speed):
            tt = min(1.0, t * sp)
            pts.append((nx + (tx - nx) * tt, ny + (ty - ny) * tt))
        if t >= 0.999:
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                for s in range(4):
                    xi = int(round(x0 + (x1 - x0) * s / 4))
                    yi = int(round(y0 + (y1 - y0) * s / 4))
                    if 0 <= yi < rows and 0 <= xi < cols:
                        grid[yi][xi] = "·"
        for (x, y), sp in zip(pts, speed):
            settled = t >= 0.999
            glyph = "●" if settled else ("o" if t * sp >= 0.55 else "·")
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rows and 0 <= xi < cols:
                grid[yi][xi] = glyph
        body = "\n".join("".join(r) for r in grid)
        return f"{body}\n      {label}"

    return tuple(render(t, label) for t, label in steps)


def _pad_frame(frame: str, width: int, height: int) -> str:
    """Pad one frame to a fixed (width, height) canvas, centered.

    First equalizes ragged line lengths within the frame itself (the docking
    and diffusion pieces append a caption line shorter than their art lines
    above it, so raw frames from those two are not even internally
    rectangular), then centers that block inside the shared canvas with
    blank padding.
    """
    lines = frame.split("\n")
    own_width = max((len(ln) for ln in lines), default=0)
    lines = [ln.ljust(own_width) for ln in lines]
    pad_left = max(0, (width - own_width) // 2)
    pad_right = max(0, width - own_width - pad_left)
    lines = [(" " * pad_left) + ln + (" " * pad_right) for ln in lines]
    pad_top = max(0, (height - len(lines)) // 2)
    pad_bottom = max(0, height - len(lines) - pad_top)
    blank = " " * width
    return "\n".join([blank] * pad_top + lines + [blank] * pad_bottom)


def _normalize_gallery_canvas(
    pieces: tuple[tuple[str, tuple[str, ...]], ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Pad every frame of every gallery piece to one shared canvas size.

    Without this, the four pieces render at wildly different sizes (DNA
    23x15, peptide 44x19, docking/diffusion up to 29 wide with a shorter
    caption line) — so the TUI's surrounding "progress" panel border visibly
    resized depending on which piece start_stream() happened to pick at
    random for a given run, and the docking/diffusion pieces even resized
    frame-to-frame within themselves.
    """
    width = max(len(ln) for _t, frames in pieces for f in frames for ln in f.split("\n"))
    height = max(len(f.split("\n")) for _t, frames in pieces for f in frames)
    return tuple(
        (title, tuple(_pad_frame(f, width, height) for f in frames))
        for title, frames in pieces
    )


#: (title, frames) pairs, rotated in order during a long wait. Every piece
#: here animates — a science-themed slideshow with no motion wasn't earning
#: its place on screen during a wait that's supposed to feel alive. Every
#: frame of every piece shares one canvas size (see _normalize_gallery_canvas)
#: so the surrounding UI panel doesn't resize when the gallery rotates.
GALLERY: tuple[tuple[str, tuple[str, ...]], ...] = _normalize_gallery_canvas((
    ("DNA — double helix, rotating", _dna_frames()),
    ("Peptide — LISAAALAAIFAAALAC, real sequence, rotating", _peptide_frames()),
    ("Induced fit — peptide docking into the receptor pocket", _docking_frames()),
    ("Stage 1 — diffusion: noise resolving into a pose", _diffusion_frames()),
))


def gallery_piece(index: int) -> tuple[str, tuple[str, ...]]:
    """The ``index``-th gallery piece (title, frames), cycling."""
    return GALLERY[index % len(GALLERY)]


HOLD_FIRST_FRAME_S = 1.0
HOLD_LAST_FRAME_S = 1.0


def frame_index_for_elapsed(
    n_frames: int,
    elapsed: float,
    frame_delay: float = 0.32,
    hold_first: float = HOLD_FIRST_FRAME_S,
    hold_last: float = HOLD_LAST_FRAME_S,
) -> int:
    """Which frame (0..n_frames-1) should be showing `elapsed` seconds into
    a continuously-looping gallery animation, given the first and last frame
    each linger `hold_first`/`hold_last` seconds longer than the rest.

    Pure function of elapsed time (deterministic, testable without a clock or
    a thread) - used by the TUI's Stage-1 progress panel, which redraws from
    wall-clock elapsed time on every render tick rather than sleeping between
    frames. A uniform per-frame delay there made the loop snap straight from
    the last frame back to the first with no beat at either end, which read
    as too fast / mechanical over a long wait.
    """
    if n_frames <= 1:
        return 0
    durations = [frame_delay] * n_frames
    durations[0] += hold_first
    durations[-1] += hold_last
    total = sum(durations)
    if total <= 0:
        return 0
    t = elapsed % total
    acc = 0.0
    for i, d in enumerate(durations):
        acc += d
        if t < acc:
            return i
    return n_frames - 1


def animate_gallery_piece(
    stream: TextIO, index: int, indent: str = "   ", frame_delay: float = 0.16
) -> None:
    """Write one gallery piece to ``stream``, animating in place if it has
    more than one frame, then leave the final frame on screen.

    The first frame lingers an extra HOLD_FIRST_FRAME_S before the animation
    starts moving, and the completed last frame lingers an extra
    HOLD_LAST_FRAME_S before this function returns - the same "give both ends
    of the loop a beat" idea as frame_index_for_elapsed, applied to a one-shot
    player instead of a continuous one.

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
        for i, frame in enumerate(frames[1:], start=1):
            time.sleep(frame_delay + (HOLD_FIRST_FRAME_S if i == 1 else 0.0))
            lines = frame.splitlines()
            stream.write(f"\x1b[{n}A")
            for line in lines:
                stream.write(f"\x1b[2K{indent}{line}\n")
            stream.flush()
            n = len(lines)
        time.sleep(HOLD_LAST_FRAME_S)
        stream.write("\n")
        stream.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ambient spinner — a braille spin glyph + a cycling domain-flavored verb
# ---------------------------------------------------------------------------

_SPIN_CHAIN = ["o", "o-o", "o-o-o", "o-o-o-o", "o-o-o", "o-o", "o"]
_BRAILLE = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

#: Cycling ticker text for the in-place "still working" line — the same idea
#: as a chat assistant's own rotating "Thinking…"/"Pondering…" verb, themed
#: to this pipeline's wet-lab/dry-lab vocabulary instead of a generic one.
#: Purely cosmetic: never parsed, never logged to run_metadata.json.
SPINNER_WORDS: tuple[str, ...] = (
    "Diffusing", "Helixing", "Folding", "Docking", "Annealing", "Ligating",
    "Base-pairing", "Sequencing", "Transforming", "Titrating", "Centrifuging",
    "Denaturing", "Renaturing", "Incubating", "Pipetting", "Vortexing",
    "Electrophoresing", "PCR-cycling", "Culturing", "Autoclaving",
    "Plasmid-prepping", "Cloning", "Expressing", "Purifying", "Crystallizing",
    "Clustering", "Minimizing", "Rescoring", "Solvating", "Equilibrating",
)


def spin_frame(t: float | None = None) -> str:
    """One frame of a small ambient 'peptide chain forming' animation.

    Deterministic in ``t`` (seconds) so it's testable without a clock; the
    caller passes ``time.time()`` for real use.
    """
    now = t if t is not None else time.time()
    return _SPIN_CHAIN[int(now * 3) % len(_SPIN_CHAIN)]


def braille_spin(t: float | None = None) -> str:
    """One frame of a small braille dot spinner — smoother-looking than an
    ASCII chain at the ~10fps a heartbeat line redraws at. Deterministic in
    ``t`` for the same testability reason as :func:`spin_frame`.
    """
    now = t if t is not None else time.time()
    return _BRAILLE[int(now * 10) % len(_BRAILLE)]


def spinner_word(t: float | None = None, period_s: float = 180.0) -> str:
    """A cycling domain-flavored verb (``'Diffusing…'``, ``'Helixing…'``, …).
    Deterministic in ``t`` (seconds) so it's testable without a clock.

    period_s=180 (3 minutes): a real Stage 1 wait runs minutes, not seconds --
    the old 2.2s default cycled through the whole SPINNER_WORDS list many
    times over during one wait, which read as jittery/meaningless rather than
    as a heartbeat. At 3 minutes/word, a run typically shows only one or two
    words total, closer to how a slow, ongoing process should read.
    """
    now = t if t is not None else time.time()
    word = SPINNER_WORDS[int(now / period_s) % len(SPINNER_WORDS)]
    return f"{word}\u2026"


# ---------------------------------------------------------------------------
# Easter eggs
# ---------------------------------------------------------------------------

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

_SHINY = """
      *  .  *    \u00b7  .  *
   .    *   you found a shiny pose!   .
      \u00b7         (this is rare)        *
   *  .    keep this run's seed     .
"""


def easter_egg_for_peptide(sequence: str) -> str | None:
    """A special art block for a small, fixed set of trigger sequences.

    Case-insensitive, whitespace-stripped. One trigger:

    - ``IGEM`` — which is also, conveniently, a real valid tetrapeptide
      (Ile-Gly-Glu-Met), so it passes normal sequence validation on its
      own merits and isn't a special-cased input.

    Returns ``None`` for everything else.
    """
    if not sequence:
        return None
    seq = sequence.strip().upper()
    if seq == "IGEM":
        return _IGEM_EGG
    return None


def maybe_rare(rng: random.Random | None = None, chance: float = 0.05) -> str | None:
    """The rare bonus art with probability ``chance`` (default 1-in-20), else ``None``."""
    r = rng if rng is not None else random
    return _SHINY if r.random() < chance else None
