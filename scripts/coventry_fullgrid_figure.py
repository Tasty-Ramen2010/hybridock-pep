#!/usr/bin/env python
"""All 324 cells, coloured by rank — the plot Coventry asked for, with a scale that survives it.

HIS REQUEST, 2:12 IN THE MEETING: "just plot all the cells and not, you know, when it hits that
you don't pull out the other ones." Every earlier figure drew a shortlist, which quietly hides
the thing he wants to judge: what the model says about the 300-odd pairs it gets no credit for.

WHY A LINEAR RANK SCALE MAKES THAT PLOT UNREADABLE. Fill 324 cells with ranks 1-18 on a linear
colour ramp and two thirds of the picture is mid-tone noise, because ranks 10 through 18 are all
"not a binder" and none of them mean anything different from each other. The eye spends its whole
dynamic range separating 14th place from 15th, which is a distinction without a difference, and
the handful of cells that actually carry the answer get a sliver of the ramp each.

So the colour is a POWER FUNCTION of rank rather than the rank itself:

    v = ((18 - rank) / 17) ** GAMMA          rank 18 -> 0.000,  rank 1 -> 1.000

With GAMMA = 3 the whole of ranks 18-10 occupies the bottom 10% of the ramp and stays nearly flat,
and each further step toward first place takes a visibly bigger bite: the gap from 3rd to 2nd is
0.147 of the scale, from 2nd to 1st 0.166, against 0.044 from 11th to 10th. The decisions that
matter get the contrast, and the cells that mean "no" stay quiet without being hidden.

This is a display transform, not an analysis one. Every rank is still drawn and still printed in
its cell; nothing is thresholded away.

TWO VERSIONS.
  discrete   one flat colour per cell plus the rank as a number -- the honest, auditable one.
  gradient   the same field bicubically interpolated, with the expected-answer diagonal drawn
             over it, so the question becomes "does the bright ridge follow the line" rather
             than "can you find the bright cells". Smoothing is cosmetic and says so.

Usage: coventry_fullgrid_figure.py [--gamma 3.0]
Writes docs/coventry_fullgrid.png and docs/coventry_fullgrid_gradient.png
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
GAMMA = float(sys.argv[sys.argv.index("--gamma") + 1]) if "--gamma" in sys.argv else 3.0
#: Drop every marker that points at the diagonal -- the black cognate outlines AND the bold
#: weight on the diagonal rank numbers. Both are annotation, and if the trend is only visible
#: because we drew attention to it then it is not in the data. The grey outlines on the
#: off-diagonal measured pairs stay: they mark cross-reactants the paper measured, which is
#: information the colour does not carry and is not the trend being tested.
NO_BOXES = "--no-boxes" in sys.argv


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def grid_of(path: Path, key: str = "i_total") -> np.ndarray:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = float(r["best"].get(key, 0.0))
    return np.array([[rows.get((p, b), 0.0) for b in ORDER] for p in ORDER])


def rowrank(S: np.ndarray) -> np.ndarray:
    """Rank of every binder within each peptide's own row. 1 = we call it the best partner."""
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        for k, j in enumerate(sorted(range(N), key=lambda j: S[i, j])):
            R[i, j] = k + 1
    return R


def shade(R: np.ndarray) -> np.ndarray:
    """Rank -> colour value, compressed at the uninformative end. See module docstring."""
    return ((N - R) / (N - 1.0)) ** GAMMA


def load():
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    T = np.full((N, N), np.nan)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth.get((p, b))
            if r and r["measured"] == "1":
                try:
                    T[i, j] = -math.log10(float(r["kd_M"]))
                except (TypeError, ValueError):
                    T[i, j] = 7.0

    OURS = polish(grid_of(ROOT / "logs/sel_coventry_features_hard.jsonl"))
    COF = polish(grid_of(ROOT / "logs/sel_cofold_features.jsonl"))
    aux = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            aux[(r["peptide"], r["binder"])] = r
    IPT = np.array([[-(aux.get((p, b), {}).get("iptm") or 0.0) for b in ORDER] for p in ORDER])
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    Ro = rowrank(_z(OURS))
    Rc = rowrank(_z(COF) + _z(_mc(IPT)))
    Rb = rowrank((Ro + Rc).astype(float))
    return truth, T, Ro, Rc, Rb


def stats(R: np.ndarray) -> str:
    d = [R[i, i] for i in range(N)]
    return (f"mean rank {np.mean(d):.2f}/18 · top-1 {sum(r == 1 for r in d)} · "
            f"top-3 {sum(r <= 3 for r in d)} · top-5 {sum(r <= 5 for r in d)}")


def frame(ax, truth, diag_lw=2.0):
    for i in range(N):
        for j in range(N):
            r = truth.get((ORDER[i], ORDER[j]))
            if not (r and r["measured"] == "1"):
                continue
            if i == j and NO_BOXES:
                continue
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="#000000" if i == j else "#3b3b3b",
                                   lw=diag_lw if i == j else 1.1, zorder=5))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=7.5)
    ax.set_yticklabels(ORDER, fontsize=7.5)
    ax.set_xlabel("binder", fontsize=9.5)
    for s in ax.spines.values():
        s.set_color("#999999"); s.set_linewidth(0.6)


def truth_panel(ax, T, fig):
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#ffffff")
    im = ax.imshow(np.ma.masked_invalid(T), cmap=cmap, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("measured  −log₁₀(K$_d$)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("Wu et al. 2025, Fig. 2B — ground truth\n"
                 "30 measured of 324 · white = no binding detected",
                 fontsize=9.5, linespacing=1.5)


def rank_colorbar(fig, ax, im):
    """Ticks at the RANKS, placed at their transformed positions, so the squeeze is visible.

    Only ranks that land far enough apart on the transformed ramp get a label -- the whole point
    of the transform is that 18 through 12 are nearly coincident, so labelling each of them
    just produces a pile of overlapping digits at the bottom of the bar.
    """
    ticks = [18, 12, 9, 7, 5, 4, 3, 2, 1]
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                      ticks=[shade(np.array([t]))[0] for t in ticks])
    cb.ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    cb.set_label("our rank for that peptide  (1 = called best)", fontsize=8)
    cb.ax.text(0.5, -0.045, "↑ 18–12 nearly flat", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=6.2, style="italic", color="#5b6470")
    return cb


def main() -> None:
    truth, T, Ro, Rc, Rb = load()
    cmap = plt.get_cmap("magma_r")

    # ------------------------------------------------------------------ discrete, all cells
    panels = [("B  our poses\ncancelled ref2015", Ro),
              ("C  co-folded poses\ncancelled + ipTM", Rc),
              ("D  both arms, rank-averaged", Rb)]
    fig, axes = plt.subplots(1, 4, figsize=(23.5, 6.9))
    truth_panel(axes[0], T, fig)
    axes[0].set_ylabel("peptide", fontsize=9.5)
    frame(axes[0], truth)
    for ax, (title, R) in zip(axes[1:], panels):
        V = shade(R)
        im = ax.imshow(V, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(N):
            for j in range(N):
                ax.text(j, i, str(R[i, j]), ha="center", va="center", zorder=6,
                        fontsize=5.6,
                        fontweight="normal" if NO_BOXES else
                        ("bold" if i == j else "normal"),
                        color="#ffffff" if V[i, j] > 0.45 else "#4a4a4a")
        frame(ax, truth)
        ax.set_title(f"{title}\n{stats(R)}", fontsize=9.5, linespacing=1.5)
        rank_colorbar(fig, ax, im)
    fig.suptitle(
        "Every one of the 324 cells, coloured by where we rank that binder for that peptide.\n"
        f"Colour is ((18−rank)/17)^{GAMMA:g}: ranks 18–10 sit in the bottom tenth of the ramp, and "
        "each step toward first place takes a larger bite — without that, 324 mid-tone cells are "
        "a wash.\n" + ("No marker on the diagonal and no bold on its numbers — nothing here "
         "points at the answer; grey squares are the other pairs the paper measured."
         if NO_BOXES else
         "Black square = the cognate; grey squares = the other pairs the paper measured."),
        fontsize=10.2, y=0.985, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.875))
    out = ROOT / "docs/coventry_fullgrid.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")

    # ------------------------------------------------------------------ gradient + our line
    fig2, axes2 = plt.subplots(1, 3, figsize=(18.4, 6.9))
    truth_panel(axes2[0], T, fig2)
    axes2[0].set_ylabel("peptide", fontsize=9.5)
    frame(axes2[0], truth, diag_lw=1.6)
    axes2[0].add_line(Line2D([-0.5, N - 0.5], [-0.5, N - 0.5], color="#d62728",
                             lw=1.6, alpha=0.85, zorder=6))
    for ax, (title, R) in zip(axes2[1:], [panels[0], panels[2]]):
        V = shade(R)
        im = ax.imshow(V, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                       interpolation="bicubic")
        # the expected-answer line: every cognate lies on it, so a model that works puts its
        # bright ridge here and nowhere else
        ax.add_line(Line2D([-0.5, N - 0.5], [-0.5, N - 0.5], color="#d62728",
                           lw=1.8, alpha=0.9, zorder=6))
        frame(ax, truth, diag_lw=1.4)
        ax.set_title(f"{title}\n{stats(R)}", fontsize=9.5, linespacing=1.5)
        rank_colorbar(fig2, ax, im)
    fig2.suptitle(
        "The same field, bicubically smoothed, with the expected-answer diagonal drawn in red.\n"
        "Smoothing is cosmetic — no cell is thresholded or dropped — and the question it makes "
        "easy to answer is whether the dark ridge follows the line.",
        fontsize=10.2, y=0.98, linespacing=1.5)
    fig2.tight_layout(rect=(0, 0.04, 1, 0.89))
    out2 = ROOT / "docs/coventry_fullgrid_gradient.png"
    fig2.savefig(out2, dpi=170)
    print(f"wrote {out2}")

    print(f"\ncolour scale, gamma={GAMMA:g}:")
    for r in (18, 15, 12, 10, 8, 6, 4, 3, 2, 1):
        v = shade(np.array([r]))[0]
        print(f"  rank {r:2d} -> {v:.3f}")
    print(f"\n  ranks 18..10 span {shade(np.array([10]))[0]:.3f} of the ramp; "
          f"ranks 3..1 span {1 - shade(np.array([3]))[0]:.3f}")
    for nm, R in (("ours", Ro), ("boltz", Rc), ("combined", Rb)):
        print(f"  {nm:<9}{stats(R)}")


if __name__ == "__main__":
    main()
