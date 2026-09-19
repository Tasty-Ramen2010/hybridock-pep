#!/usr/bin/env python
"""Our grid masked to the cells Wu et al. actually saw binding in.

Left   measured Kd (Table S1), coloured by -log10(Kd). 30 of 324 cells.
Middle OUR score at those SAME 30 cells, everything else forced white.
Right  our full grid, for contrast.

READ THE MIDDLE PANEL CAREFULLY.  Masking to the measured cells asks a narrower and more
forgiving question -- "among the pairs that do bind, do we rank them the way the assay
does?" -- and it HIDES every false positive we produce.  The right-hand panel is what we
actually predict; the difference between middle and right is our false-positive rate, and
that difference is the reason the grid is not reproduced.  Never show the middle panel on
its own.

Colour scales are per-panel (different units: -log10(Kd) vs Rosetta REU), so compare the
PATTERN, not absolute shade.

Usage: coventry_grid_masked_figure.py [out.png]
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
from matplotlib.patches import Rectangle

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_grid_masked.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)


def draw(ax, M, title, vmin, vmax, hatch, sub):
    im = ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    for i in range(N):
        for j in range(N):
            if hatch[i][j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#ececec",
                                       hatch="///", edgecolor="#c8c8c8", lw=0))
    for k in range(N):
        ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False,
                               edgecolor="#e8820c", lw=1.3))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([b + "B" for b in ORDER], rotation=90, fontsize=6.5)
    ax.set_yticklabels(ORDER, fontsize=6.5)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_xlabel(sub, fontsize=7, color="#555555")
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which="minor", color="#cfcfcf", lw=0.4)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#999999")
    return im


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    pred = {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                pred[(p, b[:-1])] = r["best_iface"]

    T = np.full((N, N), np.nan)
    S = np.full((N, N), np.nan)
    Smask = np.full((N, N), np.nan)
    hatch = [[False] * N for _ in range(N)]
    none_h = [[False] * N for _ in range(N)]
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth[(p, b)]
            measured = r["measured"] == "1"
            if measured and r["kd_M"]:
                T[i, j] = -math.log10(float(r["kd_M"]))
            if (p, b) in pred:
                S[i, j] = pred[(p, b)]
                if measured:
                    Smask[i, j] = pred[(p, b)]
            else:
                hatch[i][j] = True

    fin = S[np.isfinite(S)]
    lo, hi = np.percentile(fin, 2), np.percentile(fin, 98)
    mf = Smask[np.isfinite(Smask)]
    mlo, mhi = (np.percentile(mf, 2), np.percentile(mf, 98)) if mf.size else (lo, hi)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0))
    draw(ax[0], T, "MEASURED  (Wu et al. Table S1)", 4.4, 9.0, none_h,
         "colour = -log10(Kd); white = assayed, no binding")
    draw(ax[1], -Smask, "OURS, masked to their 30 measured cells", -mhi, -mlo, none_h,
         "same cells only; hides every false positive we make")
    draw(ax[2], -S, "OURS, full grid (what we actually predict)", -hi, -lo, hatch,
         "darker = predicted tighter; hatched = not docked")

    nm = int(np.isfinite(Smask).sum())
    fig.suptitle(f"Masked comparison: our prediction restricted to the cells that bind "
                 f"({nm}/30 measured cells available)   orange = cognate diagonal",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}  ({nm}/30 measured cells shown)")

    # how many cells would we call binders at the same count as the truth?
    fin_cells = [(S[i, j], truth[(ORDER[i], ORDER[j])]["measured"] == "1")
                 for i in range(N) for j in range(N) if np.isfinite(S[i, j])]
    fin_cells.sort()
    k = sum(1 for _, m in fin_cells if m)
    tp = sum(1 for _, m in fin_cells[:k] if m)
    print(f"false-positive view: taking our top {k} cells recovers {tp} of the {k} true "
          f"binders ({100 * tp / k:.0f}%); the other {k - tp} are cells the assay says "
          f"do not bind")


if __name__ == "__main__":
    main()
