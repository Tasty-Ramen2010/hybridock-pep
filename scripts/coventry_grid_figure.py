#!/usr/bin/env python
"""Draw our predicted 18x18 specificity grid next to the measured one.

Left  : Wu et al. Table S1 -- measured Kd. Coloured by -log10(Kd); white = no measurable
        binding (294 of 324 cells).
Middle: our raw score (Rosetta ref2015 interface energy after interface repacking, best of
        24 poses). More negative = predicted tighter.
Right : our two-way interaction residual, score - row_mean - col_mean + grand_mean. The
        grid is a complete two-way design, so everything we systematically cannot model --
        a peptide's desolvation offset, its length and charge, a binder's overall interface
        size -- is a MAIN EFFECT that is constant down a row or column and cancels here.
        What is left is the interaction, which is what "specificity" means.

Rows are peptides, columns are binders, in the paper's own order (note pc18 before pc17,
as printed). The cognate diagonal is outlined. Incomplete rows are hatched, not blank, so
a partial grid can never be mistaken for a prediction of "no binding".

Usage: coventry_grid_figure.py [out.png]
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_grid.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)


def load():
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    pred = {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                pred[(p, b[:-1])] = r["best_iface"]
    return truth, pred


def panel(ax, M, title, cmap, vmin, vmax, mask, note=""):
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    for i in range(N):
        for j in range(N):
            if mask[i][j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e8e8e8",
                                       hatch="///", edgecolor="#bbbbbb", lw=0))
    for k in range(N):
        ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False,
                               edgecolor="#e8820c", lw=1.4))
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([b + "B" for b in ORDER], rotation=90, fontsize=6.5)
    ax.set_yticklabels(ORDER, fontsize=6.5)
    ax.set_title(title, fontsize=9.5, pad=7)
    if note:
        ax.set_xlabel(note, fontsize=7, color="#555555")
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("#999999")
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", lw=0.4)
    ax.tick_params(which="minor", length=0)
    return im


def main() -> None:
    truth, pred = load()
    T = np.full((N, N), np.nan)
    S = np.full((N, N), np.nan)
    missing = [[False] * N for _ in range(N)]
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth[(p, b)]
            if r["kd_M"]:
                T[i, j] = -math.log10(float(r["kd_M"]))
            if (p, b) in pred:
                S[i, j] = pred[(p, b)]
            else:
                missing[i][j] = True

    done_rows = [i for i in range(N) if not any(missing[i])]
    # two-way interaction, computed only over complete rows
    W = np.full((N, N), np.nan)
    if done_rows:
        sub = S[done_rows, :]
        rowm = np.nanmean(sub, axis=1, keepdims=True)
        colm = np.nanmean(sub, axis=0, keepdims=True)
        grand = np.nanmean(sub)
        W[done_rows, :] = sub - rowm - colm + grand

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    tmask = [[False] * N for _ in range(N)]
    panel(axes[0], T, "MEASURED  (Wu et al. Table S1)", "Blues", 4.4, 9.0, tmask,
          "colour = -log10(Kd); white = no measurable binding")
    fin = S[np.isfinite(S)]
    lo, hi = (np.percentile(fin, 2), np.percentile(fin, 98)) if fin.size else (-40, 0)
    panel(axes[1], -S, "OURS - raw ref2015 interface", "Blues", -hi, -lo, missing,
          "darker = predicted tighter; hatched = not yet docked")
    fw = W[np.isfinite(W)]
    wl, wh = (np.percentile(fw, 2), np.percentile(fw, 98)) if fw.size else (-5, 5)
    panel(axes[2], -W, "OURS - two-way interaction (specificity)", "Blues", -wh, -wl, missing,
          "row+column main effects removed")

    ndone = int(np.isfinite(S).sum())
    fig.suptitle(f"Coventry challenge: predicted vs measured specificity grid   "
                 f"({ndone}/324 cells, {len(done_rows)}/18 complete rows)   "
                 f"orange = cognate diagonal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}  ({ndone}/324 cells, {len(done_rows)} complete rows)")

    # quick text read-out of the diagonal on complete rows
    if done_rows:
        print(f"\n{'peptide':>8}  {'cognate rank (raw)':>19}  {'rank (two-way)':>15}")
        for i in done_rows:
            p = ORDER[i]
            rraw = sorted(range(N), key=lambda j: S[i, j]).index(i) + 1
            rw = sorted(range(N), key=lambda j: W[i, j]).index(i) + 1
            print(f"{p:>8}  {rraw:>19d}  {rw:>15d}")


if __name__ == "__main__":
    main()
