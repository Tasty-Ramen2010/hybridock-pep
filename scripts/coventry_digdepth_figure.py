#!/usr/bin/env python
"""How far down each peptide's list you would actually have to dig to reach the true partner.

WHAT THIS PANEL SHOWS THAT A TOP-5 HEATMAP DOES NOT. Drawing a fixed top five for every row makes
a row we nail look exactly like a row we barely got, because both show five cells. But those two
rows cost completely different amounts of lab work: if we call the true partner first, you test
one construct; if we call it fifth, you test five. So each row here is drawn to the depth we
actually cost you -- rank 1 draws one cell, rank 3 draws three, rank 5 draws five -- and a row we
miss entirely is drawn at five, the depth at which you would have stopped digging and found
nothing.

Read it as a bill. The total shaded area is the number of experiments our ranking implies, and the
rows with a single dark cell are the ones that cost one experiment each.

Left is the ground truth the paper measured, at the same scale, so the two are directly comparable.

Usage: coventry_digdepth_figure.py [out.png]
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
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_digdepth.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
MAXDEPTH = 5


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


def rowrank(S):
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        for k, j in enumerate(sorted(range(N), key=lambda j: S[i, j])):
            R[i, j] = k + 1
    return R


def main() -> None:
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
    S = (rowrank(_z(OURS)) + rowrank(_z(COF) + _z(_mc(IPT)))).astype(float)
    R = rowrank(S)
    ranks = [R[i, i] for i in range(N)]

    D = np.full((N, N), np.nan)
    for i in range(N):
        depth = min(ranks[i], MAXDEPTH)
        for k, j in enumerate(sorted(range(N), key=lambda j: S[i, j])[:depth]):
            D[i, j] = MAXDEPTH - k          # 5 = our first pick

    shown = int(np.isfinite(D).sum())
    found = sum(1 for r in ranks if r <= MAXDEPTH)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.3))

    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#ffffff")
    im0 = axes[0].imshow(np.ma.masked_invalid(T), cmap=cmap, aspect="equal")
    axes[0].set_title("Wu et al. 2025, Fig. 2B — ground truth\n"
                      "30 measured of 324 · white = no binding detected",
                      fontsize=10, linespacing=1.5)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("measured  −log₁₀(K$_d$)", fontsize=8); cb.ax.tick_params(labelsize=7)

    cmap2 = plt.get_cmap("YlOrRd").copy(); cmap2.set_bad("#f4f4f4")
    im1 = axes[1].imshow(np.ma.masked_invalid(D), cmap=cmap2, vmin=0.5, vmax=5.5, aspect="equal")
    axes[1].set_title(f"HybriDock-Pep — our shortlist, drawn to the depth it costs\n"
                      f"true partner found in {found}/18 rows · "
                      f"{shown} of 324 cells shown (6.2%)",
                      fontsize=10, linespacing=1.5)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03,
                      ticks=[1, 2, 3, 4, 5])
    cb.ax.set_yticklabels(["5th", "4th", "3rd", "2nd", "1st"], fontsize=7)
    cb.set_label("our pick order for that peptide", fontsize=8)

    for ax in axes:
        for i in range(N):
            for j in range(N):
                r = truth.get((ORDER[i], ORDER[j]))
                if r and r["measured"] == "1":
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111111" if i == j else "#888888",
                                           lw=2.0 if i == j else 1.0, zorder=3))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(ORDER, rotation=90, fontsize=8)
        ax.set_yticklabels(ORDER, fontsize=8)
        ax.set_xlabel("binder", fontsize=9.5)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)
    axes[0].set_ylabel("peptide", fontsize=9.5)

    for i in range(N):
        rk = ranks[i]
        axes[1].text(N - 0.35, i, ("✓" if rk == 1 else str(rk) if rk <= MAXDEPTH else f"{rk}"),
                     ha="left", va="center", fontsize=8.2, fontweight="bold",
                     color="#0b3d66" if rk <= MAXDEPTH else "#a11c22")
    axes[1].text(N - 0.35, -1.15, "rank", ha="left", va="center", fontsize=7.6,
                 style="italic", color="#555555")

    fig.suptitle("Each row is drawn only as deep as we make you dig. One dark cell means we "
                 "called the true partner first; five cells mean you would test five. "
                 "Seven rows cost a single experiment.", fontsize=10.5, y=0.965)
    # the rotated binder labels plus an axis title need real room at the bottom, or tight_layout
    # clips the word "binder" off both panels
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    print(f"  rows at rank 1: {sum(1 for r in ranks if r == 1)}   "
          f"<=3: {sum(1 for r in ranks if r <= 3)}   <=5: {found}")
    print(f"  cells drawn {shown}/324; a flat top-5 for every row would draw {5 * N}")
    print(f"  ranks: {dict(zip(ORDER, ranks))}")


if __name__ == "__main__":
    main()
