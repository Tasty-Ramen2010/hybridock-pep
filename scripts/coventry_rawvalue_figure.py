#!/usr/bin/env python
"""Raw model output, not rank — the plot Coventry asked for, and a test of why he asked.

HIS NOTE: "plot either the raw value coming out of your model or the log(raw_value). That's the
most fair way to do it. Plotting the ranking like you are might actually be hurting you a bit if
say #1, 2, and 3 all score about the same and then 4-rest all score poorly."

He is making a falsifiable claim about our data, not just a plotting preference, so it gets
tested rather than accepted. A rank is an order statistic: it throws away margin. If our top few
cells are nearly tied and the rest fall off a cliff, then "cognate ranked 3rd" and "cognate
ranked 3rd" mean completely different things depending on whether 3rd sat 0.1 REU or 40 REU
behind 1st -- and every number we have reported to him so far could not tell those apart.

WHAT IS PLOTTED. The raw cancelled interaction energy in REU, exactly as the model emits it,
with no ranking step anywhere. Lower is better. Median polish has already removed the per-peptide
and per-binder main effects, so zero means "as expected for this peptide and this binder" and
negative means the pair does better than either party's own baseline -- which is what specificity
is. A diverging map centred at zero is therefore the honest encoding.

Log is not used: these are energies and go negative, so log(raw) is undefined on most of the
grid. REU is already the natural scale, and the colour limits are set by a symmetric percentile
so one catastrophic cell cannot flatten everything else.

THE MARGIN TEST. For every peptide the gap from its best cell to 2nd, 3rd and 5th is measured in
REU and in units of that row's own spread. If the gaps are small, Coventry is right and rank has
been understating us; if they are large, rank was a fair summary and the ordering is real. The
answer is printed, not assumed.

Usage: coventry_rawvalue_figure.py
Writes docs/coventry_rawvalue.png
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
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs/coventry_rawvalue.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)


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
    return np.array([[rows.get((p, b), np.nan) for b in ORDER] for p in ORDER])


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    T = np.full((N, N), np.nan)
    keep = np.zeros((N, N), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth.get((p, b))
            keep[i, j] = (i == j) or (r or {}).get("measured") == "0"
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
    COFZ = _z(COF) + _z(_mc(IPT))
    # a RAW-VALUE combination, not a rank average -- his point applies to how we combine too
    BOTH = _z(OURS) + COFZ

    def grade(S):
        cog = np.array([S[i, i] for i in range(N)])
        non = np.array([S[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        rr = [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]
        return a, rr

    # ---------------------------------------------------------------- the margin test
    print("THE MARGIN TEST — is a rank hiding how close our top cells are?\n")
    print("For each peptide: how far behind our own best cell the 2nd, 3rd and 5th sit, in REU")
    print("and as a fraction of that row's full spread (best to worst).\n")
    print(f"  {'peptide':<9}{'cog rank':>9}{'gap 1→2':>9}{'gap 1→3':>9}{'gap 1→5':>9}"
          f"{'row spread':>12}{'1→3 as % of spread':>21}")
    fr23 = []
    for i, p in enumerate(ORDER):
        row = np.sort(OURS[i])
        spread = row[-1] - row[0]
        g2, g3, g5 = row[1] - row[0], row[2] - row[0], row[4] - row[0]
        pct = 100 * g3 / spread if spread else float("nan")
        fr23.append(pct)
        rk = sorted(range(N), key=lambda j: OURS[i, j]).index(i) + 1
        print(f"  {p:<9}{rk:>9}{g2:>9.2f}{g3:>9.2f}{g5:>9.2f}{spread:>12.1f}{pct:>20.1f}%")
    print(f"\n  median: the top THREE cells span {st.median(fr23):.1f}% of the row's full range")
    tight = sum(1 for x in fr23 if x < 10)
    print(f"  rows where the top 3 sit inside 10% of the range: {tight}/18")
    print("  -> " + ("Coventry is right: rank understates us, the top cells are near-ties."
                     if st.median(fr23) < 10 else
                     "the top cells are genuinely separated, so rank was a fair summary here."))

    # how often is the cognate within a whisker of our #1, regardless of its rank?
    print("\n  Cognate's distance behind our own best cell, in units of the row's spread:")
    near = 0
    for i, p in enumerate(ORDER):
        row = np.sort(OURS[i])
        spread = row[-1] - row[0]
        d = (OURS[i, i] - row[0]) / spread if spread else float("nan")
        near += d < 0.10
    print(f"    cognate within 10% of our best cell on {near}/18 peptides "
          f"(rank says top-3 on {sum(1 for i in range(N) if sorted(range(N), key=lambda j: OURS[i,j]).index(i) < 3)}/18)")

    # ---------------------------------------------------------------- the figure
    panels = [("B  our poses\ncancelled ref2015", OURS, "interaction energy (REU)"),
              ("C  co-folded poses\ncancelled + ipTM", COFZ, "combined z"),
              ("D  both arms, raw z-sum\n(not a rank average)", BOTH, "combined z")]
    fig, axes = plt.subplots(1, 4, figsize=(23.5, 6.9))

    cmap0 = plt.get_cmap("Blues").copy(); cmap0.set_bad("#ffffff")
    im = axes[0].imshow(np.ma.masked_invalid(T), cmap=cmap0, aspect="equal")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("measured  −log₁₀(K$_d$)", fontsize=8); cb.ax.tick_params(labelsize=7)
    axes[0].set_title("Wu et al. 2025, Fig. 2B — ground truth\n"
                      "30 measured of 324 · white = no binding detected",
                      fontsize=9.5, linespacing=1.5)
    axes[0].set_ylabel("peptide", fontsize=9.5)

    for ax, (title, M, unit) in zip(axes[1:], panels):
        lim = np.nanpercentile(np.abs(M), 96) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        a, rr = grade(M)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(unit + "  (lower = better)", fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\nAUC {a:.3f} · mean rank {st.mean(rr):.2f}/18 · "
                     f"top-5 {sum(r <= 5 for r in rr)}/18", fontsize=9.5, linespacing=1.5)

    for ax in axes:
        for i in range(N):
            for j in range(N):
                r = truth.get((ORDER[i], ORDER[j]))
                if r and r["measured"] == "1" and i != j:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#2b2b2b", lw=1.1, zorder=5))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(ORDER, rotation=90, fontsize=7.5)
        ax.set_yticklabels(ORDER, fontsize=7.5)
        ax.set_xlabel("binder", fontsize=9.5)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle(
        "Raw model output, no ranking step anywhere — per Brian's note that ranking can hide a "
        "near-tie at the top.\nValues are the cancelled interaction energy: median polish has "
        "already removed each peptide's and each binder's own baseline, so 0 means "
        "'as expected for this pair' and blue means the pair beats both baselines.\n"
        "Log is not used — these are energies and go negative. Grey squares mark the "
        "off-diagonal pairs the paper measured; the diagonal is unmarked.",
        fontsize=10.2, y=0.985, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.875))
    fig.savefig(OUT, dpi=170)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
