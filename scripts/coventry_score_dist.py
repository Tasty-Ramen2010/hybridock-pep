#!/usr/bin/env python
"""How separated are our scores, really? Distributions of cognate vs non-binding cells.

Left  raw ref2015 interface energy (best of 24 poses per cell).
Right the two-way interaction residual -- score minus row mean minus column mean plus the
      grand mean -- which is the estimator that matches what a specificity assay measures,
      because every systematic per-peptide and per-binder offset is a main effect and
      cancels.

The point of the figure is the OVERLAP, not the difference in means. With 294 non-binding
cells against 18 cognates, a medium effect size still leaves dozens of non-binders scoring
better than the typical true binder, which is what makes the predicted heatmap look uniformly
blue. Dashed line = median cognate; the shaded tail is the non-binders that beat it.

Usage: coventry_score_dist.py [out.png]
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_score_dist.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def cohen_d(a, b):
    s = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
    return (st.mean(a) - st.mean(b)) / s if s else 0.0


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    S = {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                S[(p, b[:-1])] = r["best_iface"]
    rm = {p: st.mean([S[(p, b)] for b in ORDER]) for p in ORDER}
    cm = {b: st.mean([S[(p, b)] for p in ORDER]) for b in ORDER}
    g = st.mean(S.values())
    W = {k: S[k] - rm[k[0]] - cm[k[1]] + g for k in S}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, M, title, unit in (
            (axes[0], S, "Raw ref2015 interface energy", "REU"),
            (axes[1], W, "Two-way interaction (specificity estimator)", "REU, centred")):
        cog = [M[k] for k in M if truth[k]["cognate"] == "1"]
        cross = [M[k] for k in M if truth[k]["cognate"] == "0" and truth[k]["measured"] == "1"]
        non = [M[k] for k in M if truth[k]["measured"] == "0"]
        lo, hi = min(min(cog), min(non)), max(max(cog), max(non))
        bins = np.linspace(lo, hi, 34)
        ax.hist(non, bins=bins, density=True, color="#c9d6e3", edgecolor="#8fa6bb",
                lw=0.5, label=f"non-binding (n={len(non)})")
        ax.hist(cog, bins=bins, density=True, histtype="step", color="#0b5394",
                lw=2.2, label=f"cognate (n={len(cog)})")
        ax.hist(cross, bins=bins, density=True, histtype="step", color="#e8820c",
                lw=1.6, ls="--", label=f"cross-reactive (n={len(cross)})")
        med = st.median(cog)
        ax.axvline(med, color="#0b5394", ls="--", lw=1.2)
        frac = 100 * sum(1 for x in non if x <= med) / len(non)
        tail = [x for x in non if x <= med]
        ax.hist(tail, bins=bins, density=True, color="#c1121f", alpha=0.28,
                label=f"non-binders beating median cognate: {frac:.0f}%")
        ax.set_title(f"{title}\nCohen d = {cohen_d(cog, non):+.2f}", fontsize=10)
        ax.set_xlabel(f"score ({unit})   — more negative = predicted tighter", fontsize=8.5)
        ax.set_ylabel("density", fontsize=8.5)
        ax.legend(fontsize=7.5, framealpha=0.95)
        ax.grid(color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle("Coventry grid, all 324 cells: the separation we actually achieve. "
                 "The overlap is the result — not the gap between means.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
