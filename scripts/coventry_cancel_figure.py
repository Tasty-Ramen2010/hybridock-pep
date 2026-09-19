#!/usr/bin/env python
"""The grid before and after cancellation — what removing the main effects actually does.

  A  Wu et al. 2025, Fig. 2B. Filled = a Kd was measured; white = no measurable binding.
  B  Raw Rosetta interface energy on our own docked poses. This is the thing being replaced.
  C  The SAME numbers after two-way cancellation (Tukey median polish). No new physics, no
     fitting, no labels: S(i,j) = grand + peptide(i) + binder(j) + interaction(i,j), and what is
     drawn is the interaction. On this grid 71-74% of the raw variance is main effect.
  D  Cancellation applied to co-folded poses, which is the best combination we have.

WHY A MEDIAN AND NOT A MEAN. A docked grid always contains failed poses worth hundreds of REU.
A row MEAN is dragged by them, so the "main effect" it removes is partly an artefact of the worst
cell in that row, and every other cell in the row gets shifted to compensate. A median is unmoved
by a handful of bad cells. That is the whole difference between panel C and plain double-centring,
and it is worth 0.04-0.07 AUC.

Each diagonal cell carries the cognate's rank in its own row, because shade alone does not say
whether the right answer won.

Usage: coventry_cancel_figure.py [out.png]
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_cancellation.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
FAIL_CAP = 50.0


def main() -> None:
    from hybridock_pep.scoring.cancellation import two_way

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

    ours = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                ours[(r["peptide"], r["binder"])] = float(r["best"].get("i_total", 0.0))
    REF = np.array([[ours[(p, b)] for b in ORDER] for p in ORDER])

    cof = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            v = r.get("best_iface")
            cof[(r["peptide"], r["binder"])] = FAIL_CAP if (v is None or v > FAIL_CAP) else v
    COF = np.array([[cof[(p, b)] for b in ORDER] for p in ORDER])

    def metrics(A):
        rr, t3 = [], 0
        for i in range(N):
            o = sorted(range(N), key=lambda j: A[i, j])
            r = o.index(i) + 1
            rr.append(r); t3 += r <= 3
        cog = [A[i, i] for i in range(N)]
        non = [A[i, j] for i, p in enumerate(ORDER) for j, b in enumerate(ORDER)
               if truth.get((p, b), {}).get("measured") == "0"]
        auc = sum(1.0 if x < y else 0.5 if x == y else 0.0
                  for x in cog for y in non) / (len(cog) * len(non))
        return st.mean(rr), auc, t3

    ip = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("iptm") is not None:
                ip[(r["peptide"], r["binder"])] = -r["iptm"]
    IPT = np.array([[ip[(p, b)] for b in ORDER] for p in ORDER])
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    _BEST = _z(two_way(COF, robust=True, scale=True)) + _z(_mc(IPT))

    panels = [
        ("A  Wu et al. 2025, Fig. 2B", T, None, "measured  −log₁₀(K$_d$)", True),
        ("B  our poses, RAW Rosetta energy", REF, metrics(REF), "ref2015 (REU)", False),
        ("C  our poses, AFTER CANCELLATION", two_way(REF, robust=True),
         metrics(two_way(REF, robust=True)), "interaction (REU)", False),
        ("D  co-folded + cancellation", two_way(COF, robust=True, scale=True),
         metrics(two_way(COF, robust=True, scale=True)), "interaction (scaled)", False),
        ("E  + co-folding confidence", _BEST, metrics(_BEST), "combined z", False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26.5, 6.3))
    for ax, (title, Mx, met, cbar, is_truth) in zip(axes, panels):
        if is_truth:
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#ffffff")
            im = ax.imshow(np.ma.masked_invalid(Mx), cmap=cmap, aspect="equal")
        else:
            lim = np.nanpercentile(np.abs(Mx), 97) or 1.0
            im = ax.imshow(Mx, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        for i, p in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                r = truth.get((p, b))
                if r and r["measured"] == "1":
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111111" if i == j else "#777777",
                                           lw=1.9 if i == j else 1.0, zorder=3))
        if not is_truth:
            for i in range(N):
                rank = sorted(range(N), key=lambda j: Mx[i, j]).index(i) + 1
                deep = abs(Mx[i, i]) > 0.62 * np.nanpercentile(np.abs(Mx), 97)
                ax.text(i, i, str(rank), ha="center", va="center", zorder=4,
                        fontsize=6.6, fontweight="bold",
                        color="#ffffff" if deep else ("#0b3d66" if rank <= 3 else "#8a1c22"))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(ORDER, rotation=90, fontsize=7.5)
        ax.set_yticklabels(ORDER, fontsize=7.5)
        ax.set_xlabel("binder", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("peptide", fontsize=9)
        sub = (f"mean cognate rank {met[0]:.2f}/18 · AUC {met[1]:.3f} · top-3 {met[2]}/18"
               if met else "30 measured of 324 · white = no binding detected")
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, linespacing=1.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7); cb.set_label(cbar, fontsize=8)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle("Two-way cancellation: strip the per-peptide and per-binder main effects, keep "
                 "the interaction. Same energies in B and C — no new physics, no fitting, no "
                 "labels. Random is 9.50 / 0.500 / 3.", fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165)
    print(f"wrote {OUT}")
    for title, _, met, _, _ in panels:
        if met:
            print(f"  {title:38s} rank {met[0]:5.2f}  AUC {met[1]:.3f}  top3 {met[2]:2d}/18")


if __name__ == "__main__":
    main()
