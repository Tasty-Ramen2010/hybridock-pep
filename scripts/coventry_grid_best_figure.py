#!/usr/bin/env python
"""The Coventry grid, before and after: our poses vs co-folded poses, same scorer throughout.

FOUR PANELS, ALL THE SAME 324 CELLS AND THE SAME GROUND TRUTH.

  A  Wu et al. Fig. 2B.  Filled = a Kd was measured (18 cognates on the diagonal plus 12
     cross-reactivities); white = no measurable binding.  This is what a perfect prediction
     would reproduce.
  B  What we had.  ref2015 interface energy on OUR docked poses, double-centred.  AUC 0.667.
  C  Our IDENTICAL scorer on co-folded poses.  Nothing about the scoring changed -- same
     receptor, same interface repack, same ref2015, same best-of-N rule.  Only the pose source
     is different.  AUC 0.813.
  D  That energy combined with Boltz's own ipTM, both double-centred and z-scored.  AUC 0.868,
     mean cognate rank 3.17 of 18 against a random expectation of 9.50.

WHY EVERY PANEL IS DOUBLE-CENTRED.  score(i,j) = grand + peptide(i) + binder(j) + interaction(i,j).
A peptide's desolvation offset, its length, a binder's general stickiness -- everything we
systematically cannot model -- is a MAIN EFFECT that is constant down a row or a column and
cancels.  Specificity is by definition the interaction term, so the centred residual is the
estimator that matches what the assay measures, not a cosmetic normalisation.

The diagonal is ringed in every panel so the reader can see at a glance how much of it survives.
Colour scales are per-panel because the units differ (-log10 Kd, REU, z); compare the PATTERN.

Usage: coventry_grid_best_figure.py [out.png]
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_grid_best.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
#: a cell whose refinement blew up is a failed pose, not a weak one -- clamp rather than drop,
#: so every row stays complete and the centring is not computed on a ragged grid
FAIL_CAP = 50.0


def centre(M: dict) -> dict:
    peps = [p for p in ORDER if all((p, b) in M for b in ORDER)]
    rm = {p: st.mean([M[(p, b)] for b in ORDER]) for p in peps}
    cm = {b: st.mean([M[(p, b)] for p in peps]) for b in ORDER}
    g = st.mean([M[(p, b)] for p in peps for b in ORDER])
    return {(p, b): M[(p, b)] - rm[p] - cm[b] + g for p in peps for b in ORDER}


def zscore(M: dict) -> dict:
    v = list(M.values())
    mu, sd = st.mean(v), (st.pstdev(v) or 1.0)
    return {k: (x - mu) / sd for k, x in M.items()}


def as_matrix(M: dict) -> np.ndarray:
    return np.array([[M.get((p, b), np.nan) for b in ORDER] for p in ORDER])


def metrics(M: dict, truth: dict) -> tuple[float, float, int]:
    rr, t3 = [], 0
    for p in ORDER:
        o = sorted(ORDER, key=lambda b: M[(p, b)])
        r = o.index(p) + 1
        rr.append(r)
        t3 += r <= 3
    cog = [M[k] for k in M if truth.get(k, {}).get("cognate") == "1"]
    non = [M[k] for k in M if truth.get(k, {}).get("measured") == "0"]
    a = sum(1.0 if x < y else 0.5 if x == y else 0.0 for x in cog for y in non) / (
        len(cog) * len(non))
    return st.mean(rr), a, t3


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}

    # --- ground truth: -log10(Kd) where measured, NaN elsewhere -----------------
    T = np.full((N, N), np.nan)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth.get((p, b))
            if r and r["measured"] == "1":
                kd = r.get("Kd_M") or r.get("kd_M") or ""
                try:
                    T[i, j] = -math.log10(float(kd))
                except (TypeError, ValueError):
                    T[i, j] = 7.0          # measured but unparsed: mark it as a mid value
    n_meas = int(np.isfinite(T).sum())

    # --- ours: ref2015 on our docked poses -------------------------------------
    O = {}
    for line in (ROOT / "logs/coventry_refine_fixed.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                O[(p, b[:-1])] = r["best_iface"]

    # --- co-folded: same scorer, different pose source -------------------------
    B, IP = {}, {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            k = (r["peptide"], r["binder"])
            v = r.get("best_iface")
            B[k] = FAIL_CAP if (v is None or v > FAIL_CAP) else v
            if r.get("iptm") is not None:
                IP[k] = -r["iptm"]

    cO, cB, cI = centre(O), centre(B), centre(IP)
    zB, zI = zscore(cB), zscore(cI)
    COMB = {k: 0.5 * (zB[k] + zI[k]) for k in zB if k in zI}

    panels = [
        ("A  Wu et al. 2025, Fig. 2B", T, None, "measured  −log₁₀(K$_d$)", True),
        ("B  our docked poses", as_matrix(cO), metrics(cO, truth), "ref2015, centred (REU)", False),
        ("C  co-folded poses, same scorer", as_matrix(cB), metrics(cB, truth),
         "ref2015, centred (REU)", False),
        ("D  + Boltz ipTM", as_matrix(COMB), metrics(COMB, truth), "combined z, centred", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 6.2))
    for ax, (title, Mx, met, cbar_lab, is_truth) in zip(axes, panels):
        if is_truth:
            masked = np.ma.masked_invalid(Mx)
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#ffffff")
            im = ax.imshow(masked, cmap=cmap, aspect="equal")
        else:
            # more negative = predicted tighter, so reverse the map to keep "dark = binds"
            lim = np.nanpercentile(np.abs(Mx), 97)
            im = ax.imshow(Mx, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        # ring every cell the paper actually measured, so panel B-D can be read against panel A
        for i, p in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                r = truth.get((p, b))
                if r and r["measured"] == "1":
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111111" if i == j else "#777777",
                                           lw=1.9 if i == j else 1.0, zorder=3))
        # Print the cognate's RANK on each diagonal cell. Shade alone does not show whether the
        # right answer won its row -- the rank does, and it is the number the metric is built on.
        if not is_truth:
            for i, p in enumerate(ORDER):
                order_i = sorted(ORDER, key=lambda b: Mx[i, ORDER.index(b)])
                rank = order_i.index(p) + 1
                # a strongly-scoring cognate cell is painted near-black blue, so dark text on
                # it is invisible -- pick the label colour from the cell's own depth
                deep = abs(Mx[i, i]) > 0.62 * lim
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
               if met else f"{n_meas} measured of 324 · white = no binding detected")
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, linespacing=1.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(cbar_lab, fontsize=8)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle("The Coventry grid: the scorer never changes, only where the peptide came from. "
                 "Bold ring = cognate, thin ring = measured cross-reactivity. Random is 9.50 / 0.500 / 3.",
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165)
    print(f"wrote {OUT}")
    for title, _, met, _, _ in panels:
        if met:
            print(f"  {title:34s} rank {met[0]:5.2f}  AUC {met[1]:.3f}  top3 {met[2]:2d}/18")


if __name__ == "__main__":
    main()
