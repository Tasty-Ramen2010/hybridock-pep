#!/usr/bin/env python
"""The grid as a SHORTLIST: only the top 5 predictions per row survive.

WHY THIS VIEW.  A full heat map shows 324 numbers, and a reader has to work out which of them we
would actually have acted on.  A screening tool does not hand you 324 numbers -- it hands you a
shortlist to put in the plate.  So this shows exactly that: for each peptide, the five binders we
would have tested, and nothing else.  Every surviving cell is numbered 1-5 by our own confidence
order, and a cell the paper measured is ringed.

Read it as a recall question: of the 18 cognate pairs, how many land in their peptide's top five?
Of the 12 measured cross-reactivities, how many? Picking 5 of 18 at random recovers 5.0 cognates.

The same three pose sources as the full figure, so the panels are directly comparable:
  A  our docked poses                     B  co-folded poses, same scorer
  C  + Boltz ipTM combined
All are double-centred -- specificity is the interaction term, and the per-peptide and per-binder
main effects are exactly what we cannot model, so they are removed before shortlisting.

Usage: coventry_top5_figure.py [out.png] [--k 5] [--axis row|col]
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
from matplotlib.patches import Rectangle

ROOT = Path("/home/igem/unknown_software")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args[0]) if args else ROOT / "docs/coventry_top5.png"
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5
AXIS = sys.argv[sys.argv.index("--axis") + 1] if "--axis" in sys.argv else "row"
# --no-rings removes the outline drawn around every measured pair. Those rings sit on the
# diagonal, so they can MAKE a diagonal look present whether or not the predictions put one
# there. Turning them off is the honest check: if the trend survives with no annotation
# pointing at it, the trend is in the data.
RINGS = "--no-rings" not in sys.argv
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
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


def shortlist(M: dict) -> np.ndarray:
    """Rank 1..K for the cells we would actually test, NaN for everything else."""
    S = np.full((N, N), np.nan)
    if AXIS == "row":
        for i, p in enumerate(ORDER):
            for r, b in enumerate(sorted(ORDER, key=lambda b: M[(p, b)])[:K], 1):
                S[i, ORDER.index(b)] = r
    else:
        for j, b in enumerate(ORDER):
            for r, p in enumerate(sorted(ORDER, key=lambda p: M[(p, b)])[:K], 1):
                S[ORDER.index(p), j] = r
    return S


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}

    O = {}
    for line in (ROOT / "logs/coventry_refine_fixed.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                O[(p, b[:-1])] = r["best_iface"]
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

    import os
    if os.environ.get("SEL_PANELS") == "cancel":
        # The cancellation view: the SAME energies before and after the main effects are
        # stripped, so the shortlist change is attributable to the estimator and nothing else.
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from hybridock_pep.scoring.cancellation import two_way
        import numpy as _np

        def _mat(M):
            return _np.array([[M[(p, b)] for b in ORDER] for p in ORDER], dtype=float)

        def _dict(A):
            return {(p, b): float(A[i, j]) for i, p in enumerate(ORDER)
                    for j, b in enumerate(ORDER)}

        panels = [
            ("A  raw Rosetta energy, no cancellation", O),
            ("B  after CANCELLATION (median polish)", _dict(two_way(_mat(O), robust=True))),
            ("C  co-folded poses + cancellation",
             _dict(two_way(_mat(B), robust=True, scale=True))),
        ]
    else:
        panels = [("A  our docked poses", cO),
                  ("B  co-folded poses, same scorer", cB),
                  ("C  + Boltz ipTM", COMB)]

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.4))
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("#F4F6F8")

    for ax, (title, M) in zip(axes, panels):
        S = shortlist(M)
        ax.imshow(np.ma.masked_invalid(S), cmap=cmap, vmin=0.4, vmax=K + 1.6, aspect="equal")
        cog_hit = cross_hit = 0
        for i, p in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                t = truth.get((p, b))
                listed = np.isfinite(S[i, j])
                if t and t["measured"] == "1":
                    is_cog = i == j
                    if RINGS:
                        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor="#B3261E" if not listed else "#0B5394",
                                               lw=2.1 if is_cog else 1.3, zorder=3))
                    if listed:
                        cog_hit += is_cog
                        cross_hit += not is_cog
                if listed:
                    ax.text(j, i, str(int(S[i, j])), ha="center", va="center", zorder=4,
                            fontsize=6.4, fontweight="bold",
                            color="#FFFFFF" if S[i, j] <= 2 else "#1B3A57")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(ORDER, rotation=90, fontsize=7.5)
        ax.set_yticklabels(ORDER, fontsize=7.5)
        ax.set_xlabel("binder", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("peptide", fontsize=9)
        ax.set_title(f"{title}\ncognates recovered {cog_hit}/18 · "
                     f"cross-reactives {cross_hit}/12", fontsize=10, linespacing=1.6)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle(
        f"If we handed over a shortlist: the top {K} binders per peptide, and nothing else. "
        + (f"Blue ring = measured pair we caught, red ring = measured pair we missed. "
           if RINGS else "No annotation: nothing is drawn around the true pairs. ")
        + f"Random shortlisting recovers {K} of 18 cognates.",
        fontsize=11, y=0.975)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.subplots_adjust(bottom=0.13)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}  (top-{K} per {AXIS})")
    for title, M in panels:
        S = shortlist(M)
        cog = sum(1 for i in range(N) if np.isfinite(S[i, i]))
        cross = sum(1 for i, p in enumerate(ORDER) for j, b in enumerate(ORDER)
                    if i != j and np.isfinite(S[i, j])
                    and truth.get((p, b), {}).get("measured") == "1")
        print(f"  {title:34s} cognates {cog:2d}/18   cross-reactives {cross:2d}/12")


if __name__ == "__main__":
    main()
