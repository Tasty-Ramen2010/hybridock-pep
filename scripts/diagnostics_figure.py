#!/usr/bin/env python
"""Where the remaining error actually lives: sampling, capacity, or data.

Four panels, one per line of evidence, all measured on the same 387-complex bench.

  a  SAMPLING. Oracle best-of-N on the 58 long complexes at N=24 and N=100. If more draws still
     buy points, N=24 was never the ceiling.
  b  SHARED FAILURE. What fraction of our misses are also missed by the same architecture with
     different training (RAPiDock) and by a model 466x larger that was TRAINED ON THESE EXACT
     STRUCTURES (Boltz-2, cutoff 2023-06-01 vs a 2020-2023 bench). A failure that a memorising
     giant also makes is not a capacity failure.
  c  DATA QUALITY. The complexes no method solves, against everything else, on interface burial
     and peptide length. This is the signature the corpus filter was built from.
  d  ESMFOLD, decomposed. A 3.5-billion-parameter folder is on this bench as the only co-folding
     arm that is leakage-clean (cutoff 2020-05-01; 0 of 387 predate it). It has no multimer mode,
     so the receptor and peptide are joined by a 25-Gly linker. Separating how well it folds the
     pocket from where it puts the peptide says which half fails.

Usage: diagnostics_figure.py [--out docs/figures]
"""
from __future__ import annotations

import argparse, csv, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/igem/unknown_software")
BLUE, GREY, PURPLE, PINK, GREEN, ORANGE = "#1f6feb", "#6e7781", "#7048e8", "#d6336c", "#2f9e44", "#b07d2b"


def ca(p, chain=None):
    seen, x = set(), []
    for l in Path(p).read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        if chain and l[21] != chain:
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        x.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(x)


def rms(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def kab(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    a = ap.parse_args()

    c = np.load(ROOT / "logs/balanced_figures_cache.npz", allow_pickle=True)
    R = {k: c[k].item() for k in c.files}
    rows = list(csv.DictReader(open(ROOT / "data/bench_balanced_length.csv")))
    meta = {r["name"]: r for r in rows}
    bucket = {r["name"]: ("long" if r["length_bucket"] in ("long", "vlong") else r["length_bucket"])
              for r in rows}

    def parse(p):
        d = {}
        for l in open(p):
            m = re.match(r"\[(\S+)\] partial=\S+ n=\d+ dt=\d+s best_direct=([\d.]+)A", l)
            if m:
                d[m.group(1)] = float(m.group(2))
        return d
    n100 = {arm: parse(ROOT / f"logs/long_n100_{arm}.log")
            for arm in ("hybridock_ft", "rapidock_og")}

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.4))

    # ---- a  sampling
    A = ax[0][0]
    shared = sorted(set(n100["hybridock_ft"]) & set(n100["rapidock_og"]) & set(R["hybridock_ft"]))
    NS = [1, 2, 4, 8, 12, 16, 24]
    for arm, lab, col in (("hybridock_ft", "HybriDock-Pep", BLUE), ("rapidock_og", "RAPiDock", GREY)):
        ys = [100 * np.mean([min([x for x in R[arm][n][:N] if np.isfinite(x)] or [9e9]) <= 5
                             for n in shared]) for N in NS]
        A.plot(NS, ys, color=col, lw=1.7, label=lab)
        y100 = 100 * np.mean([n100[arm][n] <= 5 for n in shared])
        A.plot([100], [y100], marker="o", ms=6, color=col)
        A.annotate(f"{y100:.0f}%", (100, y100), textcoords="offset points", xytext=(-4, 7),
                   ha="right", fontsize=7.5, color=col, fontweight="bold")
        A.plot([24, 100], [ys[-1], y100], color=col, lw=1.7, ls=":")
    A.set_xscale("log"); A.set_xticks([1, 2, 4, 8, 16, 24, 100])
    A.set_xticklabels(["1", "2", "4", "8", "16", "24", "100"])
    A.set_xlabel("poses sampled (N, log scale)"); A.set_ylabel("≤5 Å success (%)")
    A.set_title(f"a   sampling is NOT saturated — long peptides, n={len(shared)}",
                loc="left", fontweight="bold")
    A.set_ylim(0, 80); A.grid(alpha=.25, lw=.5); A.legend(fontsize=7.5, frameon=False, loc="lower right")

    # ---- b  shared failure
    B = ax[0][1]
    labs, ours, withra, withbo = [], [], [], []
    for b, thr in (("short", 2.0), ("med", 2.0), ("long", 5.0)):
        names = [n for n in R["hybridock_ft"] if bucket.get(n) == b]
        F = {n for n in names if min([x for x in R["hybridock_ft"][n] if np.isfinite(x)] or [9e9]) > thr}
        Fr = {n for n in names if min([x for x in R["rapidock_og"][n] if np.isfinite(x)] or [9e9]) > thr}
        bn = {n for n in F if R.get("boltz2", {}).get(n)}
        Fb = {n for n in bn if min(R["boltz2"][n]) > thr}
        labs.append(f"{b}\n(≤{thr:g} Å)  n={len(F)}")
        ours.append(100.0)
        withra.append(100 * len(F & Fr) / max(len(F), 1))
        withbo.append(100 * len(Fb) / max(len(bn), 1))
    x = np.arange(3); w = 0.27
    B.bar(x - w, ours, w, color="#dee2e6", label="our failures")
    B.bar(x, withra, w, color=GREY, label="also RAPiDock (7.6 M, same arch)")
    B.bar(x + w, withbo, w, color=PINK, hatch="///", edgecolor="white",
          label="also Boltz-2 (466×, trained on these)")
    for xi, v in zip(x + w, withbo):
        B.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5, color=PINK, fontweight="bold")
    B.set_xticks(x); B.set_xticklabels(labs, fontsize=7.5)
    B.set_ylabel("% of our failures"); B.set_ylim(0, 112)
    B.set_title("b   our failures are other models' failures too", loc="left", fontweight="bold")
    B.legend(fontsize=7, frameon=False, loc="lower left"); B.grid(alpha=.25, axis="y", lw=.5)

    # ---- c  data quality signature
    C = ax[1][0]
    thr = {"short": 2.0, "med": 2.0, "long": 5.0}
    missed, rest = [], []
    for n, r in meta.items():
        if n not in R["hybridock_ft"] or not R.get("boltz2", {}).get(n):
            continue
        t = thr[bucket[n]]
        bad = (min(R["hybridock_ft"][n]) > t and min(R["rapidock_og"][n]) > t
               and min(R["boltz2"][n]) > t)
        P, Rc = ca(r["peptide_pdb"]), ca(r["receptor"])
        if len(P) < 3 or len(Rc) < 3:
            continue
        D = np.linalg.norm(Rc[:, None] - P[None], axis=2)
        (missed if bad else rest).append((float((D < 8.0).sum()) / len(P), len(P)))
    m, s = np.array(missed), np.array(rest)
    parts = C.violinplot([s[:, 0], m[:, 0]], positions=[0, 1], widths=.7, showmedians=True)
    for pc, col in zip(parts["bodies"], (BLUE, PINK)):
        pc.set_facecolor(col); pc.set_alpha(.45)
    C.set_xticks([0, 1])
    C.set_xticklabels([f"solved by someone\nn={len(s)}", f"NO method solves\nn={len(m)}"], fontsize=8)
    C.set_ylabel("interface burial per peptide residue")
    C.set_title("c   the unsolvable set is shallower, not just harder", loc="left", fontweight="bold")
    C.grid(alpha=.25, axis="y", lw=.5)
    C.text(.5, .95, f"median {np.median(s[:,0]):.2f} vs {np.median(m[:,0]):.2f}\n"
                    f"peptide {np.median(s[:,1]):.0f} vs {np.median(m[:,1]):.0f} aa",
           transform=C.transAxes, ha="center", va="top", fontsize=7.5,
           bbox=dict(fc="white", ec="#ced4da", lw=.6))

    # ---- d  ESMFold decomposed
    D_ = ax[1][1]
    rec, pep = [], []
    for r in rows:
        f = ROOT / f"datasets/esmfold_pred/{r['name']}_model_0.pdb"
        if not f.exists():
            continue
        Ax, Bx = ca(f, "A"), ca(f, "B")
        Cx, Px = ca(r["receptor"]), ca(r["peptide_pdb"])
        if len(Ax) != len(Cx) or len(Ax) < 3 or len(Bx) < 3:
            continue
        Rt, t = kab(Ax, Cx)
        rec.append(rms((Rt @ Ax.T).T + t, Cx)); pep.append(rms((Rt @ Bx.T).T + t, Px))
    rec, pep = np.array(rec), np.array(pep)
    good = rec < 2.0
    D_.scatter(rec[~good], pep[~good], s=9, c="#ced4da", edgecolors="none", label="pocket fold ≥2 Å")
    D_.scatter(rec[good], pep[good], s=11, c=PURPLE, edgecolors="none", label="pocket fold <2 Å")
    D_.axhline(5, color=GREEN, lw=1, ls="--")
    D_.text(0.2, 5.6, "peptide ≤5 Å", fontsize=7, color=GREEN)
    D_.set_xscale("log"); D_.set_yscale("log")
    D_.set_xlabel("ESMFold pocket fold, Cα RMSD (Å)")
    D_.set_ylabel("peptide placement, Cα RMSD (Å)")
    D_.set_title("d   ESMFold folds some pockets, docks almost none", loc="left", fontweight="bold")
    D_.legend(fontsize=7, frameon=False, loc="lower left")
    D_.grid(alpha=.25, lw=.5, which="both")
    D_.text(.97, .04, f"even where the fold is right (<2 Å, n={int(good.sum())}),\n"
                      f"only {int((pep[good]<=5).sum())} of {int(good.sum())} peptides land ≤5 Å",
            transform=D_.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(fc="white", ec="#ced4da", lw=.6))

    fig.tight_layout()
    out = Path(a.out) / "where_the_error_lives.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
