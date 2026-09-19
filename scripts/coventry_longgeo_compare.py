#!/usr/bin/env python
"""Does longgeo help on the class it was actually built for — designed repeat grooves?

WHY THIS EXISTS AFTER A NULL BENCHMARK. longgeo went 189/342 on RecentSet (sign p = 0.058, mean
dRMSD -0.006A), which is a wash. But RecentSet is NATURAL complexes, and the longgeo pool was cut
for long peptides and high-enclosure grooves -- a class RecentSet barely contains (18 long and
very-long cells of 345). Coventry's eighteen de novo repeat-groove binders ARE that class. So the
benchmark bounds longgeo on natural targets and is simply silent about the target class; this is
the measurement that is not.

THE COMPARISON IS HELD TIGHT. Both arms are scored through the identical extractor -- same hard
repack, no soft-repulsive pre-pass, same best-of-N aggregation -- and put through the identical
median-polish cancellation. The generator checkpoint is the only thing that differs.

ONE CONFOUND IS DELIBERATELY NOT CONTROLLED, AND IT MATTERS. The longgeo arm was docked against
binders_best (AF3 models for n3/n7/pc21/pc26, where Boltz and AF3 agree against our ESMFold model
being 5.7-16.9A off); the existing arm was docked against the original ESMFold set. So a
difference in those four columns could be the receptor rather than the checkpoint. The columns
are therefore reported split -- the fourteen clean columns are the honest checkpoint comparison,
and the four rebuilt ones are reported separately rather than averaged in silently.

Usage: coventry_longgeo_compare.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
REBUILT = {"n3", "n7", "pc21", "pc26"}


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def grid_of(path: Path, key: str = "i_total"):
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = float(r["best"].get(key, 0.0))
    miss = [(p, b) for p in ORDER for b in ORDER if (p, b) not in rows]
    M = np.array([[rows.get((p, b), np.nan) for b in ORDER] for p in ORDER])
    return M, miss


def signtest(w: int, n: int) -> float:
    if n == 0:
        return float("nan")
    k = min(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def grade(S, keep):
    cog = np.array([S[i, i] for i in range(N)])
    non = np.array([S[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
    auc = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
    rr = [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]
    return auc, rr


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    keep = np.zeros((N, N), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            keep[i, j] = (i == j) or truth.get((p, b), {}).get("measured") == "0"

    A, ma = grid_of(ROOT / "logs/sel_coventry_features_hard.jsonl")
    B, mb = grid_of(ROOT / "logs/sel_coventry_features_longgeo.jsonl")
    print("longgeo re-dock of the Coventry grid vs the current longft_tanh arm.")
    print("Same extractor, same cancellation; the RAPiDock checkpoint is the variable.\n")
    if mb:
        print(f"  longgeo grid incomplete: {len(mb)} of 324 cells missing — numbers below "
              f"treat them as unscored\n")

    both = np.isfinite(A) & np.isfinite(B)
    print(f"  cells scored in both arms: {int(both.sum())}/324")

    # raw interface energy, cell by cell -- does longgeo simply find deeper poses?
    d = (B - A)[both]
    w = int((d < 0).sum())
    print(f"\nRAW INTERFACE ENERGY (lower is better), paired per cell")
    print(f"  longgeo deeper on {w}/{int(both.sum())} cells ({100 * w / max(int(both.sum()), 1):.0f}%)"
          f"   sign p = {signtest(w, int(both.sum())):.4f}")
    print(f"  mean dREU {d.mean():+.2f}   median dREU {np.median(d):+.2f}")

    cog = [(i, i) for i in range(N) if both[i, i]]
    if cog:
        dc = np.array([B[i, j] - A[i, j] for i, j in cog])
        wc = int((dc < 0).sum())
        print(f"  on the 18 COGNATE cells: longgeo deeper on {wc}/{len(cog)}, "
              f"mean dREU {dc.mean():+.2f}")

    if mb:
        print("\n(ranking metrics wait for the full grid)")
        return

    print("\nRANKING AFTER CANCELLATION — what the grid is actually judged on\n")
    print(f"  {'arm':<34}{'AUC':>8}{'rank':>8}{'t3':>5}{'t5':>5}")
    res = {}
    for name, M in (("longft_tanh (current)", A), ("longgeo re-dock", B)):
        auc, rr = grade(polish(M), keep)
        res[name] = (auc, rr)
        print(f"  {name:<34}{auc:>8.3f}{st.mean(rr):>8.2f}"
              f"{sum(r <= 3 for r in rr):>5}{sum(r <= 5 for r in rr):>5}")

    ra = res["longft_tanh (current)"][1]
    rb = res["longgeo re-dock"][1]
    better = sum(1 for x, y in zip(rb, ra) if x < y)
    worse = sum(1 for x, y in zip(rb, ra) if x > y)
    print(f"\n  cognate rank improved on {better}/18 peptides, worsened on {worse}, "
          f"unchanged on {18 - better - worse}")
    print(f"  sign p = {signtest(better, better + worse):.4f}")

    clean = [j for j, b in enumerate(ORDER) if b not in REBUILT]
    print(f"\n  SPLIT BY RECEPTOR PROVENANCE (the one uncontrolled confound):")
    for label, cols in (("14 columns both arms share (ESMFold)", clean),
                        ("4 columns longgeo got as AF3", [j for j in range(N) if j not in clean])):
        sub = [(i, j) for i in range(N) for j in cols if both[i, j]]
        if not sub:
            continue
        dd = np.array([B[i, j] - A[i, j] for i, j in sub])
        ww = int((dd < 0).sum())
        print(f"    {label:<40}{ww:>4}/{len(sub):<5} deeper   mean dREU {dd.mean():+7.2f}")

    print("\n  per-peptide cognate rank (tanh -> longgeo):")
    for i, p in enumerate(ORDER):
        mark = "  better" if rb[i] < ra[i] else ("  worse" if rb[i] > ra[i] else "")
        print(f"    {p:<7}{ra[i]:>3} -> {rb[i]:<3}{mark}")


if __name__ == "__main__":
    main()
