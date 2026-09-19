#!/usr/bin/env python
"""Does the cancellation model beat plain Rosetta energy units on the grid?

The bar is ref2015 interface energy with no cancellation at all: mean cognate rank 8.28 of 18,
AUC 0.584. The bar that has to be cleared to claim the cancellation model is worth building is
plain MEAN double-centring, which already reaches 6.67 and 0.667 for free.

Every variant below is unsupervised -- nothing is fitted to the Coventry labels, so there is no
train/test question and no leakage. They differ only in how the main effects are estimated.

Usage: coventry_cancel_eval.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def load(path: str, field: str = "best_iface", cap: float | None = None) -> np.ndarray:
    M = {}
    for line in (ROOT / path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "name" in r:
                a, b = r["name"].split("__")
                b = b[:-1] if b.endswith("B") else b
            else:
                a, b = r["peptide"], r["binder"]
            v = r.get(field)
            if v is not None:
                M[(a, b)] = cap if (cap is not None and v > cap) else v
    if len(M) < len(ORDER) ** 2:
        return np.array([])
    return np.array([[M[(p, b)] for b in ORDER] for p in ORDER], dtype=float)


def report(name: str, A: np.ndarray, truth: dict) -> dict:
    rr, t1, t3 = [], 0, 0
    for i, p in enumerate(ORDER):
        o = sorted(range(len(ORDER)), key=lambda j: A[i, j])
        r = o.index(i) + 1
        rr.append(r); t1 += r == 1; t3 += r <= 3
    cc = []
    for j, b in enumerate(ORDER):
        o = sorted(range(len(ORDER)), key=lambda i: A[i, j])
        cc.append(o.index(j) + 1)
    cog = [A[i, i] for i in range(len(ORDER))]
    non = [A[i, j] for i, p in enumerate(ORDER) for j, b in enumerate(ORDER)
           if truth.get((p, b), {}).get("measured") == "0"]
    a = sum(1.0 if x < y else 0.5 if x == y else 0.0 for x in cog for y in non) / (
        len(cog) * len(non))
    top5 = sum(1 for i in range(len(ORDER))
               if i in sorted(range(len(ORDER)), key=lambda j: A[i, j])[:5])
    print(f"  {name:<44}{st.mean(rr):>6.2f}{st.mean(cc):>6.2f}{t1:>5d}{t3:>5d}"
          f"{top5:>6d}{a:>8.3f}")
    return {"row": st.mean(rr), "col": st.mean(cc), "top1": t1, "top3": t3,
            "top5": top5, "auc": a}


def main() -> None:
    from hybridock_pep.scoring.cancellation import (cross_scorer, mean_centre, median_polish,
                                                    two_way)

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    REF = load("logs/coventry_refine_fixed.jsonl")
    DG = load("logs/coventry_dg.jsonl")
    COF = load("logs/coventry_boltz_refine.jsonl", cap=50.0)

    print("Coventry 18x18 — unsupervised cancellation, nothing fitted to these labels")
    print(f"  {'estimator':<44}{'row':>6}{'col':>6}{'t1':>5}{'t3':>5}{'top5':>6}{'AUC':>8}")
    print(f"  {'random':<44}{9.50:>6.2f}{9.50:>6.2f}{1:>5}{3:>5}{5:>6}{0.500:>8.3f}")
    print("\n  -- ref2015 interface energy, the thing we are trying to replace --")
    res = {}
    res["raw"] = report("no cancellation (raw REU)", REF, truth)
    res["mean"] = report("MEAN double-centring", mean_centre(REF), truth)
    res["median"] = report("MEDIAN POLISH (robust)", two_way(REF, robust=True), truth)
    res["median_scale"] = report("median polish + MAD scale balancing",
                                 two_way(REF, robust=True, scale=True), truth)
    res["mean_scale"] = report("mean centring + MAD scale balancing",
                               two_way(REF, robust=False, scale=True), truth)

    if DG.size:
        print("\n  -- our calibrated dG on its own --")
        report("no cancellation", DG, truth)
        report("median polish", two_way(DG, robust=True), truth)
        print("\n  -- cross-scorer: effects from our dG, interaction from ref2015 --")
        for w in (0.25, 0.5, 1.0):
            res[f"cross{w}"] = report(f"cross_scorer, weight {w}",
                                      cross_scorer(REF, DG, weight=w), truth)

    if COF.size:
        print("\n  -- same cancellation applied to the co-folded arm --")
        report("co-folded, no cancellation", COF, truth)
        report("co-folded, mean centring", mean_centre(COF), truth)
        report("co-folded, MEDIAN POLISH", two_way(COF, robust=True), truth)
        report("co-folded, median polish + scale", two_way(COF, robust=True, scale=True), truth)

    # how much of the grid is main effect, and is the median estimate different?
    print("\n  variance accounting on ref2015 (how much IS main effect?)")
    for lab, fn in (("mean centring", mean_centre), ("median polish", lambda M: two_way(M, True))):
        R = fn(REF)
        print(f"     {lab:<22} interaction keeps "
              f"{100 * R.var() / REF.var():.1f}% of the raw variance")
    _, row, col, grand = median_polish(REF)
    print(f"     median-polish effects: peptide sd {row.std():.1f} REU, "
          f"binder sd {col.std():.1f} REU, grand {grand:.1f}")


if __name__ == "__main__":
    main()
