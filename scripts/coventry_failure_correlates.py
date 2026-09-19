#!/usr/bin/env python
"""What the rows we still miss have in common — pose, pose quality, or the assay itself.

Three candidate explanations for a row whose true partner misses the top five, each measurable:

  POSE          the structure we scored is wrong. Proxied by how far our docked interaction term
                sits from the co-folded one for the same pair, and by whether switching arms
                rescues the row.
  POSE QUALITY  the structure is bad in a way the pipeline could have flagged: co-folding
                confidence, binder fold-RMSD at transplant, direction-prior axis quality, and
                closest contact after refinement.
  THE ASSAY     the row is intrinsically hard. A cognate that binds at 2 nM against neighbours
                that do not bind at all is an easy row; a cognate at 1 uM with three measured
                cross-reactive partners is not, and failing it says less about us. This is the
                cross-reactivity the paper itself reports, so it is the paper's own difficulty
                measure and not one we invented.

Reported as a per-row table and as correlations of rank against each covariate, separately for
each arm, because an explanation that only holds in one arm is an explanation about that arm.

Usage: coventry_failure_correlates.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
FEA_T = "i_total"


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def grid_of(path: Path, key: str) -> np.ndarray:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = float(r["best"].get(key, 0.0))
    return np.array([[rows.get((p, b), 0.0) for b in ORDER] for p in ORDER])


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}

    OUR = polish(grid_of(ROOT / "logs/sel_coventry_features_hard.jsonl", FEA_T))
    COF = polish(grid_of(ROOT / "logs/sel_cofold_features.jsonl", FEA_T))
    aux = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            aux[(r["peptide"], r["binder"])] = r
    IPT = np.array([[-(aux.get((p, b), {}).get("iptm") or 0.0) for b in ORDER] for p in ORDER])
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    COF_FULL = _z(COF) + _z(_mc(IPT))
    D = _z(OUR) - _z(COF)

    rank = lambda S, i: sorted(range(N), key=lambda j: S[i, j]).index(i) + 1
    rows = []
    print("PER-ROW.  rank is the true partner's position in its own row.\n")
    print(f"  {'pep':<7}{'OUR':>5}{'BOLTZ':>7}{'meas':>6}{'cogKd':>8}{'margin':>8}"
          f"{'dis@cog':>9}{'ipTM':>7}{'foldA':>7}{'axisQ':>7}{'minC':>7}")
    for i, p in enumerate(ORDER):
        meas = [b for b in ORDER if truth.get((p, b), {}).get("measured") == "1"]
        pk = lambda b: -math.log10(float(truth[(p, b)]["kd_M"]))
        cog = pk(p) if truth.get((p, p), {}).get("kd_M") else float("nan")
        others = [pk(b) for b in meas if b != p]
        margin = (cog - max(others)) if others else float("nan")
        a = aux.get((p, p), {})
        r = dict(pep=p, our=rank(OUR, i), cof=rank(COF_FULL, i), meas=len(meas), pkd=cog,
                 margin=margin, dis=float(D[i, i]),
                 iptm=a.get("iptm") or float("nan"), fold=a.get("fold_rmsd") or float("nan"),
                 axisq=a.get("axis_quality") or float("nan"),
                 minc=a.get("min_contact") or float("nan"),
                 ent=-sum((v / len(seqs[p])) * math.log2(v / len(seqs[p]))
                          for v in Counter(seqs[p]).values()), length=len(seqs[p]))
        rows.append(r)
        print(f"  {p:<7}{r['our']:>5}{r['cof']:>7}{r['meas']:>6}{cog:>8.2f}"
              f"{margin:>8.2f}{r['dis']:>9.2f}{r['iptm']:>7.2f}{r['fold']:>7.2f}"
              f"{r['axisq']:>7.2f}{r['minc']:>7.2f}")

    print("\nCORRELATION of rank with each covariate (positive = rank gets WORSE as it rises).")
    print("A covariate that only moves one arm is telling you about that arm.\n")
    print(f"  {'covariate':<36}{'OUR poses':>12}{'BOLTZ poses':>14}")
    for label, key in (("disagreement at the cognate cell", "dis"),
                       ("cognate affinity  -log10 Kd", "pkd"),
                       ("margin over best cross-reactant", "margin"),
                       ("measured partners in row (paper)", "meas"),
                       ("co-fold ipTM", "iptm"),
                       ("binder fold RMSD at transplant", "fold"),
                       ("direction-prior axis quality", "axisq"),
                       ("closest contact after refinement", "minc"),
                       ("peptide length", "length"),
                       ("peptide sequence entropy", "ent")):
        v = [r[key] for r in rows]
        print(f"  {label:<36}{pearson([r['our'] for r in rows], v):>12.3f}"
              f"{pearson([r['cof'] for r in rows], v):>14.3f}")

    both = [r for r in rows if r["our"] > 5 and r["cof"] > 5]
    one = [r for r in rows if (r["our"] > 5) != (r["cof"] > 5)]
    fine = [r for r in rows if r["our"] <= 5 and r["cof"] <= 5]
    print(f"\n  rows BOTH arms miss   (n={len(both)}): {', '.join(r['pep'] for r in both)}")
    print(f"  rows ONE arm rescues  (n={len(one)}): {', '.join(r['pep'] for r in one)}")
    print(f"  rows both get right   (n={len(fine)}): {', '.join(r['pep'] for r in fine)}")
    for label, grp in (("both miss", both), ("one rescues", one), ("both fine", fine)):
        if not grp:
            continue
        print(f"    {label:<14} ipTM {st.mean(r['iptm'] for r in grp):.2f}   "
              f"|disagree| {st.mean(abs(r['dis']) for r in grp):.2f}   "
              f"measured partners {st.mean(r['meas'] for r in grp):.1f}   "
              f"cognate pKd {st.mean(r['pkd'] for r in grp):.2f}")

    print("\n  arms agree on the ranking of rows?  "
          f"corr(OUR rank, BOLTZ rank) = {pearson([r['our'] for r in rows], [r['cof'] for r in rows]):+.3f}")


if __name__ == "__main__":
    main()
