#!/usr/bin/env python
"""The gate e438 failed: is this correction term actually conditioned on WHICH residue you put in?

WHY THIS RUNS BEFORE ANYTHING IS BUILT ON THE NEW MODEL. E438 proposed exactly the scheme now
being revisited -- swap a hard amino acid for an easier one, score the easier pose, correct back
with a dDG head. It failed, and it failed silently: the head predicted a near-constant "any
mutation is mildly destabilising" offset. The tell was that its prediction for the intended
substitution and its prediction for an ARBITRARY DIFFERENT substitution correlated at r = 0.998.
It was not doing chemistry, it was adding roughly +1 kcal/mol to everything, and whether that
helped a given peptide was pure luck of which way that peptide's pre-existing error already
pointed (corr with prior error direction r = 0.657).

The memory from that session left an instruction: sanity-check any correction term's variance
against amino-acid identity BEFORE running a full pipeline. This is that check, on the new SSM
substitution model.

THREE TESTS, in increasing strictness:

  1  SPREAD WITHIN A POSITION. Hold the position fixed, vary only the incoming amino acid. A
     collapsed model gives all 19 substitutions nearly the same score; a real one separates them.
     Reported as the within-position standard deviation against the overall spread -- if
     within-position variance is a tiny share of the total, the model is only reading position,
     not chemistry.

  2  THE E438 TEST ITSELF. Correlate the prediction for substitution A against the prediction for
     a different substitution B at the same site. e438 scored r = 0.998 here. Anything above ~0.9
     is the same failure wearing a different hat.

  3  DOES IT AGREE WITH REAL CHEMISTRY? Correlate the model's per-amino-acid mean prediction with
     an external scale it never saw -- BLOSUM62 similarity to the residue being replaced, and
     hydrophobicity change. A model that has learned substitution physics should track these; one
     that memorised position identity should not.

Usage: ssm_identity_gate.py
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SSM = ROOT / "data/cao_ssm.csv"
POS = ROOT / "data/cao_ssm_positions.csv"
AAS = "ACDEFGHIKLMNPQRSTVWY"
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
      'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
      'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4,
       'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9,
       'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0}
CHG = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
HELIX = {'A': 1.42, 'L': 1.21, 'M': 1.45, 'E': 1.51, 'Q': 1.11, 'K': 1.16, 'R': 0.98,
         'H': 1.00, 'F': 1.13, 'I': 1.08, 'W': 1.08, 'V': 1.06, 'D': 1.01, 'T': 0.83,
         'S': 0.77, 'C': 0.70, 'N': 0.67, 'Y': 0.69, 'P': 0.57, 'G': 0.57}
AROM = set("FWY")


def feats(w: str, m: str) -> list[float]:
    return [KD[m] - KD[w], VOL[m] - VOL[w], CHG.get(m, 0) - CHG.get(w, 0),
            HELIX[m] - HELIX[w], KD[w], VOL[w], CHG.get(w, 0), HELIX[w],
            KD[m], VOL[m], CHG.get(m, 0), HELIX[m],
            float(m in AROM) - float(w in AROM),
            float(m == "P"), float(m == "G"), float(w == "P"), float(w == "G"),
            float(m == "C"), float(w == "C"),
            abs(KD[m] - KD[w]), abs(VOL[m] - VOL[w])]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = list(csv.DictReader(open(SSM)))
    hot = {(p["target"], p["parent"], int(p["pos"])): float(p["frac_killed"])
           for p in csv.DictReader(open(POS))}
    seen = defaultdict(set)
    for r in rows:
        if r["mut_aa"] in AAS:
            seen[(r["target"], r["parent"], int(r["pos"]))].add(r["mut_aa"])
    wt = {k: (set(AAS) - v).pop() for k, v in seen.items() if len(set(AAS) - v) == 1}

    X, y, grp, meta = [], [], [], []
    for r in rows:
        pos = int(r["pos"])
        k = (r["target"], r["parent"], pos)
        if pos == 0 or r["mut_aa"] not in AAS or k not in wt or k not in hot:
            continue
        if hot[k] < 0.6:
            continue
        w, m = wt[k], r["mut_aa"]
        if w == m:
            continue
        X.append(feats(w, m)); y.append(int(r["unmeasurable"]))
        grp.append(r["target"]); meta.append((k, w, m))
    X, y, grp = np.array(X, float), np.array(y), np.array(grp)

    # train on everything except one target, then probe ON that target -- never on training rows
    held = sorted(set(grp))[0]
    te = grp == held
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                         random_state=0).fit(X[~te], y[~te])
    print(f"trained on {int((~te).sum())} rows, probing on held-out target '{held}' "
          f"({int(te.sum())} rows)\n")

    # ---------------------------------------------------------- 1. spread within a position
    bypos = defaultdict(list)
    P = clf.predict_proba(X[te])[:, 1]
    for p, (k, w, m) in zip(P, [meta[i] for i in np.where(te)[0]]):
        bypos[k].append((m, float(p)))
    within = [np.std([v for _, v in vs]) for vs in bypos.values() if len(vs) >= 10]
    total = float(P.std())
    print("1. DOES THE PREDICTION MOVE WHEN ONLY THE INCOMING RESIDUE CHANGES?\n")
    print(f"   overall spread of predictions          sd = {total:.3f}")
    print(f"   median spread WITHIN a single position sd = {st.median(within):.3f}")
    print(f"   within-position share of total spread     = "
          f"{100 * st.median(within) / total:.0f}%")
    print("   " + ("-> identity matters: most of the variation is chemistry, not position."
                   if st.median(within) / total > 0.5 else
                   "-> mostly position, not chemistry. Same shape as the e438 collapse."))

    # ---------------------------------------------------------- 2. the exact e438 test
    pairs = [(vs[0][1], vs[len(vs) // 2][1]) for vs in bypos.values() if len(vs) >= 10]
    a, b = [x for x, _ in pairs], [y_ for _, y_ in pairs]
    r = corr(a, b)
    print(f"\n2. THE E438 TEST — prediction for substitution A vs a DIFFERENT substitution B,")
    print(f"   at the same site ({len(pairs)} positions)\n")
    print(f"   r = {r:+.3f}      (e438's collapsed head scored r = 0.998)")
    print("   " + ("-> PASSES. The two differ, so the model is reading the substitution."
                   if r < 0.90 else "-> FAILS. Same degeneracy as e438."))

    # ---------------------------------------------------------- 3. external chemistry axes
    try:
        from Bio.Align import substitution_matrices
        BL = substitution_matrices.load("BLOSUM62")
    except Exception:  # noqa: BLE001
        BL = None
    bl, kd, pr = [], [], []
    for vs, (k, w, m0) in zip(bypos.values(), bypos.keys()):
        pass
    for k, vs in bypos.items():
        w = wt[k]
        for m, p in vs:
            if BL is not None:
                try:
                    bl.append(float(BL[w, m]))
                except Exception:  # noqa: BLE001
                    continue
            else:
                bl.append(0.0)
            kd.append(abs(KD[m] - KD[w])); pr.append(p)
    print(f"\n3. DOES IT AGREE WITH CHEMISTRY IT NEVER SAW? ({len(pr)} substitutions)\n")
    if BL is not None:
        print(f"   corr(prediction of 'kills binding', BLOSUM62 similarity) = {corr(bl, pr):+.3f}")
        print("      negative is correct: a conservative swap should be LESS likely to kill")
    print(f"   corr(prediction, |hydrophobicity change|)                 = {corr(kd, pr):+.3f}")
    print("      positive is correct: a bigger chemical jump should be MORE likely to kill")

    print("\n   per-residue mean predicted kill probability on the held-out target:")
    byaa = defaultdict(list)
    for k, vs in bypos.items():
        for m, p in vs:
            byaa[m].append(p)
    order = sorted(byaa, key=lambda m: -st.mean(byaa[m]))
    for m in order[:5]:
        print(f"      {m}  {st.mean(byaa[m]):.3f}   (most disruptive)")
    for m in order[-5:]:
        print(f"      {m}  {st.mean(byaa[m]):.3f}   (best tolerated)")


if __name__ == "__main__":
    main()
