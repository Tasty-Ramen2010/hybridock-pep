#!/usr/bin/env python
"""Is the AUC 0.729 real, or an artifact? Four ways to kill it, plus MAE before/after.

A cross-target AUC of 0.729 with a shuffle control at 0.501 is necessary but nowhere near
sufficient, and there are four specific ways a number like that turns out to be worthless. Each
gets its own test here, and the model has to survive all of them.

  1  IS IT A LOOKUP TABLE? If BLOSUM62 alone -- a fifty-year-old substitution matrix, no training
     at all -- scores the same, then the model is an expensive way to reproduce something we could
     have read off a chart. This is the test that matters most, because the model's single most
     impressive property (corr -0.820 with BLOSUM) is also exactly what you would see if it had
     learned nothing BUT BLOSUM.

  2  IS IT THE CLASS IMBALANCE? 84% of the filtered rows abolish binding. AUC is insensitive to
     base rate but precision-recall AUC is not, and a model that is right mostly because "almost
     everything kills binding" will show it there.

  3  DOES IT SURVIVE THE DATA BRIAN ACTUALLY TRUSTS? He shipped three validity tiers and uses only
     the strictest 25 himself, saying the looser sets get "iffy". If the result only exists in the
     unlisted majority, it is measuring assay noise.

  4  DOES IT TRANSFER OFF ITS OWN DOMAIN? Trained on designed mini-binders against their targets,
     tested on SKEMPI single-residue PEPTIDE mutations with measured ddG -- a different assay, a
     different kind of molecule, and the exact domain our scorer works in. This is where the e436
     head died, and it is the only test that says whether any of this helps US.

Then the number Ram asked for: our existing scorer's MAE on those peptide mutations, before and
after adding this model as a term. If MAE does not move, the AUC is interesting and useless.

Usage: ssm_artifact_check.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SSM = ROOT / "data/cao_ssm.csv"
POS = ROOT / "data/cao_ssm_positions.csv"
PEP = ROOT / "data/skempi_single_peptide.csv"
AAS = "ACDEFGHIKLMNPQRSTVWY"
R_GAS = 0.0019872
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


def feats(w, m):
    return [KD[m] - KD[w], VOL[m] - VOL[w], CHG.get(m, 0) - CHG.get(w, 0),
            HELIX[m] - HELIX[w], KD[w], VOL[w], CHG.get(w, 0), HELIX[w],
            KD[m], VOL[m], CHG.get(m, 0), HELIX[m],
            float(m in AROM) - float(w in AROM),
            float(m == "P"), float(m == "G"), float(w == "P"), float(w == "G"),
            float(m == "C"), float(w == "C"),
            abs(KD[m] - KD[w]), abs(VOL[m] - VOL[w])]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def build(tier_keep):
    rows = list(csv.DictReader(open(SSM)))
    hot = {(p["target"], p["parent"], int(p["pos"])): float(p["frac_killed"])
           for p in csv.DictReader(open(POS))}
    seen = defaultdict(set)
    for r in rows:
        if r["mut_aa"] in AAS:
            seen[(r["target"], r["parent"], int(r["pos"]))].add(r["mut_aa"])
    wt = {k: (set(AAS) - v).pop() for k, v in seen.items() if len(set(AAS) - v) == 1}
    X, y, g, wm = [], [], [], []
    for r in rows:
        pos = int(r["pos"]); k = (r["target"], r["parent"], pos)
        if pos == 0 or r["mut_aa"] not in AAS or k not in wt or k not in hot:
            continue
        if hot[k] < 0.6 or r["tier"] not in tier_keep:
            continue
        w, m = wt[k], r["mut_aa"]
        if w == m:
            continue
        X.append(feats(w, m)); y.append(int(r["unmeasurable"]))
        g.append(r["target"]); wm.append((w, m))
    return np.array(X, float), np.array(y), np.array(g), wm


def loto(X, y, g, model_fn):
    from sklearn.metrics import average_precision_score, roc_auc_score
    aucs, aps = [], []
    for t in sorted(set(g)):
        te = g == t
        if te.sum() < 50 or len(set(y[te])) < 2 or len(set(y[~te])) < 2:
            continue
        p = model_fn(X[~te], y[~te], X[te])
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p) - y[te].mean())
    return aucs, aps


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    try:
        from Bio.Align import substitution_matrices
        BL = substitution_matrices.load("BLOSUM62")
    except Exception:  # noqa: BLE001
        BL = None

    ALL = {"quite", "reasonable", "somewhat", ""}
    X, y, g, wm = build(ALL)
    print(f"{len(y)} rows, {len(set(g))} targets, {100 * y.mean():.0f}% abolish binding\n")

    def gb(Xtr, ytr, Xte):
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                              random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    print("TEST 1 — IS IT JUST A LOOKUP TABLE?\n")
    print(f"  {'predictor':<44}{'mean AUC':>10}{'PR-AUC lift':>13}")
    a, p = loto(X, y, g, gb)
    print(f"  {'the model (21 features, learned)':<44}{st.mean(a):>10.3f}{st.mean(p):>13.3f}")

    if BL is not None:
        blv = np.array([[-float(BL[w, m])] for w, m in wm])   # negated: dissimilar -> kills
        a2 = []
        for t in sorted(set(g)):
            te = g == t
            if te.sum() < 50 or len(set(y[te])) < 2:
                continue
            a2.append(roc_auc_score(y[te], blv[te, 0]))
        print(f"  {'BLOSUM62 alone, NO training':<44}{st.mean(a2):>10.3f}{'-':>13}")
    for nm, col in (("|hydrophobicity change| alone", 19), ("|volume change| alone", 20),
                    ("'is it proline' alone", 13)):
        a3 = []
        for t in sorted(set(g)):
            te = g == t
            if te.sum() < 50 or len(set(y[te])) < 2:
                continue
            a3.append(roc_auc_score(y[te], X[te, col]))
        print(f"  {nm:<44}{st.mean(a3):>10.3f}{'-':>13}")

    print("\nTEST 2 — CLASS IMBALANCE. PR-AUC lift is precision-recall AUC minus the base rate;")
    print("          a model riding the 84% base rate shows ~0 here.\n")
    print(f"  PR-AUC lift over base rate: {st.mean(p):+.3f}")

    print("\nTEST 3 — DOES IT SURVIVE THE DATA BRIAN ACTUALLY TRUSTS?\n")
    print(f"  {'validity tier':<44}{'rows':>8}{'targets':>9}{'mean AUC':>10}")
    for lbl, keep in (("all (incl. 163 unlisted parents)", ALL),
                      ("somewhat+ (65 parents)", {"quite", "reasonable", "somewhat"}),
                      ("reasonable+ (37)", {"quite", "reasonable"}),
                      ("quite only (25 — what he uses)", {"quite"})):
        Xs, ys, gs, _ = build(keep)
        if len(ys) < 200 or len(set(gs)) < 2:
            print(f"  {lbl:<44}{len(ys):>8}{len(set(gs)):>9}{'too few':>10}")
            continue
        aa, _ = loto(Xs, ys, gs, gb)
        print(f"  {lbl:<44}{len(ys):>8}{len(set(gs)):>9}"
              f"{(st.mean(aa) if aa else float('nan')):>10.3f}")

    # ------------------------------------------------------- TEST 4: transfer + MAE
    print("\nTEST 4 — TRANSFER TO PEPTIDES, and the MAE question.\n")
    prows = []
    for r in csv.DictReader(open(PEP)):
        w, m = r["wt_aa"], r["mut_aa"]
        if w not in AAS or m not in AAS or w == m:
            continue
        try:
            km, kw, T = float(r["affinity_mut"]), float(r["affinity_wt"]), float(r["temp"])
        except (ValueError, KeyError):
            try:
                km, kw, T = float(r["affinity_mut"]), float(r["affinity_wt"]), 298.0
            except (ValueError, KeyError):
                continue
        if not (0 < km < 1 and 0 < kw < 1):
            continue
        prows.append({"w": w, "m": m, "pdb": r["pdb"], "loc": r.get("location", ""),
                      "ddg": R_GAS * T * math.log(km / kw)})
    if not prows:
        print("  no usable peptide mutations"); return
    Xp = np.array([feats(r["w"], r["m"]) for r in prows], float)
    yp = np.array([r["ddg"] for r in prows], float)
    print(f"  {len(prows)} SKEMPI peptide single mutations, "
          f"ddG {yp.min():+.1f} to {yp.max():+.1f}, median {np.median(yp):+.2f} kcal/mol")

    mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                         random_state=0).fit(X, y)
    pk = mdl.predict_proba(Xp)[:, 1]
    print(f"\n  corr(predicted kill probability, measured ddG) = {corr(pk, yp):+.3f}")
    print("     positive is correct: a mutation the model calls fatal should DESTABILISE "
          "(ddG > 0)")
    core = [i for i, r in enumerate(prows) if r["loc"].upper().startswith("COR")]
    if len(core) > 30:
        print(f"  restricted to buried-core mutations (n={len(core)}): "
              f"{corr(pk[core], yp[core]):+.3f}")
    if BL is not None:
        blp = np.array([-float(BL[r['w'], r['m']]) for r in prows])
        print(f"  BLOSUM62 alone on the same rows:               {corr(blp, yp):+.3f}")

    print("\n  MAE on ddG, before and after adding this model as a term:")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    grp = np.array([r["pdb"] for r in prows])
    base = np.abs(yp - np.median(yp)).mean()
    print(f"    {'predict the median ddG (no model)':<44}{base:>8.3f} kcal/mol")
    for nm, M in (("physchem features only (our current kind)", Xp),
                  ("physchem + SSM kill probability", np.column_stack([Xp, pk]))):
        pred = np.zeros(len(yp))
        gkf = GroupKFold(n_splits=min(5, len(set(grp))))
        for tr, te in gkf.split(M, yp, grp):
            pred[te] = LinearRegression().fit(M[tr], yp[tr]).predict(M[te])
        print(f"    {nm:<44}{np.abs(pred - yp).mean():>8.3f} kcal/mol   "
              f"r = {corr(pred, yp):+.3f}")


if __name__ == "__main__":
    main()
