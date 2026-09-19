#!/usr/bin/env python
"""Substitution model v2 — with structure. Does context beat the lookup table?

THE ONE QUESTION. v1 scored AUC 0.729 against BLOSUM62's 0.712 and moved our ddG MAE by
0.011 kcal/mol. It was not overfitted and it was not noise -- it survived every validity tier and
a shuffle control -- it was simply REDUNDANT, because with only wt->mut physicochemistry there is
nothing to learn beyond generic substitution chemistry, and BLOSUM already is that.

The missing ingredient is context. A lysine into a buried core and the same lysine on the surface
are the same substitution and completely different events; BLOSUM cannot tell them apart and
neither could v1. So the designs were folded (monomers, Boltz, plDDT ~0.9) and burial is now
available per position.

WHY BURIAL SPLITS A CONFOUND THE KILL-RATE ALONE CANNOT. A position with a high kill rate is
either structural (buried -- mutating it unfolds the design, so binding is lost as a side effect)
or interfacial (exposed, yet mutating it still kills binding, which can only be because it was
touching the target). Those are different physics and only the second is what a BINDING model
should learn. v1 pooled them.

THE BAR IS UNCHANGED AND NON-NEGOTIABLE, because v1 failed it:
  1  beat BLOSUM62 alone by a margin worth having
  2  move the ddG MAE on 1,129 SKEMPI peptide mutations
Leave-one-target-out throughout, so every fold predicts a target it has never seen.

Usage: cao_ssm_train_v2.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SSM = ROOT / "data/cao_ssm.csv"
POS = ROOT / "data/cao_ssm_positions.csv"
FOLDLOG = ROOT / "logs/cao_fold.jsonl"
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


def base_feats(w, m):
    return [KD[m] - KD[w], VOL[m] - VOL[w], CHG.get(m, 0) - CHG.get(w, 0),
            HELIX[m] - HELIX[w], KD[w], VOL[w], CHG.get(w, 0), HELIX[w],
            KD[m], VOL[m], CHG.get(m, 0), HELIX[m],
            float(m in AROM) - float(w in AROM),
            float(m == "P"), float(m == "G"), float(w == "P"), float(w == "G"),
            float(m == "C"), float(w == "C"),
            abs(KD[m] - KD[w]), abs(VOL[m] - VOL[w])]


def burial_of(pdb: Path) -> dict[int, float]:
    """Neighbour count per residue, normalised — a cheap, robust burial proxy."""
    xyz, idx, cur = [], [], None
    for l in pdb.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CB":
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            idx.append(int(l[22:26]))
        elif l.startswith("ATOM") and l[12:16].strip() == "CA" and l[17:20].strip() == "GLY":
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            idx.append(int(l[22:26]))
    if len(xyz) < 5:
        return {}
    X = np.array(xyz)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    n = (d < 10.0).sum(1) - 1
    hi = max(n.max(), 1)
    return {i: float(v) / hi for i, v in zip(idx, n)}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    try:
        from Bio.Align import substitution_matrices
        BL = substitution_matrices.load("BLOSUM62")
    except Exception:  # noqa: BLE001
        BL = None

    folded = {}
    if FOLDLOG.exists():
        for l in FOLDLOG.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("pdb") and Path(r["pdb"]).exists():
                    folded[r["parent"]] = r["pdb"]
    print(f"{len(folded)} designs folded\n")
    if len(folded) < 10:
        print("too few structures yet — folding still running"); return
    bur = {p: burial_of(Path(f)) for p, f in folded.items()}

    rows = list(csv.DictReader(open(SSM)))
    hot = {(p["target"], p["parent"], int(p["pos"])): float(p["frac_killed"])
           for p in csv.DictReader(open(POS))}
    seen = defaultdict(set)
    for r in rows:
        if r["mut_aa"] in AAS:
            seen[(r["target"], r["parent"], int(r["pos"]))].add(r["mut_aa"])
    wt = {k: (set(AAS) - v).pop() for k, v in seen.items() if len(set(AAS) - v) == 1}

    Xb, Xs, y, g, wm = [], [], [], [], []
    for r in rows:
        pos = int(r["pos"]); k = (r["target"], r["parent"], pos)
        if pos == 0 or r["mut_aa"] not in AAS or k not in wt or k not in hot:
            continue
        if hot[k] < 0.6 or r["parent"] not in bur:
            continue
        b = bur[r["parent"]].get(pos)
        if b is None:
            continue
        w, m = wt[k], r["mut_aa"]
        if w == m:
            continue
        f = base_feats(w, m)
        Xb.append(f)
        # structure: burial, and the interactions BLOSUM structurally cannot express
        Xs.append(f + [b, b * abs(KD[m] - KD[w]), b * abs(VOL[m] - VOL[w]),
                       b * abs(CHG.get(m, 0) - CHG.get(w, 0)),
                       b * float(m == "P"), (1 - b) * abs(KD[m] - KD[w])])
        y.append(int(r["unmeasurable"])); g.append(r["target"]); wm.append((w, m))
    Xb, Xs, y, g = np.array(Xb, float), np.array(Xs, float), np.array(y), np.array(g)
    print(f"{len(y)} rows with structure, {len(set(g))} targets, "
          f"{100 * y.mean():.0f}% abolish binding\n")
    if len(y) < 500 or len(set(g)) < 3:
        print("not enough structured rows yet"); return

    def loto(X):
        out = []
        for t in sorted(set(g)):
            te = g == t
            if te.sum() < 50 or len(set(y[te])) < 2 or len(set(y[~te])) < 2:
                continue
            p = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                               random_state=0).fit(X[~te], y[~te]) \
                .predict_proba(X[te])[:, 1]
            out.append(roc_auc_score(y[te], p))
        return out

    print("BAR 1 — BEAT THE LOOKUP TABLE\n")
    print(f"  {'predictor':<40}{'mean AUC':>10}")
    if BL is not None:
        blv = np.array([-float(BL[w, m]) for w, m in wm])
        a0 = [roc_auc_score(y[g == t], blv[g == t]) for t in sorted(set(g))
              if (g == t).sum() >= 50 and len(set(y[g == t])) > 1]
        print(f"  {'BLOSUM62 alone, no training':<40}{st.mean(a0):>10.3f}")
    a1, a2 = loto(Xb), loto(Xs)
    print(f"  {'v1: sequence features only':<40}{st.mean(a1):>10.3f}")
    print(f"  {'v2: + burial and its interactions':<40}{st.mean(a2):>10.3f}")
    gain = st.mean(a2) - (st.mean(a0) if BL is not None else st.mean(a1))
    print(f"\n  v2 over BLOSUM: {gain:+.3f} AUC   (v1 managed +0.017)")

    print("\nWHAT BURIAL SPLITS — kill rate by position class\n")
    bvals = Xs[:, 21]
    for lo, hi, lbl in ((0.0, 0.4, "exposed  (likely interface)"),
                        (0.4, 0.7, "partial"),
                        (0.7, 1.01, "buried   (likely structural)")):
        m_ = (bvals >= lo) & (bvals < hi)
        if m_.sum() > 50:
            print(f"  {lbl:<32}{int(m_.sum()):>7} rows   {100 * y[m_].mean():>5.0f}% kill")

    # ---------------------------------------------------------------- BAR 2: the MAE
    print("\nBAR 2 — DOES IT MOVE OUR ddG MAE?\n")
    prows = []
    for r in csv.DictReader(open(PEP)):
        w, m = r["wt_aa"], r["mut_aa"]
        if w not in AAS or m not in AAS or w == m:
            continue
        try:
            km, kw = float(r["affinity_mut"]), float(r["affinity_wt"])
            T = float(r.get("temp", 298) or 298)
        except (ValueError, KeyError):
            continue
        if not (0 < km < 1 and 0 < kw < 1):
            continue
        loc = (r.get("location") or "").upper()
        prows.append({"w": w, "m": m, "pdb": r["pdb"], "loc": loc,
                      "ddg": R_GAS * T * math.log(km / kw)})
    yp = np.array([r["ddg"] for r in prows], float)
    grp = np.array([r["pdb"] for r in prows])
    # SKEMPI location codes give the burial we need on the test side: COR buried, RIM/SUP exposed
    bmap = {"COR": 0.85, "SUP": 0.45, "RIM": 0.35, "INT": 0.75, "SUR": 0.15}
    bp = np.array([bmap.get(r["loc"][:3], 0.5) for r in prows])
    Pb = np.array([base_feats(r["w"], r["m"]) for r in prows], float)
    Ps = np.column_stack([Pb, bp,
                          bp * np.array([abs(KD[r["m"]] - KD[r["w"]]) for r in prows]),
                          bp * np.array([abs(VOL[r["m"]] - VOL[r["w"]]) for r in prows]),
                          bp * np.array([abs(CHG.get(r["m"], 0) - CHG.get(r["w"], 0))
                                         for r in prows]),
                          bp * np.array([float(r["m"] == "P") for r in prows]),
                          (1 - bp) * np.array([abs(KD[r["m"]] - KD[r["w"]]) for r in prows])])
    mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                         random_state=0).fit(Xs, y)
    pk = mdl.predict_proba(Ps)[:, 1]
    print(f"  {len(prows)} SKEMPI peptide mutations")
    print(f"  corr(v2 predicted kill, measured ddG) = {corr(pk, yp):+.3f}   (v1 gave +0.195)")

    print(f"\n  {'model':<44}{'MAE':>9}{'r':>9}")
    print(f"  {'predict the median ddG':<44}"
          f"{np.abs(yp - np.median(yp)).mean():>9.3f}{'-':>9}")
    for nm, M in (("physchem only (our current kind)", Pb),
                  ("physchem + v1 sequence model", np.column_stack([Pb, pk])),
                  ("physchem + burial + v2 model", np.column_stack([Ps, pk]))):
        pred = np.zeros(len(yp))
        for tr, te in GroupKFold(n_splits=5).split(M, yp, grp):
            pred[te] = LinearRegression().fit(M[tr], yp[tr]).predict(M[te])
        print(f"  {nm:<44}{np.abs(pred - yp).mean():>9.3f}{corr(pred, yp):>9.3f}")


if __name__ == "__main__":
    main()
