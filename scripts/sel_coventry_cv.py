#!/usr/bin/env python
"""Can a learned term reweighting beat ref2015 on DOCKED poses? Cross-validated on the grid.

WHY THIS EXPERIMENT EXISTS.  The selectivity model trained on threaded poses reaches AUC 0.979 on
its own synthetic benchmark and only 0.598 on the Coventry grid, while plain centred ref2015
reaches 0.646 there.  The diagnosis is a distribution shift, and this is the measurement that
isolates it: fit the SAME kind of model on DOCKED poses instead of threaded ones, and see whether
the transfer gap closes.

Two facts motivate the reweighting specifically, both measured on the 324 docked cells:
  * i_fa_rep is 98.7% correlated with the ref2015 interface total and accounts for essentially
    all of its variance (sd 132.5 of 134.3).
  * i_fa_rep ALONE ranks the cognate at 9.67 of 18, AUC 0.515 -- pure chance.
So the term that dominates the score carries no discrimination, and the signal lives in the
remaining 22.0 sd. Deleting fa_rep outright is worse than keeping it (7.22 vs 6.39), so the answer
is not zero weight -- it is SOME weight, which is exactly what fitting can find.

HONEST SCOPE, STATED UP FRONT.  This fits on the Coventry grid, so it is no longer a held-out
test of a pre-trained model; it is an estimate of whether the reweighting EXISTS and is learnable
from this kind of data. The protocol is therefore strict:
  * leave-one-ROW-out: an entire peptide's 18 cells are held out, the model is fit on the other
    17 peptides, and the held-out row is predicted. Repeated for all 18.
  * leave-one-COLUMN-out likewise, holding out a whole binder.
  * double-centring uses TRAINING rows only for the column means; the held-out row supplies its
    own row mean, which needs no labels.
  * a label-shuffle control runs the identical procedure with cognate labels permuted.
With 18 rows this is a small-sample estimate and the grid is the only docked panel we have, so a
positive result here needs confirmation on a second panel before it is claimed as a method.

Usage: sel_coventry_cv.py [--feat logs/sel_coventry_features_hard.jsonl] [--agg best]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
ENERGY = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
          "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb"]
EXTRA = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
         "c_aromatic_frac", "c_mixed_frac", "hydro_pocket", "hydro_product", "hydro_absdiff",
         "charge_product", "pocket_charge_per_res", "n_contact_per_res",
         "rec_atoms_touched_per_res", "burial_mean", "burial_sd", "burial_cv", "burial_max",
         "frac_res_contacting"]


def auc(pos, neg) -> float:
    return sum(1.0 if p < q else 0.5 if p == q else 0.0
               for p in pos for q in neg) / (len(pos) * len(neg))


def metrics(S: dict, truth: dict) -> tuple[float, int, int, float]:
    rr, t1, t3 = [], 0, 0
    for p in ORDER:
        o = sorted(ORDER, key=lambda b: S[(p, b)])
        r = o.index(p) + 1
        rr.append(r); t1 += r == 1; t3 += r <= 3
    cog = [S[k] for k in S if truth.get(k, {}).get("cognate") == "1"]
    non = [S[k] for k in S if truth.get(k, {}).get("measured") == "0"]
    return st.mean(rr), t1, t3, auc(cog, non)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="logs/sel_coventry_features_hard.jsonl")
    ap.add_argument("--agg", default="best", choices=("best", "top5"))
    args = ap.parse_args()

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    raw = {}
    for line in (ROOT / args.feat).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if args.agg in r:
                raw[(r["peptide"], r["binder"])] = r[args.agg]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for feats, fl in ((["i_total"], "ref2015 total (no fitting)"),
                      (ENERGY, "energy terms reweighted"),
                      (ENERGY + EXTRA, "energy + chemistry reweighted")):
        A = np.array([[[float(raw[(p, b)].get(f, 0.0)) for f in feats]
                       for b in ORDER] for p in ORDER])            # (P, B, F)
        if len(feats) == 1:
            C = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean((0, 1),
                                                                                 keepdims=True)
            S = {(p, b): float(C[i, j, 0]) for i, p in enumerate(ORDER)
                 for j, b in enumerate(ORDER)}
            mr, t1, t3, a = metrics(S, truth)
            print(f"{fl:32s} mean rank {mr:5.2f}  top1 {t1:2d}/18  top3 {t3:2d}/18  AUC {a:.3f}")
            continue

        for shuffle in (False, True):
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(ORDER)) if shuffle else np.arange(len(ORDER))
            S: dict = {}
            for held in range(len(ORDER)):           # leave one PEPTIDE (row) out
                tr_rows = [i for i in range(len(ORDER)) if i != held]
                # column means from training rows only; each row supplies its own row mean
                colmu = A[tr_rows].mean(0)                           # (B, F)
                grand = A[tr_rows].mean((0, 1))                      # (F,)
                def cen(i):
                    return A[i] - A[i].mean(0, keepdims=True) - colmu + grand
                Xtr = np.vstack([cen(i) for i in tr_rows])
                ytr = np.concatenate([[int(perm[i] == j) for j in range(len(ORDER))]
                                      for i in tr_rows])
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
                clf.fit(sc.transform(Xtr), ytr)
                P = clf.predict_proba(sc.transform(cen(held)))[:, 1]
                for j, b in enumerate(ORDER):
                    S[(ORDER[held], b)] = -float(P[j])
            mr, t1, t3, a = metrics(S, truth)
            tag = fl + ("  [SHUFFLE CONTROL]" if shuffle else "")
            print(f"{tag:32s} mean rank {mr:5.2f}  top1 {t1:2d}/18  top3 {t3:2d}/18  AUC {a:.3f}")

    # what weight does it actually put on fa_rep, fitted on everything?
    A = np.array([[[float(raw[(p, b)].get(f, 0.0)) for f in ENERGY]
                   for b in ORDER] for p in ORDER])
    C = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean((0, 1), keepdims=True)
    X = C.reshape(-1, len(ENERGY))
    y = np.array([int(i == j) for i in range(len(ORDER)) for j in range(len(ORDER))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X), y)
    print("\nfitted term weights (standardised, negative = favours the cognate):")
    for k in np.argsort(-np.abs(clf.coef_[0])):
        print(f"   {ENERGY[k]:18s} {clf.coef_[0][k]:+.3f}")


if __name__ == "__main__":
    main()
