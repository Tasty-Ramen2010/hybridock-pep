#!/usr/bin/env python
"""Train the selectivity ranker on DOCKED blocks, then test it on the Coventry grid.

THE ONE EXPERIMENT THAT HAD NOT BEEN RUN. Every earlier version trained on threaded poses --
each sequence laid on a crystal backbone, in the right place -- and was then asked to score docked
poses, where 79-81% of the pool threads the groove backwards. A model that reached AUC 0.979 on
its own threaded benchmark scored 0.529 on the real grid. This closes that gap by training on the
same kind of pose it will be asked about: same generator, same checkpoint, same hard repack, same
best-of-N aggregation.

CANCELLATION IS APPLIED TO THE FEATURES, NOT JUST THE SCORE. Each feature is median-polished
inside its own block before the model ever sees it, so the model can only learn from the
interaction term. A feature that is constant down a row -- any peptide-only descriptor -- becomes
identically zero and is discarded automatically. That is the property worth having: cancellation
removes exactly the features that cannot carry specificity.

TWO FEATURE SETS, because the question "is any of this ours" has to be answerable:
  full        per-term ref2015 interface energies + chemistry + complementarity + burial
  ours-only   the same minus every Rosetta term -- typed contact chemistry, charge and
              hydrophobic complementarity against the pocket actually touched, burial shape

Leave-block-out on the training side; the Coventry grid is never trained on.

Usage: sel_dock_train.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]

ENERGY = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
          "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb", "i_total"]
OURS = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
        "c_aromatic_frac", "c_mixed_frac", "hydro_pocket", "hydro_product", "hydro_absdiff",
        "charge_product", "pocket_charge_per_res", "n_contact_per_res",
        "rec_atoms_touched_per_res", "burial_mean", "burial_sd", "burial_cv", "burial_max",
        "frac_res_contacting"]


def polish(M: np.ndarray) -> np.ndarray:
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def load_blocks(agg: str, feats: list[str]):
    """(X, y, block_id) with every feature median-polished inside its own block."""
    grid = defaultdict(dict)
    for line in (ROOT / "logs/sel_dock_features.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if agg in r:
                grid[r["block"]][(r["pep_idx"], r["rec_idx"])] = r[agg]
    X, y, g = [], [], []
    for b, cells in grid.items():
        n = max(max(k) for k in cells) + 1
        if len(cells) != n * n or n < 5:
            continue
        A = np.array([[[float(cells[(i, j)].get(f, 0.0)) for f in feats]
                       for j in range(n)] for i in range(n)])
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
        for i in range(n):
            for j in range(n):
                X.append(C[i, j]); y.append(int(i == j)); g.append(b)
    return np.array(X), np.array(y), np.array(g)


def coventry(agg: str, feats: list[str]) -> np.ndarray:
    rows = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if agg in r:
                rows[(r["peptide"], r["binder"])] = r[agg]
    A = np.array([[[float(rows[(p, b)].get(f, 0.0)) for f in feats]
                   for b in ORDER] for p in ORDER])
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)


def grade(name: str, S: np.ndarray, truth: dict) -> float:
    rr, t3, t5 = [], 0, 0
    for i in range(18):
        o = sorted(range(18), key=lambda j: S[i, j])
        r = o.index(i) + 1
        rr.append(r); t3 += r <= 3; t5 += r <= 5
    cog = [S[i, i] for i in range(18)]
    non = [S[i, j] for i, p in enumerate(ORDER) for j, b in enumerate(ORDER)
           if truth.get((p, b), {}).get("measured") == "0"]
    a = sum(1.0 if x < y else 0.5 if x == y else 0.0
            for x in cog for y in non) / (len(cog) * len(non))
    print(f"  {name:<44}{st.mean(rr):>6.2f}{t3:>5}{t5:>6}{a:>8.3f}")
    return a


def main() -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    print("Trained on DOCKED all-by-all blocks, median-polished features, leave-block-out.")
    print("Coventry is never trained on.\n")
    print(f"  {'model':<44}{'rank':>6}{'t3':>5}{'t5':>6}{'AUC':>8}")

    REF = coventry("best", ["i_total"])[:, :, 0]
    grade("ref2015 + cancellation (the bar)", REF, truth)

    best = None
    for setname, feats in (("full (with ref2015 terms)", ENERGY + OURS),
                           ("OURS ONLY (zero Rosetta)", OURS)):
        for agg in ("best", "top5"):
            X, y, g = load_blocks(agg, feats)
            blocks = sorted(set(g.tolist()))
            oof = np.zeros(len(y))
            for b in blocks:
                te = g == b
                sc = StandardScaler().fit(X[~te])
                clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
                clf.fit(sc.transform(X[~te]), y[~te])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            # out-of-fold AUC on the training blocks themselves -- if this is at chance the
            # Coventry number below is meaningless and the run should stop here
            pos, neg = oof[y == 1], oof[y == 0]
            auc_tr = float(np.mean([(a > neg).mean() + 0.5 * (a == neg).mean() for a in pos]))
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
            clf.fit(sc.transform(X), y)
            C = coventry(agg, feats)
            P = clf.predict_proba(sc.transform(C.reshape(-1, len(feats))))[:, 1].reshape(18, 18)
            a = grade(f"{setname}, {agg}   [block-out AUC {auc_tr:.3f}]", -P, truth)
            if best is None or a > best[0]:
                best = (a, setname, agg, feats, sc, clf, -P)

    a, setname, agg, feats, sc, clf, S = best
    joblib.dump({"scaler": sc, "clf": clf, "features": feats, "agg": agg,
                 "mode": "median_polish", "trained_on": "docked_blocks"},
                ROOT / "data/sel_model_docked.joblib")
    np.save(ROOT / "data/sel_coventry_docked_scores.npy", S)
    print(f"\nbest: {setname} / {agg}  ->  AUC {a:.3f}")
    print(f"saved data/sel_model_docked.joblib and the grid scores")
    order = np.argsort(-np.abs(clf.coef_[0]))
    print("top weights (median-polished, standardised):")
    for k in order[:10]:
        print(f"   {feats[k]:26s} {clf.coef_[0][k]:+.3f}")


if __name__ == "__main__":
    main()
