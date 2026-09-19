#!/usr/bin/env python
"""Can we refit what ref2015 already does — and how much data would it take?

THE FINDING THIS INTERROGATES. On the Coventry grid, after cancellation, ref2015's total interface
energy reaches AUC 0.735. Every one of the nine WEIGHTED terms that sums to it reaches between
0.422 and 0.551 -- chance, or worse than chance. The terms are already multiplied by their ref2015
weights when we extract them, so this is not a story about weights being clever: the discriminating
information is not in any term, it is in what the terms CANCEL when added. fa_atr and fa_rep both
grow with how much peptide is buried, which is not specific to a pair; in the sum that shared
growth subtracts out and what survives is the part that depends on which residues face which.

That is the same principle as the median polish, one level down. Cancellation across TERMS inside
a single energy, cancellation across the GRID between cells. Both are subtraction, neither is
fitted, and on this grid both are worth more than any model we have trained.

So the question for our own scorer is not "which feature is missing". It is whether we can learn a
cancelling combination from the data we have. Three tests:

  REFIT       fit a linear combination of the same nine weighted terms on our docked blocks and
              test it on Coventry. It sees exactly the information ref2015's sum sees. If it
              loses to plain addition, the deficit is our training set, not the feature set --
              ref2015's weights were fitted on orders of magnitude more structure than 72
              cognate cells, and a delicate cancellation is exactly what small data cannot find.
  CURVE       retrain on 3, 5, 7, 9 blocks and watch Coventry AUC. A rising curve says more
              blocks reach the bar and says roughly how many. A flat curve says they do not.
  STACK       does our orthogonal feature set add anything ON TOP of ref2015's sum, rather than
              instead of it? Our 18 features reconstruct the polished total at R2 = 0.07, so
              they are nearly independent of it -- independence is worth testing, not assuming.

Usage: sel_refit_test.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]

TERMS = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
         "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb"]
OURS = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
        "c_aromatic_frac", "c_mixed_frac", "hydro_pocket", "hydro_product", "hydro_absdiff",
        "charge_product", "pocket_charge_per_res", "n_contact_per_res",
        "rec_atoms_touched_per_res", "burial_mean", "burial_sd", "burial_cv", "burial_max",
        "frac_res_contacting"]


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def auc(pos, neg) -> float:
    return float(np.mean([(p < neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def blocks(feats):
    """Per-block polished feature cubes from the docked all-by-all set."""
    grid = defaultdict(dict)
    for line in (ROOT / "logs/sel_dock_features.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                grid[r["block"]][(r["pep_idx"], r["rec_idx"])] = r["best"]
    out = {}
    for b, cells in grid.items():
        n = max(max(k) for k in cells) + 1
        if len(cells) != n * n or n < 5:
            continue
        A = np.nan_to_num(np.array([[[float(cells[(i, j)].get(f, 0.0)) for f in feats]
                                     for j in range(n)] for i in range(n)]), nan=0.0,
                          posinf=0.0, neginf=0.0)
        C = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
        X = C.reshape(-1, len(feats))
        y = np.eye(n, dtype=int).reshape(-1)
        out[b] = (X, y)
    return out


def coventry(feats):
    rows = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = r["best"]
    A = np.nan_to_num(np.array([[[float(rows[(p, b)].get(f, 0.0)) for f in feats]
                                 for b in ORDER] for p in ORDER]), nan=0.0,
                      posinf=0.0, neginf=0.0)
    C = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    keep = np.zeros((18, 18), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            keep[i, j] = (i == j) or truth.get((p, b), {}).get("measured") == "0"
    return C, keep


def grade(S, keep) -> tuple[float, float, int]:
    """(AUC, mean cognate rank, top-3) for a score where lower means better."""
    cog = np.array([S[i, i] for i in range(18)])
    non = np.array([S[i, j] for i in range(18) for j in range(18)
                    if i != j and keep[i, j]])
    rr = [sorted(range(18), key=lambda j: S[i, j]).index(i) + 1 for i in range(18)]
    return auc(cog, non), float(np.mean(rr)), sum(r <= 3 for r in rr)


def fit_score(Xtr, ytr, Xte, shape=(18, 18)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return -clf.predict_proba(sc.transform(Xte))[:, 1].reshape(shape)


def main() -> None:
    rng = np.random.default_rng(0)
    Ct, keep = coventry(TERMS + ["i_total"] + OURS)
    it = TERMS + ["i_total"] + OURS
    BAR = Ct[:, :, it.index("i_total")]
    a, r, t3 = grade(BAR, keep)
    print("REFIT TEST.  All models see the same median-polished cells; Coventry is never")
    print("trained on. 'plain sum' is ref2015's own addition with no fitting at all.\n")
    print(f"  {'model':<52}{'AUC':>7}{'rank':>7}{'t3':>5}")
    print(f"  {'ref2015 plain sum of the same nine terms':<52}{a:>7.3f}{r:>7.2f}{t3:>5}")

    B = blocks(TERMS + ["i_total"] + OURS)
    names = sorted(B)
    Xall = np.vstack([B[b][0] for b in names])
    yall = np.concatenate([B[b][1] for b in names])

    for label, feats in (("refit the nine terms (linear)", TERMS),
                         ("refit the nine terms + i_total", TERMS + ["i_total"]),
                         ("our 18 features only", OURS),
                         ("everything", TERMS + ["i_total"] + OURS)):
        idx = [it.index(f) for f in feats]
        S = fit_score(Xall[:, idx], yall, Ct[:, :, idx].reshape(-1, len(idx)))
        a, r, t3 = grade(S, keep)
        print(f"  {label:<52}{a:>7.3f}{r:>7.2f}{t3:>5}")

    print("\nLEARNING CURVE.  Coventry AUC against how many docked blocks the model was trained")
    print("on; 6 random draws per size, so a rise is not one lucky subset.\n")
    print(f"  {'blocks':>7}{'cells':>8}{'cognate':>9}{'terms refit':>14}{'ours only':>12}"
          f"{'everything':>13}")
    for k in (3, 5, 7, 9):
        got = {"terms": [], "ours": [], "all": []}
        ncell = ncog = 0
        for _ in range(6 if k < len(names) else 1):
            pick = list(rng.choice(names, size=k, replace=False))
            X = np.vstack([B[b][0] for b in pick])
            y = np.concatenate([B[b][1] for b in pick])
            ncell, ncog = len(y), int(y.sum())
            for key, feats in (("terms", TERMS), ("ours", OURS),
                               ("all", TERMS + ["i_total"] + OURS)):
                idx = [it.index(f) for f in feats]
                S = fit_score(X[:, idx], y, Ct[:, :, idx].reshape(-1, len(idx)))
                got[key].append(grade(S, keep)[0])
        print(f"  {k:>7}{ncell:>8}{ncog:>9}{np.mean(got['terms']):>14.3f}"
              f"{np.mean(got['ours']):>12.3f}{np.mean(got['all']):>13.3f}")

    print("\nSTACK TEST.  Does our orthogonal feature set add anything on TOP of the bar?")
    print("Both z-scored; w is the weight on our learned score added to ref2015's sum.\n")
    idx = [it.index(f) for f in OURS]
    OUR_S = fit_score(Xall[:, idx], yall, Ct[:, :, idx].reshape(-1, len(idx)))
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    print(f"  {'w':>6}{'AUC':>8}{'rank':>7}{'t3':>5}")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        a, r, t3 = grade(_z(BAR) + w * _z(OUR_S), keep)
        print(f"  {w:>6.2f}{a:>8.3f}{r:>7.2f}{t3:>5}")
    print(f"\n  corr(bar, our learned score) over the 18x18 grid: "
          f"{np.corrcoef(BAR.ravel(), OUR_S.ravel())[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
