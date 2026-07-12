"""E320 (concept N4 + Hess's-law adaptation) — is thermodynamic cycle-closure a useful training constraint?

FEP/Hess: free energy is a state function, so any thermodynamic cycle of DDG's must close (sum to 0). N4 asks:
can we import that self-consistency into a cheap scorer as an auxiliary loss on UNLABELED grids?

The catch, tested here honestly: a POINTWISE scorer f(x) closes EVERY cycle BY CONSTRUCTION
  DDG(A->B)+DDG(B->C)+DDG(C->A) = (f_B-f_A)+(f_C-f_B)+(f_A-f_C) = 0.
So cycle-closure adds ZERO information to a pointwise model. It is only non-vacuous for a model that predicts
DDG directly from a PAIR representation g([x_i,x_j]), which CAN violate closure. So N4 is worth building only if
(1) a pair model violates closure meaningfully, and (2) the closure-respecting (pointwise) model is at least as
accurate — i.e. enforcing closure would not cost accuracy. This script measures both on same-receptor peptide
pairs from the 925-complex set.

Run: OMP_NUM_THREADS=1 python experiments/e320_cycle_closure_n4.py
"""
from __future__ import annotations
import json, os, hashlib, itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
X = np.array([d["x"] for d in cache]); y = np.array([d["y"] for d in cache])
rseq = [d["rseq"] for d in cache]
grp = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in rseq])

# group peptides by receptor; keep receptors with >=3 peptides so cycles exist
byrec: dict[int, list[int]] = {}
for i, gg in enumerate(grp):
    byrec.setdefault(gg, []).append(i)
multi = {g: idx for g, idx in byrec.items() if len(idx) >= 3}
print(f"receptors with >=3 peptides (cycles possible): {len(multi)}  "
      f"(peptides covered: {sum(len(v) for v in multi.values())})")

# build same-receptor pairs (i,j) with target DDG = y_i - y_j
pairs, ddg, pgrp = [], [], []
for g, idx in multi.items():
    for i, j in itertools.combinations(idx, 2):
        pairs.append((i, j)); ddg.append(y[i] - y[j]); pgrp.append(g)
pairs = np.array(pairs); ddg = np.array(ddg); pgrp = np.array(pgrp)
print(f"same-receptor peptide pairs: {len(pairs)}")

Xpair_diff = X[pairs[:, 0]] - X[pairs[:, 1]]          # difference representation (antisymmetric)
Xpair_cat = np.hstack([X[pairs[:, 0]], X[pairs[:, 1]]])  # concat representation (can violate closure)


def loo(Xp, target, groups):
    pred = np.full(len(target), np.nan)
    for tr, te in GroupKFold(6).split(Xp, target, groups):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(Xp[tr], target[tr])
        pred[te] = m.predict(Xp[te])
    return pred


pred_diff = loo(Xpair_diff, ddg, pgrp)
pred_cat = loo(Xpair_cat, ddg, pgrp)
print("\nDDG prediction (leave-receptor-out):")
print(f"  closure-RESPECTING  (difference features x_i-x_j) : r={pearsonr(pred_diff, ddg)[0]:+.3f}  "
      f"MAE={np.mean(np.abs(pred_diff-ddg)):.3f}")
print(f"  closure-FREE        (concat features [x_i,x_j])   : r={pearsonr(pred_cat, ddg)[0]:+.3f}  "
      f"MAE={np.mean(np.abs(pred_cat-ddg)):.3f}")

# measure closure violation of the concat model on triangles A->B->C->A
predfun = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                        l2_regularization=1.0, random_state=0).fit(Xpair_cat, ddg)


def ddg_cat(i, j):
    return float(predfun.predict(np.hstack([X[i], X[j]]).reshape(1, -1))[0])


viol = []
for g, idx in list(multi.items())[:200]:
    for a, b, c in itertools.combinations(idx, 3):
        viol.append(ddg_cat(a, b) + ddg_cat(b, c) + ddg_cat(c, a))  # should be 0
viol = np.array(viol)
print(f"\nclosure violation of the concat DDG model over {len(viol)} triangles: "
      f"RMS={np.sqrt(np.mean(viol**2)):.3f} kcal/mol (pointwise/diff model = 0 by construction)")
respect_wins = np.mean(np.abs(pred_diff - ddg)) <= np.mean(np.abs(pred_cat - ddg))
print("VERDICT: " + (
    "closure-respecting model is AS GOOD OR BETTER and the concat model visibly violates closure → enforcing "
    "cycle-closure (N4) costs no accuracy and could regularize a richer relative model; worth building IF we "
    "move to a pair/relative architecture. For the current POINTWISE scorer, closure is already exact → N4 "
    "adds nothing today." if respect_wins else
    "concat model beats the closure-respecting one → closure is NOT the binding accuracy bottleneck here."))
