"""E297 — IFP-space ALCHEMY: learn ΔG-difference from INTERACTION-MAP difference (Ram's bond create/destroy/
strengthen idea, operationalized). Uses the e296 cache.

The alchemy: instead of mutating residues, represent the change between two complexes as the DIFFERENCE in
their interaction fingerprints (which bonds were created/destroyed/strengthened), and learn ΔG-difference
from it. Test whether IFP-relative scoring (a) works WITHIN-receptor (the easy leg) and (b) transfers
CROSS-receptor better than aggregate-feature relative scoring (the wall) — because IFP is better physics so
the offset may be smaller in IFP space.

Arms (PDBbind, pairs of complexes):
  AGG_rel_within : ΔG-diff from aggregate-feature difference, same receptor (baseline relative)
  IFP_rel_within : ΔG-diff from IFP difference, same receptor (the alchemy, easy leg)
  AGG_rel_cross  : aggregate-feature relative, DIFFERENT receptor (the wall)
  IFP_rel_cross  : IFP relative, DIFFERENT receptor (does better physics shrink the wall?)
Run: OMP_NUM_THREADS=1 python scripts/e297_ifp_alchemy.py
"""
from __future__ import annotations
import json, os, numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
data = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
y = np.array([d["y"] for d in data])
rseq = [d["rec_seq"] if "rec_seq" in d else d["rseq"] for d in data]
import hashlib
recid = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in rseq])
by_rec = defaultdict(list)
for i in range(len(data)):
    by_rec[recid[i]].append(i)

# ---- build pair datasets: features = difference, target = ΔG difference ----
def make_pairs(cross: bool, n_max=8000):
    dXa, dXi, dY, grp = [], [], [], []
    recs = list(by_rec.keys())
    if cross:
        # cross-receptor pairs: sample i from one receptor, j from another
        idx = np.arange(len(data))
        for _ in range(n_max):
            i, j = rng.choice(idx, 2, replace=False)
            if recid[i] == recid[j]:
                continue
            dXa.append(X[i] - X[j]); dXi.append(IFP[i] - IFP[j]); dY.append(y[i] - y[j])
            grp.append(recid[i])
    else:
        for rr, cells in by_rec.items():
            if len(cells) < 2:
                continue
            for i, j in combinations(cells, 2):
                dXa.append(X[i] - X[j]); dXi.append(IFP[i] - IFP[j]); dY.append(y[i] - y[j])
                grp.append(rr)
                dXa.append(X[j] - X[i]); dXi.append(IFP[j] - IFP[i]); dY.append(y[j] - y[i])
                grp.append(rr)
    return np.array(dXa), np.array(dXi), np.array(dY), np.array(grp)


def cv_pairs(M, t, grp):
    if len(t) < 30:
        return float("nan"), 0
    p = np.full(len(t), np.nan)
    for tr, te in GroupKFold(min(6, len(set(grp.tolist())))).split(M, t, grp):
        p[te] = HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(M[tr], t[tr]).predict(M[te])
    return pearsonr(t, p)[0], len(t)


print("=== IFP-space ALCHEMY: predict ΔG-difference from feature-difference ===")
# within-receptor (the easy leg)
aw, iw, yw, gw = make_pairs(cross=False)
ra, na = cv_pairs(aw, yw, gw); ri, ni = cv_pairs(iw, yw, gw)
print(f"\nWITHIN-receptor pairs (n={na}):")
print(f"  AGG-feature-diff -> ΔG : r={ra:+.3f}")
print(f"  IFP-diff -> ΔG (alchemy): r={ri:+.3f}")
# cross-receptor (the wall)
ac, ic, yc, gc = make_pairs(cross=True)
rac, nac = cv_pairs(ac, yc, gc); ric, nic = cv_pairs(ic, yc, gc)
print(f"\nCROSS-receptor pairs (n={nac}):")
print(f"  AGG-feature-diff -> ΔG : r={rac:+.3f}")
print(f"  IFP-diff -> ΔG (alchemy): r={ric:+.3f}")
print("\nVERDICT: IFP-within >> AGG-within => alchemy is better relative physics.")
print("IFP-cross > AGG-cross (and both >0) => IFP shrinks the cross-receptor wall (the prize).")
print("IFP-cross ~ AGG-cross ~0 => wall persists even with interaction map (offset still FEP-bound).")
json.dump(dict(agg_within=float(ra), ifp_within=float(ri), agg_cross=float(rac), ifp_cross=float(ric)),
          open(os.path.join(ROOT, "data/e297_alchemy.json"), "w"))
print("saved data/e297_alchemy.json")
