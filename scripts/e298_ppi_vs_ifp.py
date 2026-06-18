"""E298 — head-to-head: PPI-clone v2 vs ours-base vs ours+IFP, r AND MAE, on PDBbind (crystal poses).

PDBbind 925 has crystal structures -> IFP. Match with e180 ProtDCal-3D (for PPI-clone). Leave-receptor-out
(group by receptor sequence). Report r + MAE, ALL / charged / neutral. This is the honest 'us-with-IFP vs
PPI-clone' number on a structure-bearing test set.
Run: OMP_NUM_THREADS=1 python scripts/e298_ppi_vs_ifp.py
"""
from __future__ import annotations
import json, os, hashlib, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
# ProtDCal-3D for PPI-clone, by pdb
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
print(f"matched (IFP + ProtDCal): {len(data)}", flush=True)
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
DESC = np.array([protd[d["pdb"]] for d in data])
y = np.array([d["y"] for d in data]); q = np.array([d["q"] for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
ch = q >= 2


def cv(M, model_fn):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        p[te] = model_fn().fit(M[tr], y[tr]).predict(M[te])
    return p


def gbt():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)


def svr():
    return Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                     ("svr", SVR(kernel="rbf", C=4, gamma="scale"))])


preds = {
    "PPI-clone v2": cv(DESC, svr),
    "OURS base (17 feat)": cv(X, gbt),
    "OURS + IFP": cv(np.hstack([X, IFP]), gbt),
}


def rm(p, m):
    return (pearsonr(y[m], p[m])[0], float(np.mean(np.abs(y[m] - p[m]))))


print("\n=== PDBbind crystal, leave-receptor-out: r / MAE ===")
print(f"{'model':22s} {'ALL r/MAE':>16s} {'CHARGED r/MAE':>16s} {'NEUTRAL r/MAE':>16s}")
res = {}
for n, p in preds.items():
    a = rm(p, np.ones(len(y), bool)); c = rm(p, ch); nn = rm(p, q <= 1)
    res[n] = {"all": a, "charged": c, "neutral": nn}
    print(f"{n:22s} {a[0]:+.3f}/{a[1]:.2f}      {c[0]:+.3f}/{c[1]:.2f}      {nn[0]:+.3f}/{nn[1]:.2f}")
json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in res.items()},
          open(os.path.join(ROOT, "data/e298_ppi_vs_ifp.json"), "w"), indent=1)
print("\nsaved data/e298_ppi_vs_ifp.json")
