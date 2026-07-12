"""E293 — throw an ML model ZOO at the offsets + deep-dive the eta (non-additive interaction) term.

PART A: model zoo on ABSOLUTE y (leave-receptor-out) — is GBT the bottleneck or the data? Does ANY class win?
PART B: model zoo on the RESIDUAL e=S-y (stacking) — can any model EXTRACT signal our GBT left? (Ram's 'learn
        the offset'). Positive OOF r => free boost.
PART C: model zoo on RECEPTOR offset b(R) (leave-receptor-out) — exhaustive confirm it's unlearnable.
PART D: eta = e − b − c (non-additive interaction, the LARGEST chunk std 1.51). Correlate with explicit
        PAIR-interaction features (charge×charge, hydrophobic×hydrophobic, size×size, ...) + best ML on it.
Run: OMP_NUM_THREADS=1 python experiments/e293_ml_zoo_eta.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

KDH = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
       "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
       "N": -3.5, "K": -3.9, "R": -4.5}
POS = set("KR"); NEG = set("DE"); AROM = set("FWY")
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD") or not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and len(pk) == 22 and np.isfinite(y):
        s = r["seq"]; n = max(len(s), 1)
        recs.append({"rec": r["protein_seq"], "pep": s, "y": float(y), "d3": d3, "pk": pk,
                     "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])],
                     "pq": sum(c in POS for c in s) - sum(c in NEG for c in s),
                     "phyd": np.mean([KDH.get(c, 0) for c in s]) if s else 0,
                     "plen": len(s), "parom": sum(c in AROM for c in s) / n})
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
recname = [r["rec"] for r in recs]
grp = np.array([hash(s) % (10**9) for s in recname])
gkf = GroupKFold(8)


def zoo():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05)),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=1, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=10, n_jobs=1, random_state=0),
        "HistGBT": HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                 l2_regularization=1.0, random_state=0),
        "GradBoost": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
        "SVR_rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=4, gamma="scale")),
        "KernelRidge": make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                                            alpha=0.01, random_state=0)),
    }


def oof(model, Xm, target):
    p = np.full(len(target), np.nan)
    for tr, te in gkf.split(Xm, target, grp):
        m = model
        try:
            m.fit(Xm[tr], target[tr]); p[te] = m.predict(Xm[te])
        except Exception:
            return None
    return p


def run(title, Xm, target):
    print(f"\n=== {title} (leave-receptor-out OOF r) ===", flush=True)
    best = (None, -1)
    for name, m in zoo().items():
        p = oof(m, Xm, target)
        if p is None or np.std(p) < 1e-9:
            print(f"  {name:14s}   (failed)"); continue
        r = pearsonr(target, p)[0]
        print(f"  {name:14s} r={r:+.3f}")
        if r > best[1]:
            best = (name, r)
    print(f"  -> BEST: {best[0]} r={best[1]:+.3f}")
    return best


# PART A: absolute y
run("PART A: ABSOLUTE y", X, y)

# PART B: residual stacking (S from GBT, learn e=S-y)
S = oof(zoo()["HistGBT"], X, y)
e = S - y
run("PART B: RESIDUAL e=S-y (stacking — extract what GBT missed?)", X, e)

# PART C: receptor offset b(R) — ridge two-way decomp, then learn b from pocket features
pidx = {s: i for i, s in enumerate(sorted({r["pep"] for r in recs}))}
ridx = {s: i for i, s in enumerate(sorted(set(recname)))}
pi = np.array([pidx[r["pep"]] for r in recs]); ri = np.array([ridx[s] for s in recname])
rec_cells = defaultdict(list); pep_cells = defaultdict(list)
for i in range(len(e)):
    rec_cells[ri[i]].append(i); pep_cells[pi[i]].append(i)
b = np.zeros(len(ridx)); c = np.zeros(len(pidx)); lam = 3.0
for _ in range(60):
    for rr, cc in rec_cells.items():
        b[rr] = np.sum([e[i] - c[pi[i]] for i in cc]) / (len(cc) + lam)
    for pp, cc in pep_cells.items():
        c[pp] = np.sum([e[i] - b[ri[i]] for i in cc]) / (len(cc) + lam)
rec_ids = [s for s, cc in rec_cells.items() if len(cc) >= 2]
Xrec = np.array([np.mean([recs[i]["pk"] for i in rec_cells[s]], axis=0) for s in rec_ids])
brec = np.array([b[s] for s in rec_ids])
grp_r = np.arange(len(rec_ids))  # each receptor its own group (LOO-ish via KFold)
from sklearn.model_selection import KFold
print("\n=== PART C: LEARN b(R) from pocket features (zoo, 5-fold) ===")
bestc = (None, -1)
for name, m in zoo().items():
    p = np.full(len(brec), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xrec):
        try:
            m.fit(Xrec[tr], brec[tr]); p[te] = m.predict(Xrec[te])
        except Exception:
            p = None; break
    if p is None or np.std(p) < 1e-9:
        continue
    r = pearsonr(brec, p)[0]
    print(f"  {name:14s} r={r:+.3f}")
    if r > bestc[1]:
        bestc = (name, r)
print(f"  -> BEST b(R) learner: {bestc[0]} r={bestc[1]:+.3f} (vs predict-mean=0)")

# PART D: eta interaction
eta = np.array([e[i] - b[ri[i]] - c[pi[i]] for i in range(len(e))])
print(f"\n=== PART D: eta (non-additive) std={eta.std():.2f} — correlate with PAIR-interaction features ===")
# explicit pair features
pq = np.array([r["pq"] for r in recs]); phyd = np.array([r["phyd"] for r in recs])
plen = np.array([r["plen"] for r in recs]); parom = np.array([r["parom"] for r in recs])
# pocket summaries (first few pkf as proxies for pocket charge/size/hydrophobicity)
pk0 = np.array([r["pk"][0] for r in recs]); pk1 = np.array([r["pk"][1] for r in recs])
pk2 = np.array([r["pk"][2] for r in recs]); pk16 = np.array([r["pk"][16] for r in recs])
inter = {
    "pepQ x pkf0": pq * pk0, "pepQ x pkf1": pq * pk1, "pepQ x pkf2": pq * pk2,
    "|pepQ| x pkf16": np.abs(pq) * pk16, "pepHyd x pkf0": phyd * pk0, "pepHyd x pkf1": phyd * pk1,
    "pepLen x pkf0": plen * pk0, "pepArom x pkf16": parom * pk16, "pepQ^2": pq * pq,
    "|pepQ|": np.abs(pq), "pepHyd": phyd, "pepLen": plen,
}
rows = []
for k, v in inter.items():
    if np.std(v) < 1e-9:
        continue
    r, p = pearsonr(eta, v)
    rows.append((k, r, p))
rows.sort(key=lambda t: -abs(t[1]))
for k, r, p in rows:
    print(f"  eta vs {k:18s} r={r:+.3f} p={p:.1e} {'*' if p<0.05 else ''}")
# best ML on eta from full pair features
be = run("PART D2: LEARN eta from full pair features (zoo)", X, eta)
json.dump(dict(eta_std=float(eta.std()), best_bR=bestc[1]), open("data/e293_zoo.json", "w"))
print("\nsaved data/e293_zoo.json")
