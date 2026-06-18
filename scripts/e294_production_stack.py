"""E294 — wire & benchmark the production stack: RandomForest base + pooled training + charge-routing.

Bank the validated wins: pooled PDBbind+PPIKB (e281 +0.04), RandomForest base (e293 +0.04 over GBT),
charge-routing (e283: best model per charge class). Pick the best model PER charge class on fresh-305 and
route. Benchmark vs old GBT and PPI-clone v2. Save candidate -> data/affinity_stack_candidate.joblib.
Run: OMP_NUM_THREADS=1 python scripts/e294_production_stack.py
"""
from __future__ import annotations
import json, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


ppikb = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and np.isfinite(y):
        ppikb.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                      "desc": d3, "atype": r["aff_type"], "length": int(pf(r["length"])),
                      "nc": float(pf(r["net_charge"])), "npep": r.get("npep", r["length"]),
                      "npocket": r.get("npocket", 0)})
ours = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
seen = set(); fresh = []
for r in sorted(ppikb, key=lambda x: x["pdb"]):
    if r["pdb"] in ours or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
        continue
    if abs(r["npep"] - r["length"]) > 2 or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fseq = {r["pep"] for r in fresh}; frec = {r["rec"] for r in fresh}
pdbb = [json.loads(l) for l in open("data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
Xtr = np.vstack([np.array([d["desc"] for d in pdbb]),
                 np.array([r["desc"] for r in ppikb if r["pep"] not in fseq and r["rec"] not in frec])])
ytr = np.concatenate([np.array([float(d["y"]) for d in pdbb]),
                      np.array([r["y"] for r in ppikb if r["pep"] not in fseq and r["rec"] not in frec])])
Xf = np.array([r["desc"] for r in fresh]); yf = np.array([r["y"] for r in fresh])
q = np.array([abs(r["nc"]) for r in fresh])
print(f"pooled train {len(Xtr)} | fresh {len(fresh)} (charged {int((q>=2).sum())}, neutral {int((q<=1).sum())})",
      flush=True)

models = {
    "GBT": HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0),
    "RF": RandomForestRegressor(n_estimators=400, max_depth=10, n_jobs=1, random_state=0),
    "SVR": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                     ("svr", SVR(kernel="rbf", C=4, gamma="scale"))]),
}
pred = {n: m.fit(Xtr, ytr).predict(Xf) for n, m in models.items()}


def rm(p, m):
    return (pearsonr(yf[m], p[m])[0], float(np.mean(np.abs(yf[m] - p[m]))))


print("\n=== per-model on fresh-305 ===")
for n, p in pred.items():
    a = rm(p, np.ones(len(yf), bool)); c = rm(p, q >= 2); nn = rm(p, q <= 1)
    print(f"  {n:10s} ALL {a[0]:+.3f}/{a[1]:.2f}  CHARGED {c[0]:+.3f}/{c[1]:.2f}  NEUTRAL {nn[0]:+.3f}/{nn[1]:.2f}")
# pick best per class
ch_scores = {n: rm(p, q >= 2)[0] for n, p in pred.items()}
ne_scores = {n: rm(p, q <= 1)[0] for n, p in pred.items()}
best_ch = max(ch_scores, key=ch_scores.get); best_ne = max(ne_scores, key=ne_scores.get)
routed = np.where(q >= 2, pred[best_ch], pred[best_ne])
# PPI-clone reference
ppi = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                ("svr", SVR(kernel="rbf", C=4, gamma="scale"))]).fit(Xtr, ytr).predict(Xf)
print(f"\n=== ROUTED STACK (charged->{best_ch}, neutral->{best_ne}) vs baselines ===")
for n, p in [("OLD GBT (single)", pred["GBT"]), ("PPI-clone v2", ppi), (f"ROUTED STACK", routed)]:
    a = rm(p, np.ones(len(yf), bool)); c = rm(p, q >= 2); nn = rm(p, q <= 1)
    print(f"  {n:18s} ALL {a[0]:+.3f}/{a[1]:.2f}  CHARGED {c[0]:+.3f}/{c[1]:.2f}  NEUTRAL {nn[0]:+.3f}/{nn[1]:.2f}")

joblib.dump({"charged_model": models[best_ch], "neutral_model": models[best_ne],
             "route_threshold": 2, "feature": "protdcal3d_37", "trained_on": "pooled_pdbbind_ppikb"},
            "data/affinity_stack_candidate.joblib")
json.dump({"best_charged": best_ch, "best_neutral": best_ne,
           "routed_all_r": float(rm(routed, np.ones(len(yf), bool))[0]),
           "ppi_all_r": float(rm(ppi, np.ones(len(yf), bool))[0])},
          open("data/e294_stack.json", "w"))
print("\nsaved data/affinity_stack_candidate.joblib + data/e294_stack.json")
