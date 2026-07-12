"""E281 — does training on POOLED PDBbind+PPIKB (like PPI-Affinity) crack the CHARGED cases?

Both PDBbind (e180, 869) and PPIKB carry 37-dim ProtDCal-3D descriptors — PPI-Affinity's exact feature
class. Train GBT (ours) on PDBbind-only / PPIKB-only / POOLED, plus the PPI-clone-v2 SVR on pooled, and
evaluate on the strict ~304 fresh PPIKB test set (held out by seq+pdb; its RECEPTORS excluded from the
PPIKB training pool = honest leave-receptor-out). Break down by charge: neutral |q|<=1 vs charged |q|>=2.
Answers Ram: does more data + pooling help the charged cases we are bad at, or is charged FEP-bound floor?
Run: OMP_NUM_THREADS=1 python experiments/e281_pooled_charged.py
"""
from __future__ import annotations
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


# PPIKB rows with desc3d
ppikb = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); y = pf(r["y"])
    except Exception:
        continue
    if not (isinstance(d3, list) and len(d3) == 37 and np.isfinite(y)):
        continue
    ppikb.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                  "desc": d3, "atype": r["aff_type"], "length": int(pf(r["length"])),
                  "nc": float(pf(r["net_charge"])), "npep": r.get("npep", r["length"]),
                  "npocket": r.get("npocket", 0)})
ours_pdbs = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}

# strict ~304 fresh test (e215 filter)
seen = set(); fresh = []
for r in sorted(ppikb, key=lambda x: x["pdb"]):
    if r["pdb"] in ours_pdbs or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
        continue
    if abs(r["npep"] - r["length"]) > 2 or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fresh_seqs = {r["pep"] for r in fresh}; fresh_recs = {r["rec"] for r in fresh}
print(f"strict fresh test n={len(fresh)} | charged|q|>=2: {sum(abs(r['nc'])>=2 for r in fresh)} | "
      f"neutral|q|<=1: {sum(abs(r['nc'])<=1 for r in fresh)}", flush=True)

# training pools (ProtDCal-3D 37-dim). honest: drop PPIKB rows on a fresh-test receptor.
pdbb = [json.loads(l) for l in open("data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
X_pdbb = np.array([d["desc"] for d in pdbb]); y_pdbb = np.array([float(d["y"]) for d in pdbb])
ppikb_train = [r for r in ppikb if r["pep"] not in fresh_seqs and r["rec"] not in fresh_recs]
X_ppi = np.array([r["desc"] for r in ppikb_train]); y_ppi = np.array([r["y"] for r in ppikb_train])
Xf = np.array([r["desc"] for r in fresh]); yf = np.array([r["y"] for r in fresh])
q = np.array([abs(r["nc"]) for r in fresh])
print(f"pools: PDBbind {len(X_pdbb)} | PPIKB(honest) {len(X_ppi)} | POOLED {len(X_pdbb)+len(X_ppi)}",
      flush=True)


def gbt():
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)


def clone():
    return Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                     ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))])


def ev(model, name):
    p = model.predict(Xf)
    def rm(m):
        if m.sum() < 4:
            return (float("nan"), float("nan"))
        return (pearsonr(yf[m], p[m])[0], float(np.mean(np.abs(yf[m] - p[m]))))
    ro, mo = rm(np.ones(len(yf), bool)); rc, mc = rm(q >= 2); rn, mn = rm(q <= 1)
    print(f"  {name:28s} ALL r={ro:+.3f}/MAE{mo:.2f}  CHARGED r={rc:+.3f}/MAE{mc:.2f}  "
          f"NEUTRAL r={rn:+.3f}/MAE{mn:.2f}")
    return dict(all=(ro, mo), charged=(rc, mc), neutral=(rn, mn))


Xpool = np.vstack([X_pdbb, X_ppi]); ypool = np.concatenate([y_pdbb, y_ppi])
print("\n=== fresh ~304 PPIKB: training-set ablation (37-dim ProtDCal-3D) ===")
res = {}
res["ours_PDBbind"] = ev(gbt().fit(X_pdbb, y_pdbb), "OURS / PDBbind-only")
res["ours_PPIKB"] = ev(gbt().fit(X_ppi, y_ppi), "OURS / PPIKB-only")
res["ours_POOLED"] = ev(gbt().fit(Xpool, ypool), "OURS / POOLED (like PPI)")
res["clone_POOLED"] = ev(clone().fit(Xpool, ypool), "PPI-clone v2 / POOLED")
json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in res.items()},
          open("data/e281_pooled_charged.json", "w"), indent=1)
print("\nVERDICT: if POOLED charged r >> PDBbind-only charged r, pooling cracks charged (Ram right).")
print("If charged stays ~0 across all pools, the charged floor is FEP-bound (data can't fix it).")
print("saved data/e281_pooled_charged.json")
