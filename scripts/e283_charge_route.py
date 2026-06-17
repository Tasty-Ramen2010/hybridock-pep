"""E283 — charge-routed model to BEAT PPI on neutral (and keep our charged edge).

e282: GBT wins charged (0.342), SVR wins neutral (0.261) — complementary. Route by |net charge|:
neutral |q|<=1 -> SVR (PPI's model class), charged |q|>=2 -> GBT (ours). Train both on POOLED
PDBbind+PPIKB (37-dim ProtDCal-3D), eval on fresh-305 vs PPI-clone v2 and our single models.
Run: OMP_NUM_THREADS=1 python scripts/e283_charge_route.py
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
ours_pdbs = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
seen = set(); fresh = []
for r in sorted(ppikb, key=lambda x: x["pdb"]):
    if r["pdb"] in ours_pdbs or r["atype"] not in ("Kd", "KD", "pKd"):
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
print(f"fresh {len(fresh)} | charged {int((q>=2).sum())} | neutral {int((q<=1).sum())}", flush=True)

gbt = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                    l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
svr = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xtr, ytr)
pg = gbt.predict(Xf); ps = svr.predict(Xf)
# routed: neutral->SVR, charged->GBT
routed = np.where(q >= 2, pg, ps)


def rep(name, p):
    def rm(m):
        return (pearsonr(yf[m], p[m])[0] if m.sum() > 3 else float("nan"),
                float(np.mean(np.abs(yf[m] - p[m]))))
    ra = rm(np.ones(len(yf), bool)); rc = rm(q >= 2); rn = rm(q <= 1)
    print(f"  {name:18s} ALL {ra[0]:+.3f}/{ra[1]:.2f}  CHARGED {rc[0]:+.3f}/{rc[1]:.2f}  "
          f"NEUTRAL {rn[0]:+.3f}/{rn[1]:.2f}")
    return dict(all=ra, charged=rc, neutral=rn)


print("\n=== fresh-305 head-to-head ===")
res = {"GBT (ours)": rep("GBT (ours)", pg), "SVR (PPI-clone)": rep("SVR (PPI-clone)", ps),
       "CHARGE-ROUTED": rep("CHARGE-ROUTED", routed)}
json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in res.items()},
          open("data/e283_route.json", "w"), indent=1)
print("\nVERDICT: routed should match SVR on neutral AND GBT on charged = beat both single models.")
print("saved data/e283_route.json")
