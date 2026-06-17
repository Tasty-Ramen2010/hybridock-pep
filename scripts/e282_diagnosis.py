"""E282 — diagnosis for Ram: (A) the DOUBLE-DIFFERENCE math, (B) where we fail on NEUTRAL.

(A) Ram's 4-corner idea: predict ΔG(P,R) from the other 3 corners of a 2x2 grid:
      double-diff:  ΔG(P,R) ≈ y(P,Rk) + y(Pk,R) − y(Pk,Rk)        [pure experiment, no scorer]
    Algebra: with G(p,r)=f(p)+g(r)+coupling, the double-diff cancels BOTH g(R) [=b(R)] AND f(P) [=c(P)],
    leaving ONLY the non-additive coupling. So Ram is RIGHT it extracts more — IF the coupling is small
    AND you have the extra corner y(P,Rk) (query peptide on another receptor). We TEST both:
      * coupling magnitude on real 2x2 grids = |y(P,R) − y(P,Rk) − y(Pk,R) + y(Pk,Rk)|
      * leave-one-corner-out: how well does double-diff predict the held-out corner vs single-anchor.

(B) NEUTRAL: e281 showed ours(GBT) neutral r=0.214 < PPI-clone(SVR) 0.275 — we LOSE on neutral. Test
    model class (GBT vs SVR vs Ridge vs blend) on the fresh-305 neutral subset to find what beats it.
Run: OMP_NUM_THREADS=1 python scripts/e282_diagnosis.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")
        if json.loads(l).get("aff_type") in ("Kd", "Ki", "KD")]
# index y by (peptide, receptor); average duplicates
ymap = defaultdict(list)
for r in rows:
    try:
        ymap[(r["seq"], r["protein_seq"])].append(pf(r["y"]))
    except Exception:
        pass
ymap = {k: float(np.mean(v)) for k, v in ymap.items()}
pep_recs = defaultdict(set)
for (p, rr) in ymap:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]

# ---------- (A) double-difference on 2x2 grids ----------
couplings, dd_err, anc_err, dd_t, dd_p = [], [], [], [], []
grids = 0
for a, b in combinations(multi, 2):
    common = list(pep_recs[a] & pep_recs[b])
    if len(common) < 2:
        continue
    for R, Rk in combinations(common, 2):
        # corners
        try:
            yPR = ymap[(a, R)]; yPRk = ymap[(a, Rk)]; yPkR = ymap[(b, R)]; yPkRk = ymap[(b, Rk)]
        except KeyError:
            continue
        grids += 1
        coupling = yPR - yPRk - yPkR + yPkRk
        couplings.append(coupling)
        # double-diff predicts yPR from the other 3
        dd_pred = yPRk + yPkR - yPkRk
        dd_err.append(abs(dd_pred - yPR)); dd_t.append(yPR); dd_p.append(dd_pred)
        # single-anchor on R: predict yPR ≈ yPkR + [S(P,R)-S(Pk,R)]; without scorer, baseline = yPkR
        anc_err.append(abs(yPkR - yPR))
couplings = np.array(couplings)
print(f"=== (A) DOUBLE-DIFFERENCE on {grids} real 2x2 grids ===")
print(f"  coupling (non-additivity) |kcal/mol|: mean={np.mean(np.abs(couplings)):.2f} "
      f"median={np.median(np.abs(couplings)):.2f} max={np.max(np.abs(couplings)):.2f}")
print(f"  double-diff predict 4th corner : MAE={np.mean(dd_err):.2f} "
      f"r={pearsonr(dd_t, dd_p)[0]:+.3f}")
print(f"  single-anchor (same-receptor Pk on R, no scorer): MAE={np.mean(anc_err):.2f}")
print("  -> double-diff residual = the coupling; single-anchor residual = c(P)-c(Pk)+coupling.")

# ---------- (B) neutral model class on fresh-305 ----------
ppikb = []
for r in rows:
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and np.isfinite(pf(r["y"])):
        ppikb.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"],
                      "y": float(pf(r["y"])), "desc": d3, "atype": r["aff_type"],
                      "length": int(pf(r["length"])), "nc": float(pf(r["net_charge"])),
                      "npep": r.get("npep", r["length"]), "npocket": r.get("npocket", 0)})
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
fresh_seqs = {r["pep"] for r in fresh}; fresh_recs = {r["rec"] for r in fresh}
pdbb = [json.loads(l) for l in open("data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
Xpool = np.vstack([np.array([d["desc"] for d in pdbb]),
                   np.array([r["desc"] for r in ppikb if r["pep"] not in fresh_seqs
                             and r["rec"] not in fresh_recs])])
ypool = np.concatenate([np.array([float(d["y"]) for d in pdbb]),
                        np.array([r["y"] for r in ppikb if r["pep"] not in fresh_seqs
                                  and r["rec"] not in fresh_recs])])
Xf = np.array([r["desc"] for r in fresh]); yf = np.array([r["y"] for r in fresh])
q = np.array([abs(r["nc"]) for r in fresh]); neu = q <= 1

models = {
    "GBT (ours)": HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                                l2_regularization=1.0, random_state=0),
    "SVR (PPI-clone)": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                                ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]),
    "Ridge": Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))]),
}
preds = {}
print("\n=== (B) NEUTRAL (|q|<=1, n=%d) model-class on fresh-305 ===" % int(neu.sum()))
for name, m in models.items():
    p = m.fit(Xpool, ypool).predict(Xf); preds[name] = p
    r = pearsonr(yf[neu], p[neu])[0]; mae = np.mean(np.abs(yf[neu] - p[neu]))
    print(f"  {name:18s} NEUTRAL r={r:+.3f} MAE={mae:.2f}")
blend = 0.5 * preds["GBT (ours)"] + 0.5 * preds["SVR (PPI-clone)"]
rb = pearsonr(yf[neu], blend[neu])[0]; mb = np.mean(np.abs(yf[neu] - blend[neu]))
print(f"  {'GBT+SVR blend':18s} NEUTRAL r={rb:+.3f} MAE={mb:.2f}")
# also report overall + charged for the blend (does it cost elsewhere?)
for label, mk in [("ALL", np.ones(len(yf), bool)), ("CHARGED", q >= 2)]:
    rr = pearsonr(yf[mk], blend[mk])[0]; mm = np.mean(np.abs(yf[mk] - blend[mk]))
    print(f"  blend {label}: r={rr:+.3f} MAE={mm:.2f}")
json.dump(dict(coupling_mean=float(np.mean(np.abs(couplings))), grids=grids,
               dd_mae=float(np.mean(dd_err)), anc_mae=float(np.mean(anc_err)),
               neutral={k: float(pearsonr(yf[neu], p[neu])[0]) for k, p in preds.items()},
               blend_neutral_r=float(rb)), open("data/e282_diagnosis.json", "w"), indent=1)
print("\nsaved data/e282_diagnosis.json")
