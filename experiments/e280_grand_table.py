"""E280 — grand comparison: absolute-Kd under BOTH CV schemes + same-receptor lane, all datasets.

The honesty crux: random K-fold leaks receptors (same receptor in train+test => optimistic), leave-
receptor-out is the honest number. Report both for each dataset so 'where we stand' is unambiguous.
Run: OMP_NUM_THREADS=1 python experiments/e280_grand_table.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


def gbt():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)


def two_cv(X, y, grp):
    # optimistic random KFold
    p1 = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        p1[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
    # honest leave-receptor-out
    p2 = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(8, len(set(grp)))).split(X, y, grp):
        p2[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
    return (pearsonr(y, p1)[0], np.mean(np.abs(y - p1)),
            pearsonr(y, p2)[0], np.mean(np.abs(y - p2)))


out = {}

# ---- PPIKB ----
recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        yv = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if isinstance(d3, list) and isinstance(pk, list) and np.isfinite(yv):
        recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(yv),
                     "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
L = max(len(r["x"]) for r in recs); recs = [r for r in recs if len(r["x"]) == L]
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
grp = np.array([hash(r["rec"]) % (10**9) for r in recs])
out["PPIKB(n=%d)" % len(recs)] = two_cv(X, y, grp)

# ---- PDBbind peptides ----
PF = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
      "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
      "sasa_hb", "sasa_sb", "strength_bur"]
pdb = [json.loads(l) for l in open("data/pdbbind_peptides.jsonl")]
Xp = np.array([[float(d[f]) for f in PF] for d in pdb]); yp = np.array([float(d["y"]) for d in pdb])
gp = np.array([hash(d["pdb"]) % (10**9) for d in pdb])  # each pdb ~ its own receptor (no grouping info)
out["PDBbind(n=%d)" % len(pdb)] = two_cv(Xp, yp, gp)

# ---- same-receptor lane (PPIKB) ----
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
by = defaultdict(list)
for i, r in enumerate(recs):
    by[r["rec"]].append(i)
At, Aa, An = [], [], []
for s, idxs in by.items():
    if len({recs[i]["pep"] for i in idxs}) < 2:
        continue
    for i in idxs:
        others = [j for j in idxs if recs[j]["pep"] != recs[i]["pep"]]
        if others and np.isfinite(S[i]):
            d = np.linalg.norm(X[others] - X[i], axis=1)
            w = np.exp(-(d**2) / (2 * (np.median(d) or 1)**2)); w /= w.sum()
            An.append(yp[0])  # placeholder
            At.append(y[i]); Aa.append(float(np.sum(w * (y[others] + S[i] - S[others]))))
within_cold = pearsonr([y[i] for s, idxs in by.items() if len({recs[i]["pep"] for i in idxs}) >= 2
                        for i in idxs if np.isfinite(S[i])],
                       [S[i] for s, idxs in by.items() if len({recs[i]["pep"] for i in idxs}) >= 2
                        for i in idxs if np.isfinite(S[i])])[0]
anc_r = pearsonr(At, Aa)[0]; anc_mae = np.mean(np.abs(np.array(At) - np.array(Aa)))

print("=== ABSOLUTE Kd: optimistic (random KFold) vs HONEST (leave-receptor-out) ===")
print(f"{'dataset':18s} {'opt r':>7s} {'opt MAE':>8s} {'HONEST r':>9s} {'HONEST MAE':>11s}")
for k, (r1, m1, r2, m2) in out.items():
    print(f"{k:18s} {r1:>+7.3f} {m1:>8.2f} {r2:>+9.3f} {m2:>11.2f}")
print(f"\n=== SAME-RECEPTOR lane (PPIKB, n={len(At)} covered) ===")
print(f"  within-receptor COLD (absolute): r={within_cold:+.3f}")
print(f"  same-receptor ANCHORED:          r={anc_r:+.3f} MAE={anc_mae:.2f}")
json.dump({"absolute": {k: list(map(float, v)) for k, v in out.items()},
           "within_cold_r": float(within_cold), "anchored_r": float(anc_r),
           "anchored_mae": float(anc_mae)}, open("data/e280_grand.json", "w"))
print("\nsaved data/e280_grand.json")
