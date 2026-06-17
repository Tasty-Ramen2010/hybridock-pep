"""E289 — stress-test Ram's 'cheap diverse FEP' cycle: cross-corner triangulation for absolute ΔG.

Query (P1,R1) [want], reference (P2,R2) [known y]. Cross corners (P1,R2),(P2,R1). On the 26 PPIKB 2x2
grids (all 4 corners are real complexes with BOTH experimental y AND model features), compare:

  E1  single global anchor (model):     y_d + (S_a − S_d)
  E2  RAM cycle, MODEL cross-corners:    y_d + (S_b − S_d) + (S_c − S_d)
  E2k RAM cycle, K diverse refs avg:     mean over K reference corners of E2
  E3  same-receptor anchor (needs y_c):  y_c + (S_a − S_c)
  E4  pure double-diff (all exp corners): y_b + y_c − y_d
where a=(P1,R1) query, b=(P1,R2), c=(P2,R1), d=(P2,R2). S = leave-receptor-out model score.

Predicts: E2 ≈ E1 (model can't cancel its own b(R1)); only E4 (experimental corners) wins. If true, the
'cheap FEP' needs MEASURED cross-corners — substituting the biased scorer is circular.
Run: OMP_NUM_THREADS=1 python scripts/e289_cheap_fep.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from itertools import combinations, permutations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr


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
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and np.isfinite(y):
        recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                     "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
rec = [r["rec"] for r in recs]; pep = [r["pep"] for r in recs]
grp = np.array([hash(s) % (10**9) for s in rec])
# leave-receptor-out model score for every complex
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
idx = {(pep[i], rec[i]): i for i in range(len(recs))}
ymap = {k: y[i] for k, i in idx.items()}
Smap = {k: S[i] for k, i in idx.items()}
pep_recs = defaultdict(set)
for (p, rr) in idx:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]

# enumerate grids: (P1,P2) sharing (R1,R2). For each ORDERED assignment, predict y(P1,R1).
E = defaultdict(lambda: ([], []))  # name -> (truth, pred)
ref_pool = []  # for multi-ref: collect (P2,R2,y_d,S_d) reference corners globally
for P1, P2 in permutations(multi, 2):
    common = list(pep_recs[P1] & pep_recs[P2])
    for R1, R2 in permutations(common, 2):
        a, b, c, d = (P1, R1), (P1, R2), (P2, R1), (P2, R2)
        if not all(k in idx for k in (a, b, c, d)):
            continue
        ya = ymap[a]
        Sa, Sb, Sc, Sd = Smap[a], Smap[b], Smap[c], Smap[d]
        yb, yc, yd = ymap[b], ymap[c], ymap[d]
        E["E1_single_anchor"][0].append(ya); E["E1_single_anchor"][1].append(yd + Sa - Sd)
        E["E2_ram_model_cross"][0].append(ya); E["E2_ram_model_cross"][1].append(yd + (Sb - Sd) + (Sc - Sd))
        E["E3_same_receptor"][0].append(ya); E["E3_same_receptor"][1].append(yc + Sa - Sc)
        E["E4_double_diff_exp"][0].append(ya); E["E4_double_diff_exp"][1].append(yb + yc - yd)


def rmae(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))), len(t))


print(f"=== Ram's cheap-FEP cycle stress-test, n={len(E['E1_single_anchor'][0])} grid predictions ===")
print(f"  {'estimator':24s} {'r':>7s} {'MAE':>7s}  uses")
uses = {"E1_single_anchor": "1 exp anchor + model",
        "E2_ram_model_cross": "1 exp + MODEL cross-corners (Ram's cheap version)",
        "E3_same_receptor": "ref peptide MEASURED on query receptor",
        "E4_double_diff_exp": "ALL 3 cross-corners MEASURED (full double-diff)"}
res = {}
for name in ["E1_single_anchor", "E2_ram_model_cross", "E3_same_receptor", "E4_double_diff_exp"]:
    r, m, n = rmae(*E[name])
    res[name] = dict(r=float(r), mae=float(m))
    print(f"  {name:24s} {r:+7.3f} {m:7.2f}  {uses[name]}")
json.dump(res, open("data/e289_cheap_fep.json", "w"), indent=1)
print("\nKEY: if E2 ~ E1 (not better), MODEL cross-corners can't cancel b(R1) — the cheap version fails.")
print("If E4 >> E1, the cancellation is REAL but needs MEASURED corners (= true cheap-FEP, deployable).")
print("E3 = the practical middle (measure ONE reference peptide on the query receptor). saved data/e289_cheap_fep.json")
