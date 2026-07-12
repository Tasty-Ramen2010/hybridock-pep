"""E268 — the decisive disambiguation: does anchoring need a SAME-RECEPTOR ref, or do homologs suffice?

Earlier (e263/e264) I claimed the homolog radius was "flat to 50% identity". e266/e267 (leave-own-target-
out) suggest pure homolog transfer fails. Hypothesis: e264's loose clusters still CONTAINED same-exact-
receptor refs, so it never tested pure homolog transfer. This script settles it.

For each query with >=2 same-cluster refs (cluster = receptor k-mer Jaccard >= 0.5), partition by whether
an EXACT-same-receptor reference (identical protein_seq, different peptide) is available:
  GROUP A: has >=1 same-EXACT-receptor ref   -> b(R) cancels exactly
  GROUP B: refs are ALL different-receptor homologs (0.5-1.0 sim, but not identical) -> only b(R')~b(R)
Anchor each (bayes-weighted over its allowed refs) and report r/MAE per group vs absolute.
Prediction: A works (~0.6), B fails (~absolute or worse). Run: OMP_NUM_THREADS=1 python ...
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v
data = []
for r in rows:
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        yv = pf(r["y"]); d3 = pf(r.get("desc3d")); pk = pf(r.get("pocket_pkf"))
    except Exception:
        continue
    if isinstance(d3, list) and isinstance(pk, list):
        x = d3 + pk + [pf(r["length"]), pf(r["net_charge"])]
        if all(np.isfinite(x)) and np.isfinite(yv):
            data.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(yv), "x": x})
L = max(len(d["x"]) for d in data); data = [d for d in data if len(d["x"]) == L]
y = np.array([d["y"] for d in data]); X = np.array([d["x"] for d in data])
recs = [d["rec"] for d in data]; peps = [d["pep"] for d in data]
urec = sorted(set(recs)); ridx = {s: i for i, s in enumerate(urec)}
rid_of = np.array([ridx[s] for s in recs]); nR = len(urec)
def km(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
UK = [km(s) for s in urec]
def jac(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0
SIMR = np.zeros((nR, nR))
for i in range(nR):
    for j in range(i + 1, nR):
        SIMR[i, j] = SIMR[j, i] = jac(UK[i], UK[j])
# cluster at 0.5 (single linkage)
parent = list(range(nR))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(nR):
    for j in range(i + 1, nR):
        if SIMR[i, j] >= 0.5:
            parent[find(i)] = find(j)
clus = np.array([find(i) for i in range(nR)]); clus_of_row = clus[rid_of]
# leave-cluster-out absolute
FX = np.full(len(y), np.nan)
for c in np.unique(clus_of_row):
    te = clus_of_row == c
    if not te.all():
        mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=0)
        mdl.fit(X[~te], y[~te]); FX[te] = mdl.predict(X[te])
SIG = np.median([np.linalg.norm(X[i] - X[j]) for i in range(0, len(X), 7)
                 for j in range(0, len(X), 53) if i != j] or [1.0]) or 1.0
members = defaultdict(list)
for i, c in enumerate(clus_of_row):
    members[c].append(i)

def anchor(i, refs):
    d = np.array([np.linalg.norm(X[i] - X[r]) for r in refs])
    lw = -d ** 2 / (2 * SIG ** 2); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    return float(np.sum(w * (y[refs] + FX[i] - FX[refs])))

A_t, A_p, A_a, B_t, B_p, B_a = [], [], [], [], [], []
for c, mem in members.items():
    for i in mem:
        if not np.isfinite(FX[i]):
            continue
        same = [j for j in mem if j != i and rid_of[j] == rid_of[i] and peps[j] != peps[i]]
        homo = [j for j in mem if rid_of[j] != rid_of[i]]
        if same:                       # GROUP A: same-exact-receptor ref available
            A_t.append(y[i]); A_p.append(anchor(i, same)); A_a.append(FX[i])
        elif homo:                     # GROUP B: homolog-only (no same-receptor ref)
            B_t.append(y[i]); B_p.append(anchor(i, homo)); B_a.append(FX[i])

def rep(name, t, p, a):
    if len(t) < 4:
        print(f"{name}: n={len(t)} (too few)"); return None
    t, p, a = np.array(t), np.array(p), np.array(a)
    out = dict(n=len(t), anc_r=float(pearsonr(t, p)[0]), anc_mae=float(np.mean(np.abs(t - p))),
               abs_r=float(pearsonr(t, a)[0]), abs_mae=float(np.mean(np.abs(t - a))))
    print(f"{name}: n={out['n']:4d} | ABSOLUTE r={out['abs_r']:+.3f} MAE={out['abs_mae']:.2f}"
          f"  ->  ANCHORED r={out['anc_r']:+.3f} MAE={out['anc_mae']:.2f}")
    return out

print("PPIKB, cluster@0.5, leave-cluster-out absolute; anchor refs restricted per group:\n")
a = rep("GROUP A (same-EXACT-receptor ref)  ", A_t, A_p, A_a)
b = rep("GROUP B (homolog-only, 0.5-1.0 sim)", B_t, B_p, B_a)
json.dump(dict(groupA=a, groupB=b), open("data/e268_sametarget.json", "w"), indent=1)
print("\nVERDICT: if A anchored >> A absolute but B anchored ~<= B absolute, anchoring needs a")
print("SAME-RECEPTOR ref; pure homolog transfer does NOT carry b(R). (Corrects the 'flat radius' claim.)")
