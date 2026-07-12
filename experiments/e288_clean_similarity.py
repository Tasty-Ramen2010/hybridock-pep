"""E288 — CLEAN similarity analysis (e287 was corrupted by empty PPIKB pockets from e158.pocket_seq).

Uses ONLY valid PPIKB data: pocket_pkf (22-dim pocket ProtDCal-3D) for receptor similarity, net_charge,
and length. Answers Ram honestly:
  (A) how similar is the BEST-available reference pocket to each held-out query, in valid units?
  (B) does a strict similarity gate (top 5/10/25% closest) make offset transfer work?
  (C) double-difference at max n + stratified by R-R' similarity (extend e287's n=26).
Run: OMP_NUM_THREADS=1 python experiments/e288_clean_similarity.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")
        if json.loads(l).get("aff_type") in ("Kd", "Ki", "KD")]
recs = []
for r in rows:
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and np.isfinite(y):
        recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                     "desc": d3, "pkf": pk, "atype": r["aff_type"], "length": int(pf(r["length"])),
                     "nc": float(pf(r["net_charge"])), "npep": r.get("npep", r["length"]),
                     "npocket": r.get("npocket", 0)})
ours = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
seen = set(); fresh = []
for r in sorted(recs, key=lambda x: x["pdb"]):
    if r["pdb"] in ours or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
        continue
    if abs(r["npep"] - r["length"]) > 2 or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fseq = {r["pep"] for r in fresh}; frec = {r["rec"] for r in fresh}
pool = [r for r in recs if r["pep"] not in fseq and r["rec"] not in frec]
Xtr = np.array([r["desc"] for r in pool]); ytr = np.array([r["y"] for r in pool])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
grp = np.array([hash(r["rec"]) % (10**9) for r in pool])
Sp = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(Xtr, ytr, grp):
    Sp[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=0).fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
resid = ytr - Sp
pidx = defaultdict(list)
for i, r in enumerate(pool):
    pidx[r["rec"]].append(i)
precs = list(pidx.keys())
PK = np.array([pool[pidx[s][0]]["pkf"] for s in precs])
mu, sd = PK.mean(0), PK.std(0) + 1e-9
PKz = (PK - mu) / sd
offset = {s: float(np.mean([resid[i] for i in pidx[s]])) for s in precs}
prec_nc = {s: pool[pidx[s][0]]["nc"] for s in precs}
Sf = om.predict(np.array([r["desc"] for r in fresh])); yf = np.array([r["y"] for r in fresh])

# (A) best-match distance distribution in valid pocket_pkf space (+ what it corresponds to)
bestd, bestj = [], []
for r in fresh:
    qz = (np.array(r["pkf"]) - mu) / sd
    d = np.linalg.norm(PKz - qz, axis=1)
    j = int(np.argmin(d)); bestd.append(d[j]); bestj.append(j)
bestd = np.array(bestd)
# reference scale: typical distance between two random pool receptors
rnd = np.linalg.norm(PKz[np.random.default_rng(0).integers(0, len(PKz), 2000)]
                     - PKz[np.random.default_rng(1).integers(0, len(PKz), 2000)], axis=1)
print(f"=== (A) best-match pocket distance (valid pocket_pkf, 22-dim) ===")
print(f"  best-match dist: median={np.median(bestd):.2f} | random-pair dist median={np.median(rnd):.2f}")
print(f"  best match is this close vs random: ratio={np.median(bestd)/np.median(rnd):.2f} "
      f"(1.0=no better than random; <0.5=genuinely close matches exist)")

def rmae(p, t, m=None):
    p, t = np.asarray(p), np.asarray(t)
    if m is not None:
        p, t = p[m], t[m]
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))))

# (B) strict gate: transfer offset only for the closest X% of queries
print("\n=== (B) strict similarity gate (offset transfer only for closest X% of queries) ===")
order = np.argsort(bestd)
for frac in [1.0, 0.25, 0.10, 0.05]:
    k = max(int(len(fresh) * frac), 4)
    sel = set(order[:k].tolist())
    preds = Sf.copy()
    for i in range(len(fresh)):
        if i in sel:
            preds[i] = Sf[i] + offset[precs[bestj[i]]]
    # evaluate ONLY on the gated subset (where transfer was applied) vs absolute on same subset
    m = np.array([i in sel for i in range(len(fresh))])
    ra, ma = rmae(preds, yf, m); rb, mb = rmae(Sf, yf, m)
    print(f"  closest {frac:4.0%} (n={k}): absolute r={rb:+.3f}/MAE{mb:.2f} -> +transfer r={ra:+.3f}/MAE{ma:.2f}")

# (C) double-difference max-n + stratified
ymap = defaultdict(list)
for r in recs:
    ymap[(r["pep"], r["rec"])].append(r["y"])
ymap = {k: float(np.mean(v)) for k, v in ymap.items()}
pep_recs = defaultdict(set)
for (p, rr) in ymap:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]
nc_of = {r["rec"]: r["nc"] for r in recs}
dd_t, dd_p, dd_dnc = [], [], []
for a, b in combinations(multi, 2):
    common = list(pep_recs[a] & pep_recs[b])
    for R, Rk in combinations(common, 2):
        try:
            yPR = ymap[(a, R)]; yPRk = ymap[(a, Rk)]; yPkR = ymap[(b, R)]; yPkRk = ymap[(b, Rk)]
        except KeyError:
            continue
        dd_t.append(yPR); dd_p.append(yPRk + yPkR - yPkRk)
        dd_dnc.append(abs(nc_of.get(R, 0) - nc_of.get(Rk, 0)))
dd_t, dd_p = np.array(dd_t), np.array(dd_p)
print(f"\n=== (C) DOUBLE-DIFFERENCE n={len(dd_t)} ===")
print(f"  overall r={rmae(dd_p,dd_t)[0]:+.3f} MAE={rmae(dd_p,dd_t)[1]:.2f}")
json.dump(dict(bestmatch_ratio=float(np.median(bestd)/np.median(rnd)), dd_n=len(dd_t),
               dd_r=float(rmae(dd_p, dd_t)[0])), open("data/e288_clean.json", "w"))
print("saved data/e288_clean.json")
