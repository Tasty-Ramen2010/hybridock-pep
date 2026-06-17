"""E266 — DEPLOYMENT simulation of Ram's "pick the top closest receptor, anchor, verify" idea.

Unlike e263/e264 (which anchored to ALL same-cluster refs), this is the strict real-product flow:
for each held-out query peptide on receptor R, search the library for the SINGLE most similar OTHER
receptor that has >=1 known Kd, anchor to its top-3 peptide-similar references, predict Kd, verify.

The deliverable is the DEPLOYMENT CONFIDENCE CURVE: prediction error binned by how similar the nearest
library receptor is. That answers "for a NEW peptide, given our closest library receptor is X% similar,
expect error Y" — directly actionable for the iGEM tool.

Leakage control: the query's own receptor cluster (k-mer Jaccard >= LEAK_TH) is excluded from BOTH the
absolute-model training and anchor selection. Shuffle control included.
Run: OMP_NUM_THREADS=1 python scripts/e266_deployment_sim.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
LEAK_TH = 0.9   # receptors >= this k-mer Jaccard are "the same target" -> excluded (no self-anchor)

rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]


def pf(v):
    if isinstance(v, str):
        v = v.strip()
        return json.loads(v) if v.startswith("[") else float(v)
    return v


data = []
for r in rows:
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r.get("desc3d")); pk = pf(r.get("pocket_pkf"))
    except Exception:
        continue
    if not isinstance(d3, list) or not isinstance(pk, list):
        continue
    x = d3 + pk + [pf(r["length"]), pf(r["net_charge"])]
    if not all(np.isfinite(x)) or not np.isfinite(y):
        continue
    data.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y), "x": x})
L = max(len(d["x"]) for d in data)
data = [d for d in data if len(d["x"]) == L]
y = np.array([d["y"] for d in data]); X = np.array([d["x"] for d in data])
recs = [d["rec"] for d in data]; peps = [d["pep"] for d in data]
print(f"PPIKB Kd/Ki structured rows: {len(data)} | feat dim {L}", flush=True)

# unique receptors + cached k-mer sets (compute receptor-receptor similarity cheaply)
urec = sorted(set(recs))
ridx = {s: i for i, s in enumerate(urec)}
rid_of = np.array([ridx[s] for s in recs])


def km(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
UK = [km(s) for s in urec]


def jac(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b); return i / (len(a) + len(b) - i)


# full receptor-receptor Jaccard (|urec| modest)
nR = len(urec)
SIMR = np.zeros((nR, nR))
for i in range(nR):
    for j in range(i + 1, nR):
        s = jac(UK[i], UK[j]); SIMR[i, j] = SIMR[j, i] = s
print(f"unique receptors: {nR} | receptor-sim matrix built", flush=True)

# leakage clusters: connected components at LEAK_TH (union-find)
parent = list(range(nR))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(nR):
    for j in range(i + 1, nR):
        if SIMR[i, j] >= LEAK_TH:
            parent[find(i)] = find(j)
clus = np.array([find(i) for i in range(nR)])
clus_of_row = clus[rid_of]

# leave-CLUSTER-out absolute model
FX = np.full(len(y), np.nan)
for c in np.unique(clus_of_row):
    te = clus_of_row == c
    if te.all():
        continue
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    m.fit(X[~te], y[~te]); FX[te] = m.predict(X[te])
print("absolute leave-cluster-out scores ready", flush=True)

# peptide-feature scale for ref ranking within the chosen receptor
SIG = np.median([np.linalg.norm(X[i] - X[j]) for i in range(0, len(X), 7)
                 for j in range(0, len(X), 53) if i != j] or [1.0]) or 1.0

# rows grouped by unique receptor id
rows_by_rid = defaultdict(list)
for i, rid in enumerate(rid_of):
    rows_by_rid[rid].append(i)

records = []   # per-query: closest-receptor sim, absolute err, anchored err, shuffle err
for i in range(len(y)):
    if not np.isfinite(FX[i]):
        continue
    myclus = clus_of_row[i]
    myrid = rid_of[i]
    # candidate receptors NOT in my leakage cluster
    cand = [rj for rj in range(nR) if clus[rj] != myclus and rows_by_rid[rj]]
    if not cand:
        continue
    sims = np.array([SIMR[myrid, rj] for rj in cand])
    best = cand[int(np.argmax(sims))]
    closest_sim = float(sims.max())
    # anchor on top-3 peptide-feature-similar refs ON that closest receptor
    refs = rows_by_rid[best]
    d = np.array([np.linalg.norm(X[i] - X[r]) for r in refs])
    top = [refs[t] for t in np.argsort(d)[:3]]
    logw = -np.array([np.linalg.norm(X[i] - X[r]) for r in top]) ** 2 / (2 * SIG ** 2)
    logw -= logw.max(); w = np.exp(logw); w /= w.sum()
    anc = float(np.sum(w * (np.array([y[r] for r in top]) + FX[i] - np.array([FX[r] for r in top]))))
    # shuffle: a RANDOM other-cluster receptor instead of the closest
    rc = cand[int(rng.integers(len(cand)))]
    rrefs = rows_by_rid[rc]
    rs = [rrefs[t] for t in np.argsort([np.linalg.norm(X[i] - X[r]) for r in rrefs])[:3]]
    sh = float(np.mean([y[r] + FX[i] - FX[r] for r in rs]))
    records.append(dict(sim=closest_sim, abs_err=abs(y[i] - FX[i]), anc_err=abs(y[i] - anc),
                        shuf_err=abs(y[i] - sh), y=float(y[i]), anc=anc, absol=float(FX[i]), sh=sh))

R = records
print(f"\ndeployment queries scored: {len(R)}", flush=True)


def corr(key):
    t = [r["y"] for r in R]; p = [r[key] for r in R]
    return pearsonr(t, p)[0]


def mae(k):
    return float(np.mean([r[k] for r in R]))


print(f"OVERALL (all {len(R)} queries, top-1-closest-receptor anchoring):")
print(f"  ABSOLUTE   r={corr('absol'):+.3f}  MAE={mae('abs_err'):.2f}")
print(f"  ANCHORED   r={corr('anc'):+.3f}  MAE={mae('anc_err'):.2f}")
print(f"  SHUFFLE    r={corr('sh'):+.3f}  MAE={mae('shuf_err'):.2f}")

# random-100 subset (Ram's "test 100")
idx100 = rng.choice(len(R), size=min(100, len(R)), replace=False)
S = [R[k] for k in idx100]
def maeS(k):
    return float(np.mean([s[k] for s in S]))
print(f"\nRANDOM-100 subset: ABSOLUTE MAE={maeS('abs_err'):.2f} | "
      f"ANCHORED MAE={maeS('anc_err'):.2f} | SHUFFLE MAE={maeS('shuf_err'):.2f}")

# CONFIDENCE CURVE: error vs closest-receptor similarity
print("\nDEPLOYMENT CONFIDENCE CURVE (anchored MAE by closest-receptor similarity):")
print(f"  {'sim bin':12s} {'n':>4s} {'abs MAE':>8s} {'anc MAE':>8s} {'anc<abs?':>8s}")
bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
curve = []
for lo, hi in bins:
    g = [r for r in R if lo <= r["sim"] < hi]
    if not g:
        continue
    am = np.mean([r["abs_err"] for r in g]); nm = np.mean([r["anc_err"] for r in g])
    print(f"  [{lo:.1f},{hi:.1f})   {len(g):>4d} {am:>8.2f} {nm:>8.2f} {'YES' if nm < am else 'no':>8s}")
    curve.append(dict(lo=lo, hi=hi, n=len(g), abs_mae=float(am), anc_mae=float(nm)))

json.dump(dict(n=len(R), overall_abs_mae=mae("abs_err"), overall_anc_mae=mae("anc_err"),
               overall_shuf_mae=mae("shuf_err"), anc_r=float(corr("anc")), abs_r=float(corr("absol")),
               curve=curve), open("data/e266_deployment.json", "w"), indent=1)
print("\nsaved data/e266_deployment.json")
