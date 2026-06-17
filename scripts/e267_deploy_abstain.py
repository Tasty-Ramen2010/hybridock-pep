"""E267 — the CORRECTED deployment rule (turns e266's apparent failure into the design law).

e266 showed strict top-1-closest-receptor anchoring fails because (a) it forces anchoring even when the
nearest library receptor is <0.3 similar (83% of PPIKB queries), and (b) one receptor x 3 peptides is
high-variance. The fix this script tests:
  * ABSTAIN: only anchor if a homolog receptor with sim >= TAU exists (else fall back to absolute S).
  * POOL: when anchoring, use ALL refs across ALL homolog receptors with sim>=TAU, weighted by
    (receptor-sim x peptide-feature-sim). This is e264's pooling done in a leave-own-target-out frame.
Reports the HYBRID estimator (anchor where confident, absolute where not) vs absolute everywhere, plus the
coverage (what fraction of queries get a confident anchor) at each TAU. Leakage: own >=0.9 cluster excluded.
Run: OMP_NUM_THREADS=1 python scripts/e267_deploy_abstain.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

LEAK_TH = 0.9
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
        y = pf(r["y"]); d3 = pf(r.get("desc3d")); pk = pf(r.get("pocket_pkf"))
    except Exception:
        continue
    if not isinstance(d3, list) or not isinstance(pk, list):
        continue
    x = d3 + pk + [pf(r["length"]), pf(r["net_charge"])]
    if all(np.isfinite(x)) and np.isfinite(y):
        data.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y), "x": x})
L = max(len(d["x"]) for d in data)
data = [d for d in data if len(d["x"]) == L]
y = np.array([d["y"] for d in data]); X = np.array([d["x"] for d in data])
recs = [d["rec"] for d in data]
urec = sorted(set(recs)); ridx = {s: i for i, s in enumerate(urec)}
rid_of = np.array([ridx[s] for s in recs]); nR = len(urec)
print(f"rows {len(data)} | unique receptors {nR}", flush=True)


def km(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
UK = [km(s) for s in urec]
def jac(a, b):
    i = len(a & b); return i / (len(a) + len(b) - i) if (a and b) else 0.0
SIMR = np.zeros((nR, nR))
for i in range(nR):
    for j in range(i + 1, nR):
        SIMR[i, j] = SIMR[j, i] = jac(UK[i], UK[j])

parent = list(range(nR))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(nR):
    for j in range(i + 1, nR):
        if SIMR[i, j] >= LEAK_TH:
            parent[find(i)] = find(j)
clus = np.array([find(i) for i in range(nR)]); clus_of_row = clus[rid_of]

FX = np.full(len(y), np.nan)
for c in np.unique(clus_of_row):
    te = clus_of_row == c
    if not te.all():
        mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=0)
        mdl.fit(X[~te], y[~te]); FX[te] = mdl.predict(X[te])
print("absolute scores ready", flush=True)

SIG = np.median([np.linalg.norm(X[i] - X[j]) for i in range(0, len(X), 7)
                 for j in range(0, len(X), 53) if i != j] or [1.0]) or 1.0
rows_by_rid = defaultdict(list)
for i, rid in enumerate(rid_of):
    rows_by_rid[rid].append(i)


def anchored_pred(i, tau):
    """pool all refs on receptors with sim>=tau to query (different leak cluster). None if no homolog."""
    myclus = clus_of_row[i]; myrid = rid_of[i]
    refs, rsim = [], []
    for rj in range(nR):
        if clus[rj] == myclus:
            continue
        s = SIMR[myrid, rj]
        if s >= tau and rows_by_rid[rj]:
            for r in rows_by_rid[rj]:
                refs.append(r); rsim.append(s)
    if not refs:
        return None
    refs = np.array(refs); rsim = np.array(rsim)
    d = np.array([np.linalg.norm(X[i] - X[r]) for r in refs])
    logw = np.log(rsim + 1e-9) - d ** 2 / (2 * SIG ** 2)   # receptor-sim x peptide-sim
    logw -= logw.max(); w = np.exp(logw); w /= w.sum()
    return float(np.sum(w * (y[refs] + FX[i] - FX[refs])))


print(f"\n{'TAU':>5s} {'coverage':>9s} {'HYBRID r':>9s} {'HYBRID MAE':>11s} "
      f"{'absolute r':>11s} {'abs MAE':>8s} {'anchored-only(covered) r/MAE':>30s}")
out = []
for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
    hyb, cov_t, cov_p, cov_a = [], [], [], []
    n_cov = 0
    for i in range(len(y)):
        if not np.isfinite(FX[i]):
            continue
        a = anchored_pred(i, tau)
        if a is None:
            hyb.append(FX[i])
        else:
            hyb.append(a); n_cov += 1
            cov_t.append(y[i]); cov_p.append(a); cov_a.append(FX[i])
    valid = np.isfinite(FX)
    hyb = np.array(hyb); yt = y[valid]
    hr = pearsonr(yt, hyb)[0]; hm = np.mean(np.abs(yt - hyb))
    ar = pearsonr(yt, FX[valid])[0]; am = np.mean(np.abs(yt - FX[valid]))
    if len(cov_t) > 3:
        cr = pearsonr(cov_t, cov_p)[0]; cm = np.mean(np.abs(np.array(cov_t) - np.array(cov_p)))
        covabs_m = np.mean(np.abs(np.array(cov_t) - np.array(cov_a)))
        cstr = f"r={cr:+.3f} MAE={cm:.2f} (abs {covabs_m:.2f})"
    else:
        cstr = "n<4"
    cov = n_cov / valid.sum()
    print(f"{tau:>5.1f} {cov:>8.1%} {hr:>+9.3f} {hm:>11.2f} {ar:>+11.3f} {am:>8.2f} {cstr:>30s}")
    out.append(dict(tau=tau, coverage=float(cov), hybrid_r=float(hr), hybrid_mae=float(hm),
                    abs_r=float(ar), abs_mae=float(am)))
json.dump(out, open("data/e267_deploy.json", "w"), indent=1)
print("\nsaved data/e267_deploy.json")
print("read: where anchored-only(covered) r >> abs, anchoring works for THAT subset; coverage = how many")
print("queries have a sim>=TAU homolog. Hybrid = anchor-if-confident-else-absolute (the deployment rule).")
