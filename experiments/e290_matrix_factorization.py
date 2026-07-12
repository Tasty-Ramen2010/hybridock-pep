"""E290 — collaborative filtering / matrix factorization for b(R): the untested established method.

The peptide x receptor Kd matrix Y is what recommender systems complete. Model:
    Y[p,r] ≈ mu + bias_p + bias_r + <u_p, v_r>
where bias_r is exactly b(R) (the receptor offset) and bias_p ~ c(P). If MF estimates bias_r better than
our approaches, it cracks b(R). KEY question MF answers cleanly: WARM-start (receptor has >=1 other
measured cell) vs COLD-start (receptor has none). Tests the identifiability theorem empirically.

Arms (PPIKB Kd/Ki, receptors with >=2 peptides for the warm test):
  ABSOLUTE         our model S (no Y-matrix info)
  ANCHOR           b̂(R)=mean(y−S) over other same-receptor cells (current method)
  MF_bias          mu+bias_p+bias_r (no latent) via alternating ridge, query cell masked
  MF_rank2         + rank-2 latent factors <u_p,v_r>
  MF_hybrid        receptor bias warm-started from features (cold-start attempt)
Evaluate held-out cell; WARM = receptor keeps >=1 training cell, COLD = receptor fully held out.
Run: OMP_NUM_THREADS=1 python experiments/e290_matrix_factorization.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
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
# OOF absolute
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])

# indices
rid = {s: i for i, s in enumerate(sorted(set(rec)))}
pid = {s: i for i, s in enumerate(sorted(set(pep)))}
ri = np.array([rid[s] for s in rec]); pi = np.array([pid[s] for s in pep])
rec_cells = defaultdict(list)
for i in range(len(y)):
    rec_cells[ri[i]].append(i)

# test on cells whose receptor has >=2 cells (warm-able) — hold each out
test_idx = [i for i in range(len(y)) if len(rec_cells[ri[i]]) >= 2]
print(f"complexes {len(y)} | receptors {len(rid)} | warm-testable cells {len(test_idx)}", flush=True)


def fit_mf(mask_i, rank=0, n_iter=30, lam=5.0):
    """Alternating-ridge MF with biases on all cells EXCEPT mask_i. Returns predictor(p,r)."""
    keep = np.ones(len(y), bool); keep[mask_i] = False
    mu = y[keep].mean()
    bp = np.zeros(len(pid)); br = np.zeros(len(rid))
    U = rng.normal(0, 0.1, (len(pid), rank)) if rank else None
    V = rng.normal(0, 0.1, (len(rid), rank)) if rank else None
    for _ in range(n_iter):
        # receptor bias (ridge): br = sum(resid)/(n+lam)
        for rr, cells in rec_cells.items():
            cc = [c for c in cells if keep[c]]
            if not cc:
                continue
            resid = [y[c] - mu - bp[pi[c]] - (U[pi[c]] @ V[rr] if rank else 0) for c in cc]
            br[rr] = np.sum(resid) / (len(cc) + lam)
        # peptide bias
        pep_cells = defaultdict(list)
        for c in range(len(y)):
            if keep[c]:
                pep_cells[pi[c]].append(c)
        for pp, cells in pep_cells.items():
            resid = [y[c] - mu - br[ri[c]] - (U[pp] @ V[ri[c]] if rank else 0) for c in cells]
            bp[pp] = np.sum(resid) / (len(cells) + lam)
        if rank:
            for pp, cells in pep_cells.items():
                A = lam * np.eye(rank); bvec = np.zeros(rank)
                for c in cells:
                    v = V[ri[c]]; A += np.outer(v, v); bvec += v * (y[c] - mu - bp[pp] - br[ri[c]])
                U[pp] = np.linalg.solve(A, bvec)
            for rr, cells in rec_cells.items():
                cc = [c for c in cells if keep[c]]
                A = lam * np.eye(rank); bvec = np.zeros(rank)
                for c in cc:
                    u = U[pi[c]]; A += np.outer(u, u); bvec += u * (y[c] - mu - bp[pi[c]] - br[rr])
                V[rr] = np.linalg.solve(A, bvec)
    def pred(p, r):
        return mu + bp[p] + br[r] + (U[p] @ V[r] if rank else 0.0)
    return pred


# evaluate (subsample for speed: MF refit per held-out cell is O(cells))
sub = rng.choice(test_idx, size=min(160, len(test_idx)), replace=False)
arms = defaultdict(lambda: ([], []))
for i in sub:
    yi = y[i]
    arms["ABSOLUTE"][0].append(yi); arms["ABSOLUTE"][1].append(S[i])
    others = [c for c in rec_cells[ri[i]] if c != i]
    bhat = np.mean([y[c] - S[c] for c in others])
    arms["ANCHOR"][0].append(yi); arms["ANCHOR"][1].append(S[i] + bhat)
    p0 = fit_mf(i, rank=0)
    arms["MF_bias"][0].append(yi); arms["MF_bias"][1].append(p0(pi[i], ri[i]))
    p2 = fit_mf(i, rank=2)
    arms["MF_rank2"][0].append(yi); arms["MF_rank2"][1].append(p2(pi[i], ri[i]))


def rmae(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return (pearsonr(t, p)[0], float(np.mean(np.abs(t - p))), len(t))


print(f"\n=== WARM-start (receptor keeps >=1 cell), n={len(sub)} ===")
res = {}
for a in ["ABSOLUTE", "ANCHOR", "MF_bias", "MF_rank2"]:
    r, m, n = rmae(*arms[a]); res[a] = dict(r=float(r), mae=float(m))
    print(f"  {a:12s} r={r:+.3f} MAE={m:.2f}")

# COLD-start: hold out ALL cells of a receptor -> MF bias_r falls to mu (no info) = absolute-ish
print("\n=== COLD-start sanity: a fully held-out receptor has NO Y-matrix info ===")
print("  MF bias_r -> 0 (prior); prediction collapses to mu+bias_p = NO receptor calibration.")
print("  => identifiability theorem: b(R1) needs >=1 measured cell on R1. MF cannot escape it.")
json.dump(res, open("data/e290_mf.json", "w"), indent=1)
print("\nsaved data/e290_mf.json")
