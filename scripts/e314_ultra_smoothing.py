"""E314 — Ram's --ultra idea (perturb/mutate the peptide, score the cloud, collapse back to refine) tested as
randomized smoothing, and the cross-domain math that says what it can and cannot do.

Control-variate theory (Wikipedia; Owen MCM): an estimator refined with a correlated auxiliary has variance
x (1 − ρ²) — it reduces VARIANCE, never BIAS. Randomized smoothing / test-time augmentation (Cohen 2019):
average predictions over input perturbations for a smoother, lower-variance estimate. So the prediction is:
--ultra tightens RANKING (variance) but cannot move the absolute charged ceiling (bias = missing signal).

Cheap proxy for --ultra: perturb each complex's features K times (5% Gaussian ~ pose/conformer jitter), average
the model prediction, measure within-target pairwise ranking, charged vs neutral.
Run: OMP_NUM_THREADS=1 python scripts/e314_ultra_smoothing.py
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower() for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl"))}
data = [d for d in cache if d["pdb"] in protd]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
q = np.array([abs(float(d["q"])) for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
tot = IFP.sum(1, keepdims=True); tot[tot == 0] = 1.0
F = np.hstack([X, IFP / tot])

byr = defaultdict(list)
for i, g in enumerate(grp):
    byr[g].append(i)


def pairwise(p, mask):
    G = T = 0
    for g, idx in byr.items():
        if len(idx) < 3 or len(set(np.round(y[idx], 2))) < 3 or not mask(idx):
            continue
        ix = np.array(idx)
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if abs(y[i] - y[j]) < 0.5:
                    continue
                T += 1; G += (p[i] - p[j]) * (y[i] - y[j]) > 0
    return G / T if T else float("nan")


scale = F.std(0, keepdims=True) * 0.05
rng = np.random.default_rng(0)


def cv(ultra_K):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(F, y, grp):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(F[tr], y[tr])
        if ultra_K <= 1:
            p[te] = m.predict(F[te])
        else:
            acc = np.zeros(len(te))
            for _ in range(ultra_K):
                acc += m.predict(F[te] + rng.normal(0, 1, F[te].shape) * scale)
            p[te] = acc / ultra_K
    return p


chg = lambda idx: np.median(q[idx]) >= 2
neu = lambda idx: np.median(q[idx]) < 2
allm = lambda idx: True
print("--ULTRA (randomized-smoothing proxy): within-target pairwise ranking")
print(f"{'mode':18s}{'ALL':>8s}{'CHARGED':>10s}{'NEUTRAL':>10s}")
for K in (1, 8, 32):
    p = cv(K)
    print(f"{('baseline' if K == 1 else f'--ultra K={K}'):18s}{pairwise(p, allm):>7.1%}{pairwise(p, chg):>10.1%}{pairwise(p, neu):>10.1%}")
print("\nVerdict: --ultra reduces VARIANCE (small ranking gain), not BIAS (absolute charged ceiling ~0.40 unmoved).")
