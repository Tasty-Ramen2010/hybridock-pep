"""E308 — probe battery: what the crystal scorer actually reads, and how to measure ranking honestly.

Small tests on the 865-complex cache + fresh crystals:
  1. within-target PAIRWISE ordering accuracy (screening-relevant metric) vs median Spearman — they DISAGREE
     on charged targets, so no charge-routing rule is stable (corrects E307).
  2. side-chain identity sensitivity: relabel a peptide to poly-ALA (coords unchanged) and rescore.
Run: OMP_NUM_THREADS=1 python scripts/e308_probe_battery.py
Structural probes (perturbation radius, poly-ALA) are documented in docs/DEVELOPMENT_TIMELINE.md E308; the
CV metrics below reproduce from the shipped caches.
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
y = np.array([d["y"] for d in data]); q = np.array([abs(float(d["q"])) for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])


def cv(M, target=None):
    t = y if target is None else target
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
        p[te] = m.fit(M[tr], t[tr]).predict(M[te])
    return p


byrec = defaultdict(list)
for i, g in enumerate(grp):
    byrec[g].append(i)


def rankable(idx):
    return len(idx) >= 3 and len(set(np.round(y[idx], 2))) >= 3


def pairwise(p, idx):
    idx = np.array(idx); good = tot = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if abs(y[i] - y[j]) < 0.5:
                continue
            tot += 1; good += (p[i] - p[j]) * (y[i] - y[j]) > 0
    return good, tot


def report(name, p):
    rhos = []; G = T = 0
    for g, idx in byrec.items():
        if rankable(idx):
            rho = spearmanr(p[np.array(idx)], y[np.array(idx)]).statistic
            if not np.isnan(rho):
                rhos.append(rho)
            gg, tt = pairwise(p, idx); G += gg; T += tt
    print(f"  {name:20s}: median ρ={np.median(rhos):+.3f}  pooled pairwise={G}/{T}={G/T:.1%}")
    return G / T


print("=== within-target ranking: median Spearman vs pooled pairwise (geom vs geom+IFP) ===")
report("geom-only", cv(X))
report("geom+IFP", cv(np.hstack([X, IFP])))
