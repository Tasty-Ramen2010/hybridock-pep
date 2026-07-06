"""E306 — within-target ranking: can HybriDock-Pep screen candidate peptides against ONE target?

The "contribute to another project" use case: a team has one receptor and several candidate peptides and
wants to know which to test. That is within-target RANKING (the per-receptor offset cancels, so it is not
capped by the blind-absolute ceiling). Measured on the 865-complex PDBbind crystal set with the shipped
crystal feature stack (17 geometry + 19 IFP), honest leave-receptor-out CV (GroupKFold by receptor sequence
— the model never sees the query receptor). For each receptor with >=k peptides (and >=3 distinct ΔG labels),
Spearman(predicted ΔG, measured ΔG).
Run: OMP_NUM_THREADS=1 python scripts/e306_within_target_ranking.py
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
y = np.array([d["y"] for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
M = np.hstack([X, IFP])

p = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(M, y, grp):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    p[te] = m.fit(M[tr], y[tr]).predict(M[te])

print(f"OVERALL (n={len(y)}, leave-receptor-out): Pearson r={pearsonr(y, p)[0]:.3f}  MAE={np.mean(np.abs(y - p)):.2f}")

byrec = defaultdict(list)
for i, g in enumerate(grp):
    byrec[g].append(i)

out = {}
for kmin in (3, 4, 5):
    rhos, npep = [], []
    for g, idx in byrec.items():
        if len(idx) >= kmin and len(set(np.round(y[idx], 2))) >= 3:
            idx = np.array(idx)
            rho = spearmanr(p[idx], y[idx]).statistic
            if not np.isnan(rho):
                rhos.append(rho); npep.append(len(idx))
    rhos = np.array(rhos)
    print(f"  within-target (>={kmin}/receptor): n_receptors={len(rhos)}, peptides={sum(npep)}, "
          f"median Spearman={np.median(rhos):+.3f}, mean={np.mean(rhos):+.3f}, "
          f"right-direction={np.mean(rhos > 0):.0%}")
    out[f"min{kmin}"] = {"n_receptors": len(rhos), "peptides": int(sum(npep)),
                         "median_spearman": float(np.median(rhos)), "mean_spearman": float(np.mean(rhos)),
                         "frac_right_direction": float(np.mean(rhos > 0))}
json.dump({"overall_r": float(pearsonr(y, p)[0]), "overall_mae": float(np.mean(np.abs(y - p))), **out},
          open(os.path.join(ROOT, "data/e306_within_target.json"), "w"), indent=1)
print("\nsaved data/e306_within_target.json")
