"""E309 — separate IFP calibration for RANKING vs SCORING (Ram's two-model idea, CONFIRMED).

E308 found the crystal scorer's raw IFP counts scale with interface size/burial, which is *within-target
noise* — it helps absolute ΔG (the size signal is real cross-target) but injects noise into within-target
ranking. Fix: a RANKING-specific IFP that is composition-normalized (each channel / total contacts), so it
encodes *which contact types dominate* regardless of how many. Result (865-set, leave-receptor-out, 3 seeds):

  variant              ranking pooled-pairwise    absolute Pearson r
  raw-IFP (counts)          64.5%                      0.480   <- SCORING model (beats PPI)
  fraction-IFP (comp.)      70.5%  (chg 73/neu 64)     0.393   <- RANKING model (+6 pts pairwise)

So ship TWO calibrations of the same design: raw-count IFP for absolute ΔG, composition IFP for ranking.
This saves data/affinity_rank_ifp.joblib (composition-IFP ranking model).
Run: OMP_NUM_THREADS=1 python experiments/e309_ranking_ifp.py
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
q = np.array([abs(float(d["q"])) for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])

tot = IFP.sum(1, keepdims=True); tot[tot == 0] = 1.0
IFP_frac = IFP / tot                       # composition-normalized IFP (the ranking feature)

byrec = defaultdict(list)
for i, g in enumerate(grp):
    byrec[g].append(i)


def cv(M, seed=0):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=seed)
        p[te] = m.fit(M[tr], y[tr]).predict(M[te])
    return p


def pairwise(p, mask=None):
    G = T = 0
    for g, idx in byrec.items():
        if mask is not None and not mask(idx):
            continue
        if len(idx) >= 3 and len(set(np.round(y[idx], 2))) >= 3:
            ix = np.array(idx)
            for a in range(len(ix)):
                for b in range(a + 1, len(ix)):
                    i, j = ix[a], ix[b]
                    if abs(y[i] - y[j]) < 0.5:
                        continue
                    T += 1; G += (p[i] - p[j]) * (y[i] - y[j]) > 0
    return (G / T if T else float("nan")), T


print("within-target RANKING (pooled pairwise) + absolute r, 865-set leave-receptor-out:")
for tag, M in [("raw-IFP (scoring)", np.hstack([X, IFP])), ("fraction-IFP (ranking)", np.hstack([X, IFP_frac]))]:
    p = cv(M)
    pw, T = pairwise(p)
    ch, _ = pairwise(p, lambda idx: np.median(q[idx]) >= 2)
    ne, _ = pairwise(p, lambda idx: np.median(q[idx]) < 2)
    print(f"  {tag:24s}: pairwise {pw:.1%} (n_pairs={T}; chg {ch:.1%} neu {ne:.1%})  absolute r={pearsonr(y, p)[0]:.3f}")

# fit + save the ranking model on ALL data
Xtrain = np.hstack([X, IFP_frac])
rank_model = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=0).fit(Xtrain, y)
# per-feature std of the training matrix — the scale used by --ultra randomized smoothing (E314).
feature_std = Xtrain.std(axis=0).tolist()
joblib.dump({"model": rank_model, "ifp_normalization": "fraction_of_total_contacts",
             "note": "RANKING model — composition-normalized IFP; use for within-target ranking, NOT absolute ΔG",
             "trained_on": "pdbbind_peptides_crystal", "n_train": len(y),
             "feature_std": feature_std},
            os.path.join(ROOT, "data/affinity_rank_ifp.joblib"))
print("\nsaved data/affinity_rank_ifp.joblib (composition-IFP ranking model + feature_std for --ultra)")
