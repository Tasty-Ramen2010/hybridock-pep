"""E310 — a label-free confidence flag for rank_score: does the model's prediction SPREAD predict reliability?

The problem: rank_score is target-dependent (SH3 ρ=+0.91, MDM2 +0.67, PDZ +0.26, BH3 −0.63). We want a cheap
UPFRONT flag that tells a user whether to trust the ranking on their target, without any measured labels.

Finding: interface composition (hydrophobic/charged fraction) does NOT predict per-target ranking quality
(all |r|<0.18). But the model's own prediction SPREAD across the candidate panel DOES — if the candidates get
near-identical rank_scores the model cannot discriminate; if they spread, the order is trustworthy. On the
865-set (24 multi-peptide targets) spread correlates with per-target ranking Spearman at r≈+0.48.

Flag (threshold 0.40, n>=4 targets): HIGH-conf spread>=0.40 -> mean ranking ρ +0.71 (100% correct direction);
LOW-conf -> +0.12. The flag is CONSERVATIVE: "high" = reliable; "low" = uncertain (may still rank fine on a
tight panel), a "verify in wet lab" signal.
Run: OMP_NUM_THREADS=1 python scripts/e310_ranking_confidence.py
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
tot = IFP.sum(1, keepdims=True); tot[tot == 0] = 1.0; IFP_frac = IFP / tot
M = np.hstack([X, IFP_frac])

p = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(M, y, grp):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    p[te] = m.fit(M[tr], y[tr]).predict(M[te])

byrec = defaultdict(list)
for i, d in enumerate(data):
    byrec[d["rseq"]].append(i)

spreads, rhos = [], []
for rseq, idx in byrec.items():
    if len(idx) < 4 or len(set(np.round(y[idx], 2))) < 3:
        continue
    ix = np.array(idx); rho = spearmanr(p[ix], y[ix]).statistic
    if np.isnan(rho):
        continue
    spreads.append(p[ix].std()); rhos.append(rho)
spreads = np.array(spreads); rhos = np.array(rhos)

print(f"targets (n>=4 candidates): {len(rhos)}")
print(f"corr(rank_score spread, per-target ranking Spearman) = {pearsonr(spreads, rhos)[0]:+.3f}")
THR = 0.40
hi = rhos[spreads >= THR]; lo = rhos[spreads < THR]
print(f"\nCONFIDENCE FLAG (spread threshold {THR}):")
print(f"  HIGH-conf (spread>={THR}): {len(hi)} targets, mean ranking ρ={hi.mean():+.3f}, correct-direction={np.mean(hi > 0):.0%}")
print(f"  LOW-conf  (spread< {THR}): {len(lo)} targets, mean ranking ρ={lo.mean():+.3f}, correct-direction={np.mean(lo > 0):.0%}")
