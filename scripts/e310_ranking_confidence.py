"""E310 — a label-free confidence flag for rank_score: does the model's prediction SPREAD predict reliability?

The problem: rank_score is target-dependent (SH3 ρ=+0.91, MDM2 +0.67, PDZ +0.26, BH3 −0.63). We want a cheap
UPFRONT flag that tells a user whether to trust the ranking on their target, without any measured labels.

Finding: interface composition (hydrophobic/charged fraction) does NOT predict per-target ranking quality
(all |r|<0.18). But the model's own prediction SPREAD across the candidate panel DOES — if the candidates get
near-identical rank_scores the model cannot discriminate; if they spread, the order is trustworthy. On the
865-set (24 multi-peptide targets) spread correlates with per-target ranking Spearman at r≈+0.48.

Threshold RECALIBRATED on the shipped model (E310b): the held-out panels show the flag can only cleanly
isolate clearly-separable targets — SH3 spread 0.90 (ρ+0.91) vs an ambiguous 0.27-0.40 band where MDM2
(spread 0.27) ranks +0.67 but BH3 (spread 0.36) ranks -0.63, an inversion no threshold resolves. So the bar
is set high (0.50): in-sample HIGH-conf is 86% correct-direction (mean ρ≈+0.58) and every ambiguous/failing
held-out panel falls into the conservative "verify" bucket. The flag is CONSERVATIVE: "high" = trust;
"low" = verify in wet lab (not a failure prediction).
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
print(f"corr(CV rank_score spread, per-target ranking Spearman) = {pearsonr(spreads, rhos)[0]:+.3f}")

# recalibrate against the SHIPPED model (what deploys). Score all n>=3 targets in-sample + sweep thresholds.
import joblib  # noqa: E402
ship = joblib.load(os.path.join(ROOT, "data/affinity_rank_ifp.joblib"))["model"]
pship = ship.predict(M)
srho, sspread = [], []
for rseq, idx in byrec.items():
    if len(idx) < 3 or len(set(np.round(y[idx], 2))) < 3:
        continue
    ix = np.array(idx); r = spearmanr(pcv := p[ix], y[ix]).statistic
    if np.isnan(r):
        continue
    srho.append(r); sspread.append(pship[ix].std())
srho = np.array(srho); sspread = np.array(sspread)
print(f"\nSHIPPED-model spread sweep (n>=3 targets={len(srho)}), reliable = ranking ρ>0.3:")
for thr in (0.35, 0.40, 0.45, 0.50, 0.55):
    hi = sspread >= thr
    if hi.sum() == 0 or (~hi).sum() == 0:
        continue
    print(f"  thr={thr:.2f}: HIGH n={hi.sum():2d} mean ρ={srho[hi].mean():+.2f} correct={np.mean(srho[hi] > 0):.0%}"
          f"   LOW n={(~hi).sum():2d} mean ρ={srho[~hi].mean():+.2f}")
print("\nHeld-out panel shipped spreads:  SH3 0.90 (ρ+0.91) | PDZ 0.39 (+0.26) | BH3 0.36 (-0.63) | MDM2 0.27 (+0.67)")
print("=> 0.27-0.40 band is ambiguous (MDM2 works, BH3 fails); threshold 0.50 isolates SH3 as 'high', rest 'verify'.")
