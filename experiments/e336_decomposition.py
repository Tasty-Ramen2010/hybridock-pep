"""E336 — the E322 decomposition made concrete: ΔG = scorer(neutralised, shape) + FEP_charged.

Ram's architecture: our fast scorer is accurate on shape but blind to charge; the charged-FEP tier supplies the
charged term. Thermodynamic cycle:
    ΔG_bind(charged) = ΔG_bind(neutralised) − ΔΔG_elec       [ΔΔG_elec = decharge(bound) − decharge(free) > 0
                                                               ⇒ charges help binding]
So the hybrid prediction = scorer_neutral(complex) − ΔΔG_elec_FEP. This checks it on 2jqk (the one complex with
a computed FEP charged term, +6.26) vs the raw scorer.

HONEST: n=1 anecdote, and the FEP input's ACCURACY is under test (E335) — the SKEMPI D75N miss suggests our quick
FEP UNDER-estimates buried salt-bridge charged terms, so +6.26 may itself be low. This shows the ARCHITECTURE and
the sign, not a validated accuracy gain.

Run: OMP_NUM_THREADS=1 python experiments/e336_decomposition.py
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
X = np.array([d["x"] for d in cache]); y = np.array([d["y"] for d in cache])
q = np.array([abs(float(d["q"])) for d in cache])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in cache])
idx = {d["pdb"].lower(): i for i, d in enumerate(cache)}

TARGET = "2jqk"
FEP_DDG_ELEC = 6.26   # E332 decouple (E333 relative agrees 7.12); charged contribution to binding (favorable)
i = idx[TARGET]

# raw scorer: leave-receptor-out CV prediction (the number whose residual we quote)
raw_pred = np.nan
for tr, te in GroupKFold(8).split(X, y, grp):
    if i in te:
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        raw_pred = float(m.predict(X[i:i + 1])[0]); break

# scorer_neutral: trained ONLY on neutral complexes (|q|<2), applied to the charged target (= shape ΔG)
neu = q < 2
mn = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(X[neu], y[neu])
scorer_neutral = float(mn.predict(X[i:i + 1])[0])

decomp_pred = scorer_neutral - FEP_DDG_ELEC     # hybrid: shape − charged contribution
yt = float(y[i])

print(f"target {TARGET}  (net charge {int(q[i])}, true ΔG = {yt:+.2f} kcal)")
print(f"  raw scorer (leave-receptor-out)      = {raw_pred:+.2f}   |err|={abs(raw_pred-yt):.2f}")
print(f"  scorer_neutral (shape only)          = {scorer_neutral:+.2f}")
print(f"  FEP charged term  −ΔΔG_elec          = {-FEP_DDG_ELEC:+.2f}")
print(f"  HYBRID = scorer_neutral − ΔΔG_elec   = {decomp_pred:+.2f}   |err|={abs(decomp_pred-yt):.2f}")
print()
better = abs(decomp_pred - yt) < abs(raw_pred - yt)
print("VERDICT: hybrid " + ("BEATS" if better else "does NOT beat") + f" raw scorer on {TARGET} "
      f"({abs(decomp_pred-yt):.2f} vs {abs(raw_pred-yt):.2f} kcal). n=1 anecdote; needs the FEP term VALIDATED "
      "(E335) and many complexes before it is a real result. Architecture + sign demonstrated.")
