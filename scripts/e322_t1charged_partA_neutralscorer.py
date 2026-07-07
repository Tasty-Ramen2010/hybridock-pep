"""E322 Part A (concept N1 / T1-charged) — is a NEUTRAL-CALIBRATED fast scorer + charge correction a sound
decomposition? (Ram: "it would need a special calibrated fast_scorer instead of the normal one".)

The T1-charged design is  ΔG = fast_scorer(shape, charge-blind)  +  ΔG_charging_leg .
For that to be valid the fast scorer must be calibrated on its ACCURATE regime (neutral complexes) and its error
on charged complexes must be a clean, low-dimensional CHARGE correction — NOT a shape error. Test:
 1. train the scorer on NEUTRAL complexes only (|net q|<2) — the "special calibrated" scorer;
 2. apply to CHARGED complexes; residual = y - prediction = what the charge leg must supply;
 3. the decomposition is SOUND iff that residual is (a) sizeable, (b) charge-indexed (tracks |q|),
    (c) ~orthogonal to shape features (so adding a charge term doesn't fight the shape prediction).
This says whether a special calibrated scorer + charge leg (wired under --ultra) is the right architecture,
without needing charge-neutralized experimental ΔG (which don't exist).

Run: OMP_NUM_THREADS=1 python scripts/e322_t1charged_partA_neutralscorer.py
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
X = np.array([d["x"] for d in cache]); y = np.array([d["y"] for d in cache])
q = np.array([abs(float(d["q"])) for d in cache])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in cache])
FEATNAMES = ["cys_frac", "hb", "mean_burial", "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd",
             "poc_n", "poc_net", "rg_per_L", "f8", "f9", "f10", "f11", "f12", "f13", "f14"][:X.shape[1]]

neutral = q < 2
charged = q >= 2
print(f"neutral (|q|<2): n={neutral.sum()}   charged (|q|>=2): n={charged.sum()}")

# 1. "special calibrated" scorer = trained on NEUTRAL only. Honest neutral accuracy via leave-receptor-out
#    WITHIN the neutral set:
predN = np.full(neutral.sum(), np.nan)
Xn, yn, gn = X[neutral], y[neutral], grp[neutral]
for tr, te in GroupKFold(8).split(Xn, yn, gn):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(Xn[tr], yn[tr])
    predN[te] = m.predict(Xn[te])
print(f"neutral-calibrated scorer, held-out r on NEUTRAL = {pearsonr(predN, yn)[0]:+.3f} "
      f"(this is the regime it is meant for)")

# 2. apply the neutral-trained scorer to CHARGED complexes
full_neutral_model = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                   l2_regularization=1.0, random_state=0).fit(Xn, yn)
resid_c = y[charged] - full_neutral_model.predict(X[charged])
qc = q[charged]
print(f"\ncharge leg must supply: residual on charged  mean={resid_c.mean():+.2f}  std={resid_c.std():.2f} kcal/mol")

# 3a. is the residual charge-indexed?
print(f"3a. residual vs |net charge|            : r={pearsonr(qc, resid_c)[0]:+.3f}  "
      "(higher |q| → larger correction ⇒ charge-indexed)")

# 3b. is the residual ~orthogonal to shape features? (if strongly correlated, the scorer is shape-wrong on
#     charged, not just charge-blind → decomposition would be unsound)
print("3b. residual vs shape features (should be WEAK for a clean decomposition):")
shape_r = []
for j, nm in enumerate(FEATNAMES):
    r = pearsonr(X[charged][:, j], resid_c)[0]
    shape_r.append(abs(r))
    if abs(r) > 0.2:
        print(f"      {nm:14s}: r={r:+.3f}")
print(f"    max |shape correlation| = {max(shape_r):.3f}   mean = {np.mean(shape_r):.3f}")

sound = pearsonr(qc, resid_c)[0] > 0.15 and np.mean(shape_r) < 0.20
print("\nVERDICT: neutral-calibrated scorer + charge leg is " +
      ("a SOUND decomposition — the residual is charge-indexed and mostly shape-orthogonal, so a charging leg "
       "(added on top, under --ultra smoothing) targets exactly the scorer's blind spot." if sound else
       "only PARTLY clean — the charged residual carries some shape error too; the charge leg helps but the "
       "neutral scorer is not a perfect shape oracle on charged complexes.") +
      f"  [charge-index r={pearsonr(qc, resid_c)[0]:+.2f}, mean shape-corr={np.mean(shape_r):.2f}]")
