"""E278 — prototype I2 (probe-peptide receptor fingerprint) and I6 (mixed-effects few-anchor model).

Both are SAME-RECEPTOR ideas (the validated lane). PPIKB Kd/Ki, OOF absolute S (GroupKFold by receptor).

I2 — PROBE-PEPTIDE FINGERPRINT (LIE-style per-receptor calibration): to deploy on a receptor, measure K
   "probe" peptides; b̂(R) = mean over probes of (y − S). Then a held-out query on that receptor is
   predicted S(query) + b̂(R). Tests how anchored accuracy scales with K = 1,2,3,5 probes. This is the
   deployment protocol (iGEM mode-b): how many references must the wet lab measure?

I6 — MIXED-EFFECTS few-anchor model: per-receptor random intercept shrunk toward 0 by a ridge prior,
   b̂(R) = n/(n+λ) · mean_residual. Compares shrinkage-anchor vs raw-mean-anchor in the 1-anchor regime
   (does Bayesian shrinkage beat naive subtraction when references are scarce/noisy?).
Run: OMP_NUM_THREADS=1 python scripts/e278_i2_i6.py
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
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
        continue
    recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
L = max(len(r["x"]) for r in recs); recs = [r for r in recs if len(r["x"]) == L]
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs]); rec = [r["rec"] for r in recs]
grp = np.array([hash(s) % (10 ** 9) for s in rec])
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
resid = y - S
by = defaultdict(list)
for i, s in enumerate(rec):
    by[s].append(i)
panels = {s: idxs for s, idxs in by.items() if len({recs[i]["pep"] for i in idxs}) >= 3}  # need >=3 to probe+test
print(f"receptors with >=3 distinct peptides (usable panels): {len(panels)}", flush=True)


def rmae(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))))


# ---- I2: probe-peptide fingerprint, accuracy vs #probes K ----
print("\n=== I2: probe-peptide receptor fingerprint (b̂ from K probes, predict held-out) ===")
print(f"  {'#probes K':10s} {'anchored r':>11s} {'anchored MAE':>13s} {'absolute MAE':>13s}")
i2 = {}
for K in [0, 1, 2, 3, 5]:
    pt, pp, pa = [], [], []
    for s, idxs in panels.items():
        idxs = list(idxs)
        for trial in range(3):  # average over probe samplings
            rng.shuffle(idxs)
            probes = idxs[:K]; tests = idxs[K:]
            if not tests or (K > 0 and not probes):
                continue
            bhat = np.mean([resid[p] for p in probes]) if K > 0 else 0.0
            for t in tests:
                pt.append(y[t]); pp.append(S[t] + bhat); pa.append(S[t])
    r, ma = rmae(pp, pt); _, mabs = rmae(pa, pt)
    i2[K] = dict(r=float(r), mae=float(ma))
    tag = "(=absolute)" if K == 0 else ""
    print(f"  {K:<10d} {r:>+11.3f} {ma:>13.2f} {mabs:>13.2f} {tag}")

# ---- I6: shrinkage vs naive single-anchor ----
print("\n=== I6: mixed-effects shrinkage vs naive mean (1-anchor regime) ===")
print(f"  {'method':24s} {'r':>8s} {'MAE':>8s}")
for lam, name in [(0.0, "naive mean (λ=0)"), (1.0, "shrinkage λ=1"), (3.0, "shrinkage λ=3"),
                  (8.0, "shrinkage λ=8")]:
    pt, pp = [], []
    for s, idxs in panels.items():
        idxs = list(idxs)
        for t_i in range(len(idxs)):
            test = idxs[t_i]; anchors = idxs[:t_i] + idxs[t_i + 1:]
            anchors = anchors[:1]  # 1-anchor regime (scarce)
            n = len(anchors)
            bhat = (n / (n + lam)) * np.mean([resid[a] for a in anchors]) if n else 0.0
            pt.append(y[test]); pp.append(S[test] + bhat)
    r, ma = rmae(pp, pt)
    print(f"  {name:24s} {r:>+8.3f} {ma:>8.2f}")
json.dump(dict(i2=i2, n_panels=len(panels)), open("data/e278_i2_i6.json", "w"))
print("\nsaved data/e278_i2_i6.json")
