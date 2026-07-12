"""E19 step-2 validation — does pocket-baseline GENERALIZE, or is it crystal-65-specific?

The acid tests (all use cached /tmp/e19_{cr,pb}.json — no re-extraction):

  V1  SIGN UNIVERSALITY: is corr(pocket_hydrophobicity, ΔG) the same SIGN in crystal-65
      AND in PEPBI (independently)? If yes, the physics transfers even if slope doesn't.
  V2  LEAVE-DATASET-OUT (the real test): train pocket+interface on crystal-65 -> predict
      PEPBI distinct-target FAMILY MEANS (17, different source/ITC/modeled). And reverse.
      Positive => generalizes across data sources. Negative => crystal-specific.
  V3  POOLED DISTINCT-TARGET CV: crystal-65 (65 distinct) + PEPBI 17 family-means = 82
      distinct targets; 5-fold + LOO, ridge (small-n safe). The honest deployable number.
  V4  ROBUSTNESS: ridge vs OLS, feature-subset stability.

Distinct-target = one row per genuinely different pocket (crystal: each pdb; pepbi: each
family-mean, since within-family pockets are near-identical).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

cr = json.loads(Path("/tmp/e19_cr.json").read_text())
pb = json.loads(Path("/tmp/e19_pb.json").read_text())

POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
FEATS = POCK + IFACE


def fam_means(recs):
    """Collapse to one row per binding group (distinct pocket): mean features + mean y."""
    g = {}
    for r in recs:
        g.setdefault(r["grp"], []).append(r)
    out = []
    for gid, rs in g.items():
        row = {f: float(np.mean([r.get(f, 0.0) for r in rs])) for f in FEATS}
        row["y"] = float(np.mean([r["y"] for r in rs]))
        row["grp"] = gid
        row["n"] = len(rs)
        out.append(row)
    return out


def _mat(recs, feats):
    return np.array([[r.get(f, 0.0) for f in feats] for r in recs], float)


def ridge_fit(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    p = A.shape[1]
    R = lam * np.eye(p); R[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
    Zte = (Xte - mu) / sd
    return np.column_stack([np.ones(len(Zte)), Zte]) @ w


def loo(recs, feats, lam=1.0):
    X = _mat(recs, feats); y = np.array([r["y"] for r in recs])
    pred = np.zeros(len(recs))
    for i in range(len(recs)):
        tr = [j for j in range(len(recs)) if j != i]
        pred[i] = ridge_fit(X[tr], y[tr], X[i:i+1], lam)[0]
    return pearsonr(pred, y).statistic, float(np.sqrt(np.mean((pred - y) ** 2)))


print("=" * 70)
print("V1  SIGN UNIVERSALITY — corr(feature, ΔG) within each dataset independently")
print("=" * 70)
crf = fam_means(cr)   # crystal: 65 singleton 'families'
pbf = fam_means(pb)   # pepbi: 17 distinct-pocket family means
print(f"  crystal distinct targets={len(crf)}  pepbi distinct families={len(pbf)}")
print(f"\n  {'feature':<12}{'crystal-65 r':>14}{'pepbi-fam r':>14}{'same sign?':>12}")
for f in POCK + ["bsa_hyd", "sasa_hb"]:
    rc = pearsonr([r[f] for r in cr], [r["y"] for r in cr]).statistic
    rp = pearsonr([r[f] for r in pbf], [r["y"] for r in pbf]).statistic
    same = "YES" if (rc * rp > 0) else "no"
    print(f"  {f:<12}{rc:>14.3f}{rp:>14.3f}{same:>12}")

print("\n" + "=" * 70)
print("V2  LEAVE-DATASET-OUT — train on one source, predict the OTHER (acid test)")
print("=" * 70)
for name, feats in [("pocket only", POCK), ("pocket+interface", FEATS), ("interface only", IFACE)]:
    Xtr, ytr = _mat(cr, feats), np.array([r["y"] for r in cr])
    Xte, yte = _mat(pbf, feats), np.array([r["y"] for r in pbf])
    pred = ridge_fit(Xtr, ytr, Xte, lam=2.0)
    r_cp = pearsonr(pred, yte).statistic
    # reverse: train pepbi-fam -> predict crystal
    pred2 = ridge_fit(Xte, yte, Xtr, lam=2.0)
    r_pc = pearsonr(pred2, ytr).statistic
    print(f"  {name:<20} crystal->pepbiFAM r={r_cp:+.3f}   pepbiFAM->crystal r={r_pc:+.3f}")
print("  (positive crystal->pepbiFAM = pocket signal transfers across data sources)")

print("\n" + "=" * 70)
print("V3  POOLED DISTINCT-TARGET (crystal-65 + 17 pepbi-fam = 82), LOO ridge")
print("=" * 70)
pool = cr + pbf
for name, feats in [("interface only", IFACE), ("pocket only", POCK), ("pocket+interface", FEATS)]:
    r, rmse = loo(pool, feats, lam=2.0)
    print(f"  {name:<20} LOO r={r:+.3f}  RMSE={rmse:.2f}")
ybar = np.array([r["y"] for r in pool])
print(f"  mean-baseline RMSE={ybar.std():.2f}   (n={len(pool)} distinct targets)")

print("\n" + "=" * 70)
print("V4  ROBUSTNESS — pooled LOO pocket+interface across ridge strengths")
print("=" * 70)
for lam in [0.3, 1.0, 2.0, 5.0, 10.0]:
    r, rmse = loo(pool, FEATS, lam)
    print(f"  lambda={lam:<5} r={r:+.3f}  RMSE={rmse:.2f}")

print("\n=== VERDICT ===")
r_cp = pearsonr(ridge_fit(_mat(cr, FEATS), np.array([r['y'] for r in cr]),
                          _mat(pbf, FEATS), 2.0), np.array([r['y'] for r in pbf])).statistic
r_pool, _ = loo(pool, FEATS, 2.0)
print(f"  leave-DATASET-out crystal->pepbiFAM: {r_cp:+.3f}  "
      f"({'GENERALIZES' if r_cp > 0.25 else 'weak/does-not-transfer'})")
print(f"  pooled 82-distinct-target LOO:       {r_pool:+.3f}  "
      f"({'robust' if r_pool > 0.4 else 'crystal-specific' if r_pool < 0.3 else 'moderate'})")
