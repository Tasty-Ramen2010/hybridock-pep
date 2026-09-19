#!/usr/bin/env python
"""Ship the nonlinear fit, then ask the residual what feature we still lack.

PART 1 — THE UPGRADE. Under matched clustered CV on the same 59 features, a linear fit reaches
r 0.239 / MAE 1.447 and a random forest reaches r 0.475 / MAE 1.298. The collapse toward the mean
was a property of the FIT, not of the features: a linear model on 59 mutually correlated
descriptors correctly shrinks, because linearly that is all the signal there is. This measures the
upgrade honestly -- same splits, same features, nothing changed but the model class -- and reports
the calibration slope, which is the quantity that was 0.135 and is the actual symptom.

PART 2 — WHAT IS STILL MISSING. With the fit no longer the bottleneck, the residual becomes
interpretable: whatever it still correlates with is a real axis the features do not span. Each
candidate is something we can name and could build:

  peptide length / size      is the error a size effect we never normalised
  label uncertainty          how far this complex's own cluster-mates disagree, i.e. how much of
                             the residual is the assay rather than us
  affinity extremity         the tails we already know we fail
  receptor identity          a per-receptor baseline -- the wall we measured at 75% of variance
                             on the Coventry grid, reappearing here
  feature-space isolation    distance to the nearest training complex: is the error just
                             extrapolation

The distinction that matters: an axis that is PREDICTABLE from things we have is a modelling
failure we can fix, while one that tracks label uncertainty is noise we should stop chasing. The
report separates them instead of listing correlations.

Usage: scorer_nonlinear_upgrade.py
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
CORPUS = ROOT / "data/e432/corpus.npz"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    d = np.load(CORPUS, allow_pickle=True)
    m = d["has_feats"].astype(bool) & (d["source"] == "pdbbind")
    X = np.nan_to_num(d["feats"][m], posinf=0, neginf=0)
    y = d["y"][m].astype(float)
    g = d["groups"][m]
    seqs = d["seq"][m]
    pdbs = d["pdb"][m]
    X = X[:, X.std(0) > 1e-8]

    print("PART 1 — THE UPGRADE (matched clustered CV, identical features)\n")
    print(f"  {'model':<32}{'r':>8}{'MAE':>8}{'RMSE':>8}{'slope':>8}{'sd ratio':>10}")
    preds = {}
    for name, mk in (("linear (what we ship)", lambda: RidgeCV(alphas=np.logspace(-3, 4, 30))),
                     ("random forest", lambda: RandomForestRegressor(
                         n_estimators=600, min_samples_leaf=3, n_jobs=-1, random_state=0))):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            p[te] = mk().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        preds[name] = p
        slope = np.polyfit(y, p, 1)[0]
        print(f"  {name:<32}{corr(p, y):>8.3f}{np.abs(p - y).mean():>8.3f}"
              f"{np.sqrt(((p - y) ** 2).mean()):>8.3f}{slope:>8.3f}{p.std() / y.std():>10.2f}")
    print(f"  {'predict the mean':<32}{'-':>8}{np.abs(y - y.mean()).mean():>8.3f}"
          f"{y.std():>8.3f}{'0.000':>8}{'0.00':>10}")
    lin, rf = preds["linear (what we ship)"], preds["random forest"]
    print(f"\n  MAE {np.abs(lin - y).mean():.3f} -> {np.abs(rf - y).mean():.3f} "
          f"({100 * (1 - np.abs(rf - y).mean() / np.abs(lin - y).mean()):.1f}% better)")
    print(f"  calibration slope {np.polyfit(y, lin, 1)[0]:.3f} -> "
          f"{np.polyfit(y, rf, 1)[0]:.3f}   (1.0 = no shrinkage)")
    print(f"  beats predict-the-mean by "
          f"{100 * (1 - np.abs(rf - y).mean() / np.abs(y - y.mean()).mean()):.1f}% "
          f"(was {100 * (1 - np.abs(lin - y).mean() / np.abs(y - y.mean()).mean()):.1f}%)")

    sc = StandardScaler().fit(X)
    final = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, n_jobs=-1,
                                  random_state=0).fit(sc.transform(X), y)
    joblib.dump({"scaler": sc, "model": final, "n_train": len(y),
                 "cv_r": corr(rf, y), "cv_mae": float(np.abs(rf - y).mean())},
                ROOT / "data/affinity_nonlinear.joblib")
    print(f"  saved data/affinity_nonlinear.joblib")

    # ------------------------------------------------------------------ PART 2
    print("\n\nPART 2 — WHAT AXIS DOES THE RESIDUAL STILL LIVE ON?\n")
    res = rf - y
    ares = np.abs(res)

    clus = defaultdict(list)
    for gi, yi in zip(g, y):
        clus[gi].append(yi)
    unc = np.array([np.std(clus[gi]) if len(clus[gi]) >= 3 else np.nan for gi in g])

    Z = sc.transform(X)
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(Z).query(Z, k=2)
    iso = dd[:, 1]
    L = np.array([len(str(s)) for s in seqs], float)
    ext = np.abs(y - np.median(y))

    print(f"  {'candidate axis':<44}{'r vs |residual|':>17}")
    rows = [("peptide length", L), ("affinity extremity |dG - median|", ext),
            ("isolation in feature space", iso),
            ("label uncertainty (own cluster sd)", unc)]
    for nm, v in rows:
        print(f"  {nm:<44}{corr(v, ares):>17.3f}")

    # receptor baseline: how much of the residual is a per-receptor constant?
    byrec = defaultdict(list)
    for p4, r_ in zip([str(x)[:4] for x in pdbs], res):
        byrec[p4].append(r_)
    multi = {k: v for k, v in byrec.items() if len(v) >= 3}
    if multi:
        within = np.mean([np.var(v) for v in multi.values()])
        total = np.var([x for v in multi.values() for x in v])
        print(f"\n  receptors with 3+ complexes: {len(multi)}")
        print(f"  share of residual variance that is a per-receptor CONSTANT: "
              f"{100 * (1 - within / total):.0f}%")
        print("     (a high value means the model is systematically off per target -- the same"
              "\n      receptor-baseline wall we measured at ~75% on the Coventry grid)")

    print("\n  IS THE REMAINING ERROR FIXABLE OR IS IT NOISE?")
    ok = np.isfinite(unc)
    if ok.sum() > 40:
        print(f"    on the {int(ok.sum())} complexes whose cluster has 3+ members:")
        print(f"      median |residual|              {np.median(ares[ok]):.3f} kcal/mol")
        print(f"      median label uncertainty       {np.median(unc[ok]):.3f} kcal/mol")
        ratio = np.median(ares[ok]) / max(np.median(unc[ok]), 1e-9)
        print(f"      ratio                          {ratio:.2f}x")
        print("      -> " + ("most of what is left is ASSAY NOISE; chasing it is chasing the "
                             "measurement." if ratio < 1.3 else
                             "our error is well above the label noise, so there IS structure "
                             "left to model."))


if __name__ == "__main__":
    main()
