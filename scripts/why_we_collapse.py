#!/usr/bin/env python
"""Why does every model we fit collapse toward the mean? Locate where the information dies.

THE PATTERN, ON TWO UNRELATED BENCHMARKS. Blind-925: slope of prediction on truth 0.135, we
express 42% of the real spread, corr(signed error, truth) = -0.908. Coventry grid: the same
correlation at -0.869, "87% a constant". Four different feature sets on the 925 produce residuals
whose magnitudes correlate 0.81-0.96, and 53% of the failures are in the worst decile of ALL of
them. Adding features changes how big the error is, never who it happens to.

Shrinking toward the mean is not a bug -- it is the least-squares-optimal response to features
that do not discriminate. So "why do we collapse" is really "why don't the features
discriminate", and that has several candidate answers which make different predictions. Each gets
measured here rather than argued.

  1  CAPACITY. Are we underfitting? A flexible model on the same features settles this. If it
     also lands near r = 0.32, capacity is not the constraint and no amount of new parameters --
     or new architecture -- will help.

  2  FEATURE DEGENERACY, the sharpest test available. Find pairs of complexes that are NEAR
     IDENTICAL in feature space and look at how far apart their measured affinities are. Two
     complexes our features cannot tell apart, whose true dG differs by several kcal/mol, are a
     direct measurement of an error floor no model can cross. The median dG gap among nearest
     neighbours IS the irreducible error, and if it is close to our observed MAE then the
     features are simply exhausted.

  3  LABEL NOISE. PDBbind pools assays, labs, temperatures and buffers. Complexes sharing a
     cluster group should have similar affinity; how similar they actually are bounds how well
     anything could score.

  4  CLASS MIXTURE. If the corpus is several physically distinct populations, a feature can mean
     opposite things in each and cancel out in the pooled fit. Fitting within a group and testing
     within it separates this from a global feature failure.

  5  WHICH FEATURES CARRY ANYTHING AT ALL, singly and jointly, to show whether the failure is
     "no signal anywhere" or "signal that does not add up".

Usage: why_we_collapse.py
"""
from __future__ import annotations

import statistics as st
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
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    d = np.load(CORPUS, allow_pickle=True)
    m = d["has_feats"].astype(bool) & (d["source"] == "pdbbind")
    X = np.nan_to_num(d["feats"][m], posinf=0, neginf=0)
    y = d["y"][m].astype(float)
    g = d["groups"][m]
    keep = X.std(0) > 1e-8
    X = X[:, keep]
    print(f"{len(y)} PDBbind peptide complexes, {X.shape[1]} non-constant features, "
          f"label sd {y.std():.3f}, {len(np.unique(g))} clusters\n")

    # ------------------------------------------------------------------ 1. capacity
    print("1. IS IT CAPACITY? Same features, same clustered splits, rising model complexity.\n")
    print(f"   {'model':<40}{'r':>8}{'MAE':>8}{'RMSE':>8}")
    gkf = GroupKFold(n_splits=5)
    results = {}
    for name, mk in (("ridge (linear)", lambda: RidgeCV(alphas=np.logspace(-3, 4, 30))),
                     ("random forest (400 trees)",
                      lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                    n_jobs=-1, random_state=0)),
                     ("gradient boosting (deep)",
                      lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                            max_depth=None, random_state=0))):
        pred = np.zeros(len(y))
        for tr, te in gkf.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            mdl = mk().fit(sc.transform(X[tr]), y[tr])
            pred[te] = mdl.predict(sc.transform(X[te]))
        results[name] = pred
        print(f"   {name:<40}{corr(pred, y):>8.3f}{np.abs(pred - y).mean():>8.3f}"
              f"{np.sqrt(((pred - y) ** 2).mean()):>8.3f}")
    print(f"   {'predict the mean':<40}{'-':>8}{np.abs(y - y.mean()).mean():>8.3f}{y.std():>8.3f}")
    best = max(results, key=lambda k: corr(results[k], y))
    print(f"\n   best is {best} at r = {corr(results[best], y):.3f}. "
          + ("Capacity is NOT the constraint."
             if corr(results[best], y) - corr(results["ridge (linear)"], y) < 0.08
             else "Capacity helps -- we were underfitting."))

    # -------------------------------------------------- 2. feature degeneracy (the sharp one)
    print("\n2. FEATURE DEGENERACY — how different are two complexes our features call identical?\n")
    Z = StandardScaler().fit_transform(X)
    from scipy.spatial import cKDTree
    tree = cKDTree(Z)
    dd, ii = tree.query(Z, k=2)
    nn_d, nn_i = dd[:, 1], ii[:, 1]
    gaps = np.abs(y - y[nn_i])
    close = nn_d <= np.percentile(nn_d, 25)          # the quarter that are most alike
    print(f"   median feature distance to nearest neighbour: {np.median(nn_d):.2f} sd-units")
    print(f"   |dG gap| to that neighbour:  median {np.median(gaps):.2f}, "
          f"mean {gaps.mean():.2f} kcal/mol")
    print(f"   restricted to the 25% MOST similar pairs (n={int(close.sum())}):")
    print(f"       feature distance   median {np.median(nn_d[close]):.2f} sd-units")
    print(f"       |dG gap|           median {np.median(gaps[close]):.2f}, "
          f"mean {gaps[close].mean():.2f} kcal/mol")
    print(f"\n   For comparison: our blind-925 MAE is 1.399 and predicting the mean gives 1.475.")
    irr = gaps[close].mean() / 2
    print(f"   A nearest-neighbour predictor would score MAE {gaps[close].mean():.2f} on those "
          f"pairs;\n   half that gap, {irr:.2f}, is a floor any model sharing these features "
          f"cannot beat.")
    print("   -> " + ("THE FEATURES ARE EXHAUSTED. Complexes they cannot tell apart differ by "
                      "more than our whole error budget."
                      if irr > 0.9 else
                      "the features do separate complexes; the failure is in how we use them."))

    # ------------------------------------------------------------------ 3. label noise
    print("\n3. LABEL NOISE — how consistent are affinities within a cluster?\n")
    within = []
    for gid in np.unique(g):
        v = y[g == gid]
        if len(v) >= 3:
            within.append(float(np.std(v)))
    if within:
        print(f"   {len(within)} clusters with 3+ members")
        print(f"   median within-cluster sd of dG: {st.median(within):.3f} kcal/mol")
        print(f"   overall label sd:               {y.std():.3f} kcal/mol")
        print(f"   -> {100 * st.median(within) / y.std():.0f}% of the total spread lives INSIDE "
              f"clusters of near-identical complexes")

    # ------------------------------------------------------------------ 4. class mixture
    print("\n4. CLASS MIXTURE — does a feature mean different things in different populations?\n")
    sizes = {gid: (g == gid).sum() for gid in np.unique(g)}
    big = [gid for gid, n in sizes.items() if n >= 25]
    if big:
        print(f"   {len(big)} clusters with 25+ members; per-cluster sign of the top feature:")
        j = int(np.argmax([abs(corr(X[:, k], y)) for k in range(X.shape[1])]))
        signs = []
        for gid in big[:10]:
            mm = g == gid
            c = corr(X[mm, j], y[mm])
            signs.append(c)
            print(f"      cluster {str(gid)[:20]:<22} n={int(mm.sum()):<4} r = {c:+.3f}")
        if len(signs) > 2:
            pos = sum(1 for c in signs if c > 0)
            print(f"   the single most predictive feature points POSITIVE in {pos}/{len(signs)} "
                  f"clusters and negative in {len(signs) - pos}")
            print("   -> " + ("SIGN-INVERTING across populations: pooling them cancels the signal."
                              if 0 < pos < len(signs) else
                              "consistent sign; class mixture is not the main problem."))

    # ------------------------------------------------------------------ 5. where signal is
    print("\n5. WHERE IS THE SIGNAL? best single features vs all of them together\n")
    cs = sorted(((abs(corr(X[:, k], y)), k) for k in range(X.shape[1])), reverse=True)
    for c, k in cs[:5]:
        print(f"   feature {k:<4} |r| = {c:.3f}")
    print(f"   all {X.shape[1]} features, clustered CV: r = "
          f"{corr(results['ridge (linear)'], y):.3f}")
    print(f"   best single feature alone:              |r| = {cs[0][0]:.3f}")
    print("   -> " + ("the whole is barely more than its best part: the features are largely "
                      "redundant with one another."
                      if corr(results["ridge (linear)"], y) < cs[0][0] * 1.4 else
                      "combining features does add real information."))


if __name__ == "__main__":
    main()
