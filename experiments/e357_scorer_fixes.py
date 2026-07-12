"""E357 — apply fixes 1 (preorganization/entropy feature), 3 (regime flags), 4 (de-shrink calibration), 5 (ensemble)
and measure the combined absolute-Kd gain, leakage-free.

Fix 1 (preorg/entropy): cyclic/disulfide detection, proline, glycine-flexibility, N/C-term — the entropy-of-
preorganization signal that keeps surfacing (best single feature is cys_frac; cyclic peptides bind tighter by
reduced entropy cost). Sequence-derivable, cheap.
Fix 3 (regime): flags for MHC-groove-like (len 8-11) and membrane-hydrophobic peptides — a "different regime" the
general scorer can't model; used as features AND a low-confidence flag.
Fix 4 (de-shrink): nested leave-one-out linear calibration to correct the compression (slope 0.79) at the extremes.
Fix 5 (ensemble): blend the GBT with a ridge model (bag of different inductive biases) — cheap consensus.

Run: OMP_NUM_THREADS=1 python experiments/e357_scorer_fixes.py
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
        "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
HYDRO = set("AILVMFWG")


def preorg_features(seq):
    s = seq.upper(); L = max(len(s), 1)
    ncys = s.count("C")
    return [
        ncys / L,                       # cysteine fraction (already-ish, but explicit)
        1.0 if ncys >= 2 else 0.0,      # disulfide/cyclic-capable (2+ Cys)
        s.count("P") / L,               # proline (cis/trans + backbone rigidity)
        s.count("G") / L,               # glycine (flexibility — entropy penalty)
        (s.count("P") + ncys) / L,      # combined preorganization
        1.0 / L,                        # terminal-charge weight (short = termini matter more)
    ]


def regime_features(seq):
    s = seq.upper(); L = max(len(s), 1)
    hyd = sum(s.count(a) for a in HYDRO) / L
    mhc_like = 1.0 if 8 <= len(s) <= 11 else 0.0        # MHC-I/II groove length window
    membrane_like = 1.0 if hyd > 0.55 else 0.0          # very hydrophobic → membrane/entropy-driven regime
    return [hyd, mhc_like, membrane_like]


def loo_pred(X, y, groups, model_fn, k=8):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(X, y, groups):
        m = model_fn(); m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
    return oof


def loo_calibrated(X, y, groups, model_fn, k=8):
    """Nested LOO: within each outer fold, fit a de-shrink linear map on the training OOF, apply to test."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(k)
    for tr, te in gkf.split(X, y, groups):
        # inner OOF on the training set to fit calibration
        inner = np.full(len(tr), np.nan); gtr = groups[tr]
        kk = min(5, len(np.unique(gtr)))
        for itr, ite in GroupKFold(kk).split(X[tr], y[tr], gtr):
            m = model_fn(); m.fit(X[tr][itr], y[tr][itr]); inner[ite] = m.predict(X[tr][ite])
        a, b = np.polyfit(inner, y[tr], 1)               # de-shrink map from training only
        m = model_fn(); m.fit(X[tr], y[tr])
        oof[te] = a * m.predict(X[te]) + b
    return oof


def gbt():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def rep(tag, p, y):
    print(f"  {tag:34s} r={pearsonr(p,y)[0]:+.3f}  MAE={np.mean(np.abs(p-y)):.3f}  RMSE={np.sqrt(np.mean((p-y)**2)):.3f}")


def main():
    rows = [json.loads(l) for l in open("data/pdbbind_peptides.jsonl")]
    seqs = [r["seq"] for r in rows]
    y = np.array([float(r["y"]) for r in rows])
    g = np.array([hash(s[:4]) % 100000 for s in seqs])
    Xbase = np.array([[float(r[f]) for f in FULL] for r in rows])
    Xpre = np.array([preorg_features(s) for s in seqs])
    Xreg = np.array([regime_features(s) for s in seqs])
    print(f"=== E357 scorer fixes (n={len(y)}, leakage-free GroupKFold by seq-prefix) ===")
    rep("baseline (16 feats)", loo_pred(Xbase, y, g, gbt), y)
    X1 = np.hstack([Xbase, Xpre])
    rep("+fix1 preorg/entropy", loo_pred(X1, y, g, gbt), y)
    X13 = np.hstack([Xbase, Xpre, Xreg])
    rep("+fix1+3 preorg+regime", loo_pred(X13, y, g, gbt), y)
    rep("+fix1+3+4 calibrated", loo_calibrated(X13, y, g, gbt), y)
    # fix 5: ensemble GBT + ridge
    def ens():
        class E:
            def fit(s, X, Y):
                s.a = gbt().fit(X, Y); s.b = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(X, Y); return s
            def predict(s, X): return 0.6 * s.a.predict(X) + 0.4 * s.b.predict(X)
        return E()
    rep("+fix1+3+5 ensemble", loo_pred(X13, y, g, ens), y)
    rep("+ALL (1+3+4+5)", loo_calibrated(X13, y, g, ens), y)
    print("\n(calibration is ~r-invariant by construction; watch MAE/RMSE for the extreme-bias fix. r gain = fixes 1/3/5.)")


if __name__ == "__main__":
    main()
