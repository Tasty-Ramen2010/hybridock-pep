"""E70 — can we LEARN what the MD computes, to skip it? (Ram: record MD, train ML surrogate, save cost).

The expensive MD outputs we cache are: e_int_mean (⟨E_int⟩), e_int_std (fluctuation), minus_tds
(interaction entropy = the −TΔS the MD samples). If cheap sequence/structure features can predict these
out-of-sample, an ML surrogate replaces the MD at ~0 cost. Tests LOO predictability of each MD output,
pooled over crystal-65 + the-98 (the only cross-distribution honest split). Establishes the surrogate's
ceiling and which MD output is learnable vs genuinely needs simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CAT = json.loads(Path("/tmp/e63_catalog.json").read_text())
ORG = json.loads(Path("/tmp/e68_org.json").read_text())
# MD outputs live in e49 caches
M98 = json.loads(Path("/tmp/e49b_the98.json").read_text())
M65 = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text())

CHEAP = ["L", "rg_per_L", "e2e_per_L", "charged_frac", "net_charge", "abs_net_charge", "hyd_frac",
         "arom_frac", "bulky_frac", "pro_frac", "gly_frac", "polar_frac", "gravy",
         "total_bsa", "mean_burial", "n_anchor", "nonbind_frac"]


def assemble():
    rows = []
    for k, v in CAT.items():
        md = (M98.get(k[3:]) if v["ds"] == "the98" else M65.get(k[3:].upper()))
        if md is None or "minus_tds" not in md:
            continue
        o = ORG.get(k, {})
        r = {f: v.get(f, np.nan) for f in CHEAP}
        r.update(ds=v["ds"], cys_frac=o.get("cys_frac", 0.0), org_density=o.get("org_density", 0.0),
                 e_int_mean=md["e_int_mean"], e_int_std=md["e_int_std"], minus_tds=md["minus_tds"])
        rows.append(r)
    return rows


def loo(rows, feats, target):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r[target] for r in rows], float)
    ok = ~(np.isnan(X).any(1) | np.isnan(y))
    X, y = X[ok], y[ok]
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        R = 2.0 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pearsonr(p, y)[0], spearmanr(p, y).statistic, float(np.sqrt(np.mean((p - y) ** 2))), len(y)


def main():
    rows = assemble()
    feats = [f for f in CHEAP if not all(np.isnan(r[f]) for r in rows)] + ["cys_frac", "org_density"]
    print(f"=== E70 MD surrogate feasibility (n={len(rows)}) — LOO predict MD outputs from cheap feats ===\n")
    print(f"{'MD output':<16}{'LOO Pearson':>12}{'Spearman':>10}{'RMSE':>8}   verdict")
    for tgt, unit in [("minus_tds", "kcal/mol (−TΔS)"), ("e_int_std", "kcal/mol (fluctuation)"),
                      ("e_int_mean", "kcal/mol (⟨E_int⟩)")]:
        p, s, rmse, n = loo(rows, feats, tgt)
        verdict = "LEARNABLE — surrogate viable" if p > 0.5 else ("partial" if p > 0.3 else "needs MD")
        print(f"  {tgt:<14}{p:>+12.3f}{s:>+10.3f}{rmse:>8.2f}   {verdict}  [{unit}]")

    # which cheap features drive minus_tds (the entropy)?
    print("\n=== what predicts the MD entropy (−TΔS)? top single-feature correlations ===")
    cors = []
    for f in feats:
        x = np.array([r[f] for r in rows], float); y = np.array([r["minus_tds"] for r in rows], float)
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() > 20 and np.std(x[m]) > 1e-9:
            cors.append((f, spearmanr(x[m], y[m]).statistic))
    for f, c in sorted(cors, key=lambda t: -abs(t[1]))[:8]:
        print(f"  {f:<14} Spearman={c:+.3f}")
    print("\n  >> if minus_tds is LEARNABLE, the entropy term that lifted scoring can be had WITHOUT MD.")


if __name__ == "__main__":
    main()
