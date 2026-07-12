"""E9c — calibrated ± kcal/mol (RMSE/MAE) for the 60ps MD-LIE/MM-GBSA approach.

Reports BOTH:
  - in-sample calibrated RMSE (fit slope+intercept on all data) — optimistic
  - leave-one-FAMILY-out CV RMSE — the honest blind number
  - mean-predictor baseline (predict the mean ΔG for everyone)
  - within-family RMSE (selectivity regime) for the IE term
so the market comparison is apples-to-apples and not in-sample-inflated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]


def kmer_groups(seqs, th=0.3, k=3):
    ks = [{s[i:i+k] for i in range(max(0, len(s)-k+1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - th).fit_predict(D)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    out = json.loads(Path("/tmp/e9_results.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    for o in out:
        o["seq"] = sm.get(o["pdb"].upper(), "X")
    out = [o for o in out if np.isfinite(o.get("dg_pred", np.nan)) and o["seq"] != "X"]
    y = np.array([o["y"] for o in out])
    g = kmer_groups([o["seq"] for o in out], 0.3)
    print(f"n={len(out)}, families={len(set(g))}")
    print(f"experimental ΔG: range [{y.min():.1f}, {y.max():.1f}], "
          f"mean {y.mean():.2f}, sd {y.std():.2f} kcal/mol")

    # mean-predictor baseline
    print(f"\nBASELINE (predict mean for all): RMSE={rmse(y, np.full_like(y, y.mean())):.2f}  "
          f"MAE={np.mean(np.abs(y-y.mean())):.2f} kcal/mol")

    for key in ("dg_pred", "e_int_mean"):
        x = np.array([o[key] for o in out])
        # in-sample OLS calibration
        A = np.column_stack([np.ones_like(x), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred_in = A @ w
        r_in = pearsonr(x, y).statistic
        # leave-one-family-out CV
        pred_cv = np.zeros_like(y)
        for fam in set(g):
            te = g == fam
            tr = ~te
            At = np.column_stack([np.ones(tr.sum()), x[tr]])
            wt, *_ = np.linalg.lstsq(At, y[tr], rcond=None)
            pred_cv[te] = wt[0] + wt[1] * x[te]
        print(f"\n{key}:")
        print(f"  in-sample calibrated : RMSE={rmse(y,pred_in):.2f}  MAE={np.mean(np.abs(y-pred_in)):.2f}  r={r_in:+.3f}")
        print(f"  leave-family-out CV  : RMSE={rmse(y,pred_cv):.2f}  MAE={np.mean(np.abs(y-pred_cv)):.2f}  r={pearsonr(pred_cv,y).statistic:+.3f}")


if __name__ == "__main__":
    main()
