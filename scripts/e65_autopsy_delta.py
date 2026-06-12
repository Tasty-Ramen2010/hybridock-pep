"""E65 — how did the over/under-prediction autopsy CHANGE after adding the rg_per_L entropy penalty?

Re-runs the e59 over/under-by-factor breakdown on the-98 with TWO predictors, side by side:
  BASE  = MM-GBSA only (global OLS calib)            [the e59 baseline]
  CORR  = MM-GBSA + α·rg_per_L (global OLS calib)    [entropy-penalty corrected]
Reports per factor the signed error (pred−exp, +ve=under-predicts affinity) in kcal/mol for both, and
the change. Targets the question: did penalizing extendedness fix the strong/weak compression and the
extended-peptide over-rating? Data: /tmp/e63_catalog.json (rg_per_L, mmgbsa, y, all features).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

CAT = json.loads(Path("/tmp/e63_catalog.json").read_text())
rows = [r for r in CAT.values() if r["ds"] == "the98" and not np.isnan(r.get("mmgbsa", np.nan))]


def fit(cols):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    A = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return A @ w


def stats(pred, y, tag):
    rng = pred.std() / y.std()
    print(f"  {tag:<8} Pearson={pearsonr(pred,y)[0]:+.3f}  RMSE={np.sqrt(np.mean((pred-y)**2)):.2f}  "
          f"MAE={np.mean(np.abs(pred-y)):.2f}  within1={np.mean(np.abs(pred-y)<1)*100:.0f}%  "
          f"range={rng*100:.0f}% of real")


def main():
    y = np.array([r["y"] for r in rows])
    base = fit(["mmgbsa"])
    corr = fit(["mmgbsa", "rg_per_L"])
    print(f"=== E65 autopsy delta on the-98 (n={len(rows)}) — BASE vs rg_per_L-CORRECTED ===\n")
    stats(base, y, "BASE")
    stats(corr, y, "CORR")
    print(f"  exp ΔG std={y.std():.2f}  (dynamic-range recovery is the key compression metric)\n")

    med = lambda k: np.nanmedian([r[k] for r in rows])
    print("=== over/under by factor: meanErr(pred−exp) kcal/mol  [+ve = UNDER-predicts affinity] ===")
    print(f"{'factor':<22}{'BASE':>8}{'CORR':>8}{'Δ|err|':>9}   reading")
    facs = [
        ("strength=strong", lambda r: r["y"] < med("y")),
        ("strength=weak", lambda r: r["y"] >= med("y")),
        ("EXTENDED (rg_per_L hi)", lambda r: r["rg_per_L"] > med("rg_per_L")),
        ("COMPACT (rg_per_L lo)", lambda r: r["rg_per_L"] <= med("rg_per_L")),
        ("length>med", lambda r: r["L"] > med("L")),
        ("hydrophobic hi", lambda r: r["hyd_frac"] > med("hyd_frac")),
        ("charged hi", lambda r: r["charged_frac"] > med("charged_frac")),
    ]
    for nm, fn in facs:
        idx = [i for i, r in enumerate(rows) if fn(r)]
        if len(idx) < 4:
            continue
        eb = (base[idx] - y[idx]).mean()
        ec = (corr[idx] - y[idx]).mean()
        dabs = np.abs(corr[idx] - y[idx]).mean() - np.abs(base[idx] - y[idx]).mean()
        tag = "improved" if dabs < -0.05 else ("worse" if dabs > 0.05 else "~same")
        print(f"  {nm:<22}{eb:>+8.2f}{ec:>+8.2f}{dabs:>+9.2f}   {tag}")

    print("\n=== the headline: extended-peptide bias (what the penalty targets) ===")
    hi = [i for i, r in enumerate(rows) if r["rg_per_L"] > med("rg_per_L")]
    print(f"  extended peptides BASE over-rate by {(base[hi]-y[hi]).mean():+.2f} kcal/mol "
          f"-> CORR {(corr[hi]-y[hi]).mean():+.2f} kcal/mol")
    print("  (negative meanErr on extended = MM-GBSA said too strong; penalty should push toward 0)")


if __name__ == "__main__":
    main()
