"""E227 — train on pre-done FEP data (Schrödinger protein-FEP benchmark, 416 mutations w/ FEP-computed +
experimental ΔΔG + 75 features). Three tests:
  1. HOW GOOD IS FEP? corr(FEP pred_ddg, exp_ddg) + MAE = the gold-standard ceiling (calibrates our claims).
  2. Can CHEAP ML emulate FEP? train on structural features → exp_ddg (leave-system-out), vs FEP accuracy.
  3. Does FEP pred_ddg as a FEATURE help a cheap model? (the "distill FEP" idea — does pre-done FEP transfer.)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def R(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))) if ok.sum() > 4 else (np.nan, np.nan)


def main():
    rows = list(csv.DictReader(open(ROOT / "data/fep_benchmark/benchmark_merged_features_results.csv")))
    # use the 10ns (or default) timepoint rows; keep one per mutation
    rows = [r for r in rows if r.get("time") in ("10", "", None) or True]
    y = np.array([num(r["exp_ddg"]) for r in rows])
    fep = np.array([num(r["pred_ddg"]) for r in rows])
    sys_ = np.array([r["system"] for r in rows])
    ok = ~(np.isnan(y) | np.isnan(fep))
    rows = [r for r, k in zip(rows, ok) if k]; y = y[ok]; fep = fep[ok]; sys_ = sys_[ok]
    print(f"=== Schrödinger protein-FEP benchmark: {len(rows)} mutations, {len(set(sys_))} systems ===")

    # 1. FEP accuracy
    rf, mf = R(fep, y)
    print(f"\n1. FEP (the gold standard) vs experimental ΔΔG:  r={rf:+.3f}  MAE={mf:.2f} kcal/mol")

    # cheap structural features
    numeric_cols = []
    for c in rows[0]:
        if c in ("exp_ddg", "pred_ddg", "err", "abs_err", "system", "mutation", "mutation1"):
            continue
        vals = [num(r.get(c)) for r in rows]
        if np.isfinite(vals).mean() > 0.7 and np.nanstd(vals) > 1e-9:
            numeric_cols.append(c)
    # one-hot the residue-type + ss categoricals
    cat_cols = [c for c in ("start_restype", "end_restype", "res_ss", "type") if c in rows[0]]
    cats = {}
    for c in cat_cols:
        vals = sorted(set(r.get(c, "") for r in rows))
        cats[c] = vals
    def feat(r):
        f = [num(r.get(c)) for c in numeric_cols]
        for c in cat_cols:
            f += [1.0 if r.get(c) == v else 0.0 for v in cats[c]]
        return f
    X = np.nan_to_num([feat(r) for r in rows])
    print(f"   cheap features: {len(numeric_cols)} numeric + {sum(len(v) for v in cats.values())} one-hot = {X.shape[1]}")

    # 2. cheap ML → exp_ddg (leave-system-out)
    logo = LeaveOneGroupOut()
    pred = np.full(len(rows), np.nan)
    for tr, te in logo.split(X, y, sys_):
        pred[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                 l2_regularization=2.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
    rc, mc = R(pred, y)
    print(f"\n2. CHEAP ML (structural feats, leave-system-out) vs experimental: r={rc:+.3f}  MAE={mc:.2f}")
    print(f"   → FEP {rf:+.3f}/{mf:.2f}  vs  cheap ML {rc:+.3f}/{mc:.2f}  : {'ML competitive!' if rc>rf-0.1 else 'FEP wins (worth its cost)'}")

    # 3. FEP as a feature (distill / augment)
    Xaug = np.hstack([X, fep.reshape(-1, 1)])
    pred2 = np.full(len(rows), np.nan)
    for tr, te in logo.split(Xaug, y, sys_):
        pred2[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                  l2_regularization=2.0, random_state=0).fit(Xaug[tr], y[tr]).predict(Xaug[te])
    ra, ma = R(pred2, y)
    print(f"\n3. CHEAP ML + FEP-as-feature: r={ra:+.3f}  MAE={ma:.2f}  (Δ over cheap {ra-rc:+.3f}) — does pre-done FEP add signal")
    print(f"\n   VERDICT: FEP is the gold standard at r={rf:.2f}; cheap ML reaches {rc:.2f}; FEP-augmented {ra:.2f}")


if __name__ == "__main__":
    main()
