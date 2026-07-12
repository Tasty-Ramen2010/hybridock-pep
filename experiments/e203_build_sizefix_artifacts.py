"""E203 — build the two size-confound-corrected scoring artifacts (individual crystal + AI/deployment).

The forensic audit (E201) found the model over-relies on size (corr(pred,len)=−0.21 vs truth −0.10), which
sabotages vlong. E202 validated that residualising the 7 size-correlated geometry features against peptide
length — applied GLOBALLY — fixes vlong on BOTH crystal (0.072→0.158) and deployment (0.234→0.328) and lifts
short/overall, while band routing (long specialist, vlong geometry-free on poses) HURT. So we ship the
size-fix, not routing.

Artifacts (each = {model, size_regs{idx:[coef,intercept]}, feature_order, n_train, cv_r}):
  data/affinity_crystal_sizefix.joblib   (trained on crystal-925)
  data/affinity_ai_sizefix.joblib        (trained on real RAPiDock poses)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e202_band_routing_build as e202  # noqa: E402
from hybridock_pep.scoring.affinity_model import GEOMETRY_KEYS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402

SIZE_IDX = e202.SIZE_IDX
FEATURE_NAMES = (GEOMETRY_KEYS + [f"pd_{i}" for i in range(220)] + ["q_compl", "abs_q_match", "q_neutralize", "length"])


def fit_regs(X, L):
    regs = {}
    for j in SIZE_IDX:
        lr = LinearRegression().fit(L.reshape(-1, 1), X[:, j])
        regs[j] = [float(lr.coef_[0]), float(lr.intercept_)]
    return regs


def apply_regs(X, L, regs):
    X = X.copy()
    for j, (c, b) in regs.items():
        X[:, j] = X[:, j] - (c * L + b)
    return X


def R(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def build(tag, rows, out):
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
    grp, _ = e158.greedy_cluster([r[3] for r in rows], 0.7)
    # grouped-CV to report the honest number
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        regs = fit_regs(X[tr], L[tr])
        m = e202._hgb().fit(apply_regs(X[tr], L[tr], regs), y[tr])
        pred[te] = m.predict(apply_regs(X[te], L[te], regs))
    cv_r = R(pred, y); cv_mae = float(np.nanmean(np.abs(pred - y)))
    print(f"{tag}: grouped-CV r={cv_r:+.3f} MAE={cv_mae:.2f} (n={len(rows)})  "
          f"vlong={R(pred,y,L>=17):+.3f} short={R(pred,y,L<=8):+.3f}  "
          f"size-conf corr(pred,len)={R(pred,L.astype(float)):+.3f} vs truth {R(y,L.astype(float)):+.3f}")
    # final model on ALL rows, with global size-regs (stored for predict-time)
    regs_full = fit_regs(X, L)
    mfull = e202._hgb().fit(apply_regs(X, L, regs_full), y)
    joblib.dump({"model": mfull, "size_regs": {int(k): v for k, v in regs_full.items()},
                 "feature_order": FEATURE_NAMES, "n_train": len(rows), "cv_r": cv_r,
                 "size_fix": True, "pose_type": tag}, out)
    print(f"  saved {out.name}")


def main():
    build("crystal", e202.load_crystal(), ROOT / "data/affinity_crystal_sizefix.joblib")
    build("ai_realpose", e202.load_ai(), ROOT / "data/affinity_ai_sizefix.joblib")


if __name__ == "__main__":
    main()
