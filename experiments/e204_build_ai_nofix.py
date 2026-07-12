"""E204 — build the SEPARATE AI/deployment model WITHOUT the size-fix (the size-fix helps crystal but hurts
deployment on current data, E-table). Crystal keeps affinity_crystal_sizefix.joblib; AI gets a clean
non-size-fix model trained on all current real RAPiDock poses → data/affinity_ai_nofix.joblib (no size_regs,
so predict_affinity applies no residualisation for it).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e202_band_routing_build as e202  # noqa: E402
from hybridock_pep.scoring.affinity_model import GEOMETRY_KEYS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402

FEATURE_NAMES = (GEOMETRY_KEYS + [f"pd_{i}" for i in range(220)] + ["q_compl", "abs_q_match", "q_neutralize", "length"])


def rmae(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def main():
    rows = e202.load_ai()
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows]); q = np.array([r[4] for r in rows])
    grp, _ = e158.greedy_cluster([r[3] for r in rows], 0.7)
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        pred[te] = e202._hgb().fit(X[tr], y[tr]).predict(X[te])
    print(f"AI non-size-fix model, grouped-CV (n={len(rows)}):")
    print(f"  {'slice':<16}{'n':>5}{'r':>8}{'MAE':>8}")
    for nm, mk in [("OVERALL", np.ones(len(y), bool)), ("short<=8", L <= 8), ("med9-12", (L >= 9) & (L <= 12)),
                   ("long13-16", (L >= 13) & (L <= 16)), ("vlong>=17", L >= 17),
                   ("neutral|q|<=1", q <= 1), ("charged|q|>=2", q >= 2)]:
        if mk.sum() >= 4:
            r, mae = rmae(pred, y, mk)
            print(f"  {nm:<16}{int(mk.sum()):>5}{r:>+8.3f}{mae:>8.2f}")
    cv_r, _ = rmae(pred, y, np.ones(len(y), bool))
    mfull = e202._hgb().fit(X, y)
    out = ROOT / "data/affinity_ai_nofix.joblib"
    joblib.dump({"model": mfull, "feature_order": FEATURE_NAMES, "n_train": len(rows), "cv_r": cv_r,
                 "size_fix": False, "pose_type": "real_rapidock"}, out)
    print(f"  saved {out.name} (NO size_regs)")


if __name__ == "__main__":
    main()
