"""Refit the production ensemble calibration WITH strength_bur (SKEMPI experimental strength,
docs E46) added to the geometry block. Uses the same real rank-1 RAPiDock poses (crystal-65,
deployment-consistent) as v12. Since adding a geometry feature does not change the Vina or ΔG
distributions of the panel, the validated Vina blend stats are carried over from the existing
calibration (the Vina-score cache is transient); only the geometry linear block + its prediction
z-stats are refit.

Writes data/ensemble_calibration.json. Reports geometry LOO r before (11 feats) vs after (12).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import (EnsembleCalibration,  # noqa: E402
                                            GEOMETRY_FEATURES)
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
old = EnsembleCalibration.load(ROOT / "data/ensemble_calibration.json")

rows = []
for pdb, meta in bench.items():
    pose = ROOT / f"logs/crystal65_n100/cr_{pdb}/poses/pose_0.pdb"
    rec = ROOT / meta["pocket_pdb"]
    if not pose.exists() or not rec.exists():
        continue
    f = compute_geometry_features(pose, rec)
    if not f:
        continue
    rows.append(dict(f, y=meta["dg_exp"]))

y = np.array([r["y"] for r in rows])
print(f"n={len(rows)} real rank-1 poses (crystal-65)")


def loo_geo(feats):
    X = np.array([[r[f] for f in feats] for r in rows]); p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pearsonr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean()))

old_feats = [f for f in GEOMETRY_FEATURES if f != "strength_bur"]
r0, e0 = loo_geo(old_feats)
r1, e1 = loo_geo(GEOMETRY_FEATURES)
print(f"  geometry LOO  before (11) r={r0:+.3f} RMSE={e0:.2f}")
print(f"  geometry LOO  after  (12) r={r1:+.3f} RMSE={e1:.2f}   (+strength_bur)")

# Refit geometry linear block on the full panel (12 feats); carry Vina/blend from old calib.
X = np.array([[r[f] for f in GEOMETRY_FEATURES] for r in rows])
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
A = np.column_stack([np.ones(len(Z)), Z])
w, *_ = np.linalg.lstsq(A, y, rcond=None)
geo_pred = A @ w

cal = EnsembleCalibration(
    feature_names=list(GEOMETRY_FEATURES),
    geo_intercept=float(w[0]),
    geo_weights=[float(x) for x in w[1:]],
    geo_mean=[float(x) for x in mu],
    geo_std=[float(x) for x in sd],
    geo_pred_mean=float(geo_pred.mean()),
    geo_pred_std=float(geo_pred.std()),
    vina_mean=old.vina_mean,
    vina_std=old.vina_std,
    blend=old.blend,
    vina_mode=old.vina_mode,
    y_mean=float(y.mean()),
    y_std=float(y.std()),
)
out = ROOT / "data/ensemble_calibration.json"
cal.save(out)
print(f"saved {out}  ({len(GEOMETRY_FEATURES)} geo feats incl strength_bur; "
      f"blend={old.blend} carried over)")
