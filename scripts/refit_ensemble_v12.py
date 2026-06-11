"""Refit the production ensemble calibration WITH the MJ per-contact feature, using the
src geometry_features module on real rank-1 RAPiDock poses (deployment-consistent).

Writes data/ensemble_calibration.json. Reports LOO r / RMSE (kcal/mol).
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
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES, fit_ensemble_calibration  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
vr = json.loads(Path("/tmp/e22_vina_real.json").read_text())

rows = []
for pdb, meta in bench.items():
    if pdb not in vr:
        continue
    pose = ROOT / f"logs/crystal65_n100/cr_{pdb}/poses/pose_0.pdb"
    rec = ROOT / meta["pocket_pdb"]
    if not pose.exists():
        continue
    f = compute_geometry_features(pose, rec)
    if not f:
        continue
    rows.append(dict(f, y=meta["dg_exp"], vina=vr[pdb]["vina_total"]))

y = np.array([r["y"] for r in rows])
X = np.array([[r[f] for f in GEOMETRY_FEATURES] for r in rows])
v = np.array([r["vina"] for r in rows])


def loo_blend(blend):
    pg = np.zeros(len(y)); pv = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pg[i] = np.r_[1, (X[i] - mu) / sd] @ w
        a, b = np.polyfit(v[tr], y[tr], 1); pv[i] = a * v[i] + b
    blendp = blend * pg + (1 - blend) * pv
    return pearsonr(blendp, y).statistic, float(np.sqrt(((blendp - y) ** 2).mean())), pg


print(f"n={len(rows)} real rank-1 poses with geometry+MJ + Vina  (kcal/mol RMSE)")
best = None
for blend in (0.4, 0.5, 0.6, 0.7):
    r, rmse, pg = loo_blend(blend)
    geo_r = pearsonr(pg, y).statistic
    print(f"  blend={blend} (geo+MJ {geo_r:+.3f}) ensemble r={r:+.3f} RMSE={rmse:.2f}")
    if best is None or r > best[0]:
        best = (r, rmse, blend)
print(f"  >> best blend={best[2]} r={best[0]:+.3f} RMSE={best[1]:.2f}")

cal = fit_ensemble_calibration([dict(r, vina=r["vina"]) for r in rows],
                               blend=best[2], vina_mode="total")
out = ROOT / "data/ensemble_calibration.json"
cal.save(out)
print(f"saved {out} (blend={best[2]}, {len(GEOMETRY_FEATURES)} geo feats incl mj_contact)")
