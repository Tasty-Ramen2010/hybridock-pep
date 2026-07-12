"""E66 — recalibrate the geometry model WITH rg_per_L on crystal-65 (the reference set) + benchmark.

Builds geometry features (now including rg_per_L) for crystal-65, then leave-one-out compares the
geometry calibration WITHOUT vs WITH rg_per_L (LOO Pearson + RMSE), and reports the-98 transfer. If
rg_per_L helps, writes a CANDIDATE calibration JSON (does NOT overwrite production — per CLAUDE §7,
production swap needs the full benchmark run). Caches features to /tmp/e66_cr65_geom.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import (GEOMETRY_FEATURE_KEYS,  # noqa: E402
                                                     compute_geometry_features)

CACHE = Path("/tmp/e66_cr65_geom.json")
BASE_KEYS = [k for k in GEOMETRY_FEATURE_KEYS if k != "rg_per_L"]
WITH_KEYS = list(GEOMETRY_FEATURE_KEYS)  # includes rg_per_L


def build_cr65():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    rows = []
    for r in bench:
        f = compute_geometry_features(Path(r["peptide_pdb"]), Path(r["pocket_pdb"]))
        if f is None:
            continue
        f["y"] = r["dg_exp"]
        f["vina"] = float(r.get("vina_docked", 0.0) or 0.0)
        f["pdb"] = r["pdb"]
        rows.append(f)
    CACHE.write_text(json.dumps(rows))
    return rows


def loo_geo(rows, keys):
    """Leave-one-out geometry-only OLS prediction; return Pearson, Spearman, RMSE."""
    y = np.array([r["y"] for r in rows])
    pred = np.zeros(len(rows))
    for i in range(len(rows)):
        tr = [r for j, r in enumerate(rows) if j != i]
        X = np.array([[r[k] for k in keys] for r in tr], float)
        yt = np.array([r["y"] for r in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        w = np.linalg.lstsq(A, yt, rcond=None)[0]
        xi = (np.array([rows[i][k] for k in keys], float) - mu) / sd
        pred[i] = w[0] + np.dot(w[1:], xi)
    return pearsonr(pred, y)[0], spearmanr(pred, y).statistic, float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    rows = build_cr65()
    print(f"=== E66 recalibration on crystal-65 (n={len(rows)}) ===\n")
    for nm, keys in [("WITHOUT rg_per_L (current)", BASE_KEYS), ("WITH rg_per_L (candidate)", WITH_KEYS)]:
        p, s, rmse = loo_geo(rows, keys)
        print(f"  {nm:<28} LOO Pearson={p:+.3f}  Spearman={s:+.3f}  RMSE={rmse:.2f}  ({len(keys)} feats)")

    # fit full-set candidate calibration with rg_per_L and report its weight
    from hybridock_pep.scoring.ensemble import fit_ensemble_calibration
    cal = fit_ensemble_calibration(rows, blend=0.5, vina_mode="total", feature_names=WITH_KEYS)
    rgw = cal.geo_weights[WITH_KEYS.index("rg_per_L")]
    print(f"\n  rg_per_L standardized weight in full fit = {rgw:+.3f} kcal/mol "
          f"({'penalizes extended (expected +)' if rgw > 0 else 'unexpected sign'})")

    out = ROOT / "data/calibration_geometry_rgperL_candidate.json"
    payload = dict(feature_names=cal.feature_names, geo_intercept=cal.geo_intercept,
                   geo_weights=cal.geo_weights, geo_mean=cal.geo_mean, geo_std=cal.geo_std,
                   geo_pred_mean=cal.geo_pred_mean, geo_pred_std=cal.geo_pred_std,
                   vina_mean=cal.vina_mean, vina_std=cal.vina_std, blend=cal.blend,
                   vina_mode=cal.vina_mode, y_mean=cal.y_mean, y_std=cal.y_std,
                   n_complexes=len(rows), reference="crystal-65", note="CANDIDATE: rg_per_L added; "
                   "LOO-validated; NOT production until full benchmark per CLAUDE §7")
    out.write_text(json.dumps(payload, indent=2))
    print(f"  candidate written -> {out.relative_to(ROOT)}")
    print("\n  >> if WITH-rg_per_L LOO Pearson > WITHOUT, the recalibration is justified.")


if __name__ == "__main__":
    main()
