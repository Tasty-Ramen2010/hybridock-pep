"""Fit a balanced v1.4 calibration where Vina is the primary signal.

The v1.2 production-pose ridge on PepSet-6 returned w_vina=0 because the
6-complex training set has too little signal-to-noise to distinguish
Vina's contribution from entropy. That's a calibration artefact, not
physics — Vina IS an empirical free-energy estimator calibrated against
PDBbind, and should be the dominant signal.

This script pins w_vina = 1.0 and fits (w_s_ss_weighted, intercept) on
the residual ΔG - Vina = entropy contribution + bias. Compares the
resulting LOO Pearson r against v1.2.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_pepset_features() -> list[dict]:
    """Read the PepSet-6 production-pose features used to fit v1.2."""
    scores_path = ROOT / "data" / "training_scores_production_entropy.json"
    if not scores_path.exists():
        scores_path = ROOT / "data" / "training_scores_production.json"
    scores = json.loads(scores_path.read_text())
    with (ROOT / "data" / "training_complexes.csv").open() as f:
        meta = {r["pdb_id"].lower(): r for r in csv.DictReader(f)}
    rows = []
    for pdb_id, sc in scores.items():
        m = meta.get(pdb_id.lower())
        if not m:
            continue
        pkd = float(m["experimental_pkd"])
        # aggregate over top-K poses already in the JSON
        agg = sc if "vina_score" in sc else sc.get("aggregate", sc)
        if "vina_score" not in agg:
            continue
        rows.append({
            "pdb": pdb_id,
            "pkd": pkd,
            "dg_exp": -1.3633 * pkd,
            "vina": float(agg["vina_score"]),
            "ad4": float(agg.get("ad4_score", 0.0) or 0.0),
            "n_contact": int(agg.get("n_contact", 0) or 0),
            "s_ss_weighted": float(agg.get("s_ss_weighted", 0.0) or 0.0),
        })
    return rows


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y):
    if not x:
        return float("nan")
    return float(math.sqrt(((np.array(x) - np.array(y)) ** 2).mean()))


def fit_pinned_vina(rows: list[dict], w_vina_pinned: float = 1.0,
                    ridge_alpha: float = 0.1) -> dict:
    """Fit (w_s_ss_weighted, intercept) on residual = ΔG_exp − w_vina × Vina.

    Equivalent to OLS with bias term on 1 feature; ridge_alpha is small enough
    that on N=6 it's nearly OLS. Returns the calibration dict.
    """
    if not rows:
        raise ValueError("no training rows")
    vina = np.array([r["vina"] for r in rows])
    ent = np.array([r["s_ss_weighted"] for r in rows])
    y = np.array([r["dg_exp"] for r in rows])
    target = y - w_vina_pinned * vina

    # Solve [target = w_ss * ent + b] via ridge with intercept
    X = ent.reshape(-1, 1)
    # Augment with bias column
    Xb = np.hstack([X, np.ones((len(X), 1))])
    # Ridge: (X^T X + αI)^-1 X^T y
    reg = ridge_alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0  # don't regularize bias
    coef = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ target)
    w_ss, intercept = float(coef[0]), float(coef[1])

    # In-sample r/RMSE on full hybrid
    pred = w_vina_pinned * vina + w_ss * ent + intercept
    in_r = pearson(pred.tolist(), y.tolist())
    in_rmse = rmse(pred.tolist(), y.tolist())

    # LOO
    loo_pred = []
    for i in range(len(rows)):
        mask = np.arange(len(rows)) != i
        vi = vina[mask]; ei = ent[mask]; ti = y[mask] - w_vina_pinned * vi
        Xi = np.hstack([ei.reshape(-1, 1), np.ones((mask.sum(), 1))])
        regi = ridge_alpha * np.eye(2); regi[-1, -1] = 0
        ci = np.linalg.solve(Xi.T @ Xi + regi, Xi.T @ ti)
        loo_pred.append(w_vina_pinned * vina[i] + ci[0] * ent[i] + ci[1])
    loo_r = pearson(loo_pred, y.tolist())
    loo_rmse = rmse(loo_pred, y.tolist())

    return {
        "schema_version": 2,
        "model_type": "ridge",
        "w_vina": w_vina_pinned,
        "w_ad4": 0.0,
        "w_contact": 0.0,
        "w_s_sc": 0.0,
        "w_s_bb": 0.0,
        "w_s_ss_weighted": w_ss,
        "intercept": intercept,
        "ridge_alpha": ridge_alpha,
        "positive_constraint": False,
        "vina_pinned": True,
        "pearson_r": in_r,
        "rmse_kcal_mol": in_rmse,
        "loo_pearson_r": loo_r,
        "loo_rmse_kcal_mol": loo_rmse,
        "n_complexes": len(rows),
        "features_used": ["vina_score (pinned w=1.0)", "s_ss_weighted"],
        "training_csv": "data/training_complexes.csv",
        "scores_json": "data/training_scores_production_entropy.json",
        "notes": (
            "v1.4 balanced: w_vina pinned at 1.0 to make Vina the primary "
            "signal (physically motivated — Vina is already calibrated against "
            "PDBbind). Entropy contributes a smaller residual correction. "
            "Production calibration intended to replace v1.2 where w_vina was "
            "erroneously 0 due to the 6-complex training set's low signal."
        ),
    }


def main() -> None:
    rows = load_pepset_features()
    print(f"Loaded {len(rows)} PepSet-6 complexes")
    for r in rows:
        print(f"  {r['pdb']:8s}  pKd={r['pkd']:.2f}  ΔG_exp={r['dg_exp']:6.2f}  "
              f"Vina={r['vina']:6.2f}  S_ss={r['s_ss_weighted']:5.2f}")

    print(f"\n--- v1.4 balanced fit: w_vina pinned at 1.0 ---")
    cal = fit_pinned_vina(rows, w_vina_pinned=1.0, ridge_alpha=0.1)
    print(f"  w_s_ss_weighted = {cal['w_s_ss_weighted']:+.4f}")
    print(f"  intercept       = {cal['intercept']:+.4f}")
    print(f"  In-sample r     = {cal['pearson_r']:+.3f}   RMSE = {cal['rmse_kcal_mol']:.2f}")
    print(f"  LOO Pearson r   = {cal['loo_pearson_r']:+.3f}   RMSE = {cal['loo_rmse_kcal_mol']:.2f}")

    # Compare to v1.2 (w_vina=0, just entropy)
    print(f"\n--- v1.2 (w_vina=0, entropy-only) for reference ---")
    cal_v12 = json.loads((ROOT / "data" / "calibration_v1_2_production_entropy.json").read_text())
    print(f"  LOO Pearson r   = {cal_v12['loo_pearson_r']:+.3f}   RMSE = {cal_v12['loo_rmse_kcal_mol']:.2f}")

    out = ROOT / "data" / "calibration_v1_4_balanced.json"
    out.write_text(json.dumps(cal, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
