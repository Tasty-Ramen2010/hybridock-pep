"""Leakage-free per-family vs single-ridge evaluation (the honest verdict).

The shipped per-family ridges (data/calibration_per_family.json) were fit on all
240 holdout complexes — which include every benchmark complex — so scoring that
JSON directly is data leakage. This script instead does a proper leave-one-out:
for each held-out complex it refits BOTH a single global ridge and the complex's
family ridge on everything *except* that complex, then compares.

Feature set matches the production per-family fit: [vina, n_contact, s_ss].
Family assignment is sequence-based (label-independent), so reusing the cluster
membership is not label leakage; only the ridge weights are refit per fold.
Families with < MIN_FAM members after holdout fall back to the global ridge
(this is exactly the production similarity-gate behaviour).

Evaluates on two subsets: the clean Kd+Ki set (n≈101) and the crystal benchmark
subset (n≈65). Reports Pearson/Spearman/RMSE for single vs per-family.

Usage:  python scripts/eval_per_family.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("vina", "n_contact", "s_ss_weighted")  # eval_holdout keys
MIN_FAM = 4  # need ≥4 family members (after holdout) to trust a 3-feature ridge


def _feat(r: dict) -> list[float]:
    return [float(r.get("vina") or 0), float(r.get("n_contact") or 0),
            float(r.get("s_ss_weighted") or 0)]


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = alpha * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def _predict(coef: np.ndarray, x: list[float]) -> float:
    return float(np.dot(coef[:-1], x) + coef[-1])


def main() -> None:
    holdout = json.loads((ROOT / "data" / "eval_holdout_calibrations.json").read_text())
    holdout = [r for r in holdout if r.get("vina") is not None and r.get("dg_exp") is not None]
    fam = json.loads((ROOT / "data" / "calibration_per_family.json").read_text())
    membership = {k.lower(): v for k, v in fam["membership_assignments"].items()}

    pool = []  # full 240-pool rows with features, label, family
    for r in holdout:
        pid = (r.get("pdb") or "").lower()
        pool.append({"pdb": pid, "x": _feat(r), "y": float(r["dg_exp"]),
                     "fam": membership.get(pid)})
    X_all = np.array([p["x"] for p in pool]); y_all = np.array([p["y"] for p in pool])

    # eval subsets
    clean_ids = {r["pdb"].lower() for r in json.loads((ROOT / "data" / "eval_kd_ki_clean.json").read_text())}
    crystal_ids = {r["pdb"].lower() for r in json.loads((ROOT / "data" / "benchmark_crystal.json").read_text())}

    def evaluate(subset_ids: set[str], label: str) -> None:
        idx_eval = [i for i, p in enumerate(pool) if p["pdb"] in subset_ids]
        single_pred, family_pred, ys, routed = [], [], [], 0
        for i in idx_eval:
            mask = np.arange(len(pool)) != i
            # single global ridge, LOO
            g = _fit_ridge(X_all[mask], y_all[mask])
            single_pred.append(_predict(g, pool[i]["x"]))
            # per-family ridge, LOO within the complex's family
            fam_id = pool[i]["fam"]
            members = [j for j in range(len(pool)) if j != i and pool[j]["fam"] == fam_id]
            if fam_id is not None and len(members) >= MIN_FAM:
                fc = _fit_ridge(X_all[members], y_all[members])
                family_pred.append(_predict(fc, pool[i]["x"]))
                routed += 1
            else:
                family_pred.append(_predict(g, pool[i]["x"]))  # fallback = global
            ys.append(pool[i]["y"])
        ys = np.array(ys); sp = np.array(single_pred); fp = np.array(family_pred)

        def stats(pred):
            return (pearsonr(pred, ys).statistic, spearmanr(pred, ys).statistic,
                    float(np.sqrt(np.mean((pred - ys) ** 2))))
        sr, srho, srmse = stats(sp)
        fr, frho, frmse = stats(fp)
        print(f"\n=== {label}  (n={len(ys)}, {routed} routed to a real family, rest fell back) ===")
        print(f"  {'model':14s} {'r':>7s} {'rho':>7s} {'RMSE':>7s}")
        print(f"  {'single-ridge':14s} {sr:+7.3f} {srho:+7.3f} {srmse:7.2f}")
        print(f"  {'per-family':14s} {fr:+7.3f} {frho:+7.3f} {frmse:7.2f}")
        print(f"  Δ(per-family − single): r {fr-sr:+.3f}   "
              f"→ {'per-family WINS' if fr > sr + 0.02 else 'no real gain' if abs(fr-sr) <= 0.02 else 'per-family WORSE'}")

    print("Leakage-free LOO: families + global ridge refit excluding each held-out complex.")
    evaluate(clean_ids, "Clean Kd+Ki")
    evaluate(crystal_ids, "Crystal benchmark subset")


if __name__ == "__main__":
    main()
