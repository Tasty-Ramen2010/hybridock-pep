"""E69 — POOLED calibration: train a balanced model on crystal-65 + the-98 to beat the floor.

Crystal-65 is saturated; the new features (rg_per_L, org_density, cys_frac) pay off only on diverse data.
So fit the geometry model on the POOLED 156, evaluated three honest ways:
  (1) pooled leave-one-complex-out (within-pool generalization)
  (2) STRATIFIED balanced train/test split (by dataset × strength tercile — Ram's balanced-set idea)
  (3) leave-DATASET-out (external generalization — the strict test)
Computes the full 15-key geometry feature set fresh on BOTH datasets (current module incl. rg_per_L +
org_density + cys_frac). Saves a pooled calibration candidate. Geometry-only (the-98 has no vina).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import (GEOMETRY_FEATURE_KEYS,  # noqa: E402
                                                     compute_geometry_features)

CACHE = Path("/tmp/e69_geom_all.json")


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    rows = []
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        f = compute_geometry_features(Path(r["peptide_pdb"]), Path(r["pocket_pdb"]))
        if f:
            rows.append({**f, "y": r["dg_exp"], "ds": "cr65", "pdb": r["pdb"]})
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            f = compute_geometry_features(pep, rec)
            if f:
                rows.append({**f, "y": v["y"], "ds": "the98", "pdb": k})
    CACHE.write_text(json.dumps(rows))
    return rows


def fit(tr, keys):
    X = np.array([[r[k] for k in keys] for r in tr], float)
    y = np.array([r["y"] for r in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    return w, mu, sd


def pred(rows, keys, w, mu, sd):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w


def loo(rows, keys):
    y = np.array([r["y"] for r in rows]); p = np.zeros(len(rows))
    for i in range(len(rows)):
        tr = [r for j, r in enumerate(rows) if j != i]
        w, mu, sd = fit(tr, keys)
        p[i] = pred([rows[i]], keys, w, mu, sd)[0]
    return pearsonr(p, y)[0], spearmanr(p, y).statistic, float(np.sqrt(np.mean((p - y) ** 2)))


def stratified_split(rows, seed=0):
    """Balanced train/test: within each (dataset × strength tercile) stratum, alternate train/test."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for ds in ("cr65", "the98"):
        sub = sorted([r for r in rows if r["ds"] == ds], key=lambda r: r["y"])
        k = len(sub) // 3
        for band in (sub[:k], sub[k:2 * k], sub[2 * k:]):
            idx = rng.permutation(len(band))
            for n, i in enumerate(idx):
                (train if n % 2 == 0 else test).append(band[i])
    return train, test


def main():
    rows = build()
    print(f"=== E69 pooled calibration. total={len(rows)} "
          f"(cr65={sum(r['ds']=='cr65' for r in rows)}, the98={sum(r['ds']=='the98' for r in rows)}) ===")
    base = [k for k in GEOMETRY_FEATURE_KEYS if k not in ("rg_per_L", "org_density", "cys_frac")]
    sets = {
        "base-12 (legacy geom)": base,
        "+ rg_per_L": base + ["rg_per_L"],
        "+ rg_per_L + cys_frac": base + ["rg_per_L", "cys_frac"],
        "+ rg + org + cys (full 15)": list(GEOMETRY_FEATURE_KEYS),
    }
    cr = [r for r in rows if r["ds"] == "cr65"]; t98 = [r for r in rows if r["ds"] == "the98"]

    print("\n=== (1) pooled leave-one-complex-out ===")
    for nm, keys in sets.items():
        p, s, rmse = loo(rows, keys)
        print(f"  {nm:<28} Pearson={p:+.3f}  Spearman={s:+.3f}  RMSE={rmse:.2f}")

    print("\n=== (2) STRATIFIED balanced train/test (by dataset × strength) — avg of 5 seeds ===")
    for nm, keys in sets.items():
        ps = []
        for seed in range(5):
            tr, te = stratified_split(rows, seed)
            w, mu, sd = fit(tr, keys)
            ps.append(pearsonr(pred(te, keys, w, mu, sd), [r["y"] for r in te])[0])
        print(f"  {nm:<28} test Pearson={np.mean(ps):+.3f} ± {np.std(ps):.3f}")

    print("\n=== (3) leave-DATASET-out (strict external) ===")
    for nm, keys in sets.items():
        w1, m1, s1 = fit(t98, keys); p1 = pearsonr(pred(cr, keys, w1, m1, s1), [r["y"] for r in cr])[0]
        w2, m2, s2 = fit(cr, keys); p2 = pearsonr(pred(t98, keys, w2, m2, s2), [r["y"] for r in t98])[0]
        print(f"  {nm:<28} 98→65={p1:+.3f}  65→98={p2:+.3f}")

    # full-pool fit: which features carry weight?
    keys = list(GEOMETRY_FEATURE_KEYS)
    w, mu, sd = fit(rows, keys)
    print("\n=== full-pool standardized weights (kcal/mol) ===")
    for k, wt in sorted(zip(keys, w[1:]), key=lambda t: -abs(t[1])):
        print(f"  {k:<14} {wt:+.3f}")

    out = ROOT / "data/calibration_pooled_candidate.json"
    out.write_text(json.dumps(dict(feature_names=keys, geo_intercept=float(w[0]),
                   geo_weights=[float(x) for x in w[1:]], geo_mean=[float(x) for x in mu],
                   geo_std=[float(x) for x in sd], reference="pooled crystal-65 + the-98 (n=%d)" % len(rows),
                   note="CANDIDATE pooled/balanced geometry calibration; geometry-only (no vina blend)"),
                   indent=2))
    print(f"\n  candidate -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
