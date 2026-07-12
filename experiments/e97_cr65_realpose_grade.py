"""E97 — cr65 real-pose DEPLOYMENT affinity grade (r / RMSE), proper LOO.

The deployment question: when the pipeline runs on its OWN real RAPiDock poses (NOT crystal oracle
poses), how well does the production affinity model predict experimental ΔG? This is the honest
deployment number — lower than the crystal-pose benchmark upper bound by construction.

For each cr65 complex (e93 real-pose campaign, 65/65 complete):
  * rank-1   = features of pose_0 (RAPiDock diffusion top pose)
  * top-5    = mean feature vector over poses 0..4 (ensemble; E94: pose-invariant pocket dominates)
Then production 16-feature ridge + length router, leave-one-COMPLEX-out → Pearson r, Spearman ρ, RMSE.
LOO (not in-sample) so the number is not optimistic — the n=32 in-sample 0.58 reported earlier is
expected to settle lower here.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from hybridock_pep.scoring.geometry_features import (  # noqa: E402
    GEOMETRY_FEATURE_KEYS, compute_geometry_features,
)

CAMP = ROOT / "runs" / "e93_realpose_campaign"
PROD = list(GEOMETRY_FEATURE_KEYS)
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]


def loo(rows, cols, lam=1.0, router=True):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        if router and rows[i]["length"] <= 8:
            trb = [rows[j] for j in tr if rows[j]["length"] <= 8]
            if len(trb) < 6:
                trb = [rows[j] for j in tr]  # fall back to full set if short-stratum too thin
                sc = cols
            else:
                sc = SHORT
            Xt = np.array([[r["feat"][c] for c in sc] for r in trb], float)
            yt = np.array([r["y"] for r in trb])
        else:
            sc = cols
            Xt = X[tr]
            yt = y[tr]
        ok = ~np.isnan(Xt).any(1)
        Xt, yt = Xt[ok], yt[ok]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        R = np.eye(A.shape[1]) * lam
        R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        xv = np.array([rows[i]["feat"][c] for c in sc], float)
        pred[i] = np.r_[1.0, (xv - mu) / sd] @ w
    return pred


def stat(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    if m.sum() < 5:
        return np.nan, np.nan, np.nan, int(m.sum())
    return (pearsonr(p[m], y[m])[0], spearmanr(p[m], y[m]).statistic,
            float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), int(m.sum()))


def feats_for(poses, receptor, k):
    """Mean feature dict over the first k poses (k=1 → rank-1)."""
    acc: dict[str, list[float]] = {key: [] for key in PROD}
    for p in poses[:k]:
        f = compute_geometry_features(p, receptor)
        if not f:
            continue
        for key in PROD:
            v = f.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                acc[key].append(float(v))
    out = {key: (float(np.mean(vals)) if vals else np.nan) for key, vals in acc.items()}
    return out if any(not np.isnan(v) for v in out.values()) else None


def main():
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    print(f"=== E97 cr65 real-pose affinity grade ({len(complexes)} complexes with poses) ===\n")

    rows1, rows5 = [], []
    for cx in complexes:
        meta = bench.get(cx)
        if not meta:
            continue
        receptor = Path(meta["pocket_pdb"]).resolve()
        y = meta.get("dg_exp")
        if y is None:
            continue
        poses = sorted((CAMP / cx / "poses").glob("pose_*.pdb"),
                       key=lambda q: int(q.stem.split("_")[1]))
        if not poses:
            continue
        length = int(meta.get("peptide_len") or len(meta.get("peptide_seq", "")))
        f1 = feats_for(poses, receptor, 1)
        f5 = feats_for(poses, receptor, 5)
        if f1:
            rows1.append({"pdb": cx, "y": float(y), "length": length, "feat": f1})
        if f5:
            rows5.append({"pdb": cx, "y": float(y), "length": length, "feat": f5})
        print(f"  {cx}: len={length} y={y:+.2f}  feats ok (rank1={'Y' if f1 else 'N'} top5={'Y' if f5 else 'N'})",
              flush=True)

    print(f"\nscored: rank-1 n={len(rows1)}  top-5 n={len(rows5)}")
    print("\n=== DEPLOYMENT GRADE (production 16-feat ridge + length router, leave-one-complex-out) ===")
    for nm, rows in [("rank-1 (pose_0)", rows1), ("top-5 ensemble", rows5)]:
        if len(rows) < 5:
            print(f"  {nm:<18} insufficient (n={len(rows)})")
            continue
        p = loo(rows, PROD, router=True)
        y = np.array([r["y"] for r in rows])
        r, rho, rmse, n = stat(p, y)
        print(f"  {nm:<18} r={r:+.3f}  ρ={rho:+.3f}  RMSE={rmse:.2f} kcal/mol  (n={n})")
    print("\n  context: crystal oracle-pose benchmark r≈0.585 LOO (upper bound); documented")
    print("  real-pose deployment ≈0.486 (top-5). PPI-Affinity (non-FEP peer) r=0.554.")
    json.dump({"rank1_n": len(rows1), "top5_n": len(rows5)},
              open(ROOT / "runs" / "e97_cr65_realpose_grade.json", "w"))


if __name__ == "__main__":
    main()
