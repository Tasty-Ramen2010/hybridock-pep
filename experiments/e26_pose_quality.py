"""E26 — does pose QUALITY drive our affinity accuracy? (Ram: are we scoring good poses?)

We scored rank1/top5 (RAPiDock-ranked, median ~3.4Å). RAPiDock mis-ranks: the best pose
(median 1.6Å) exists but is buried. Recompute geometry+MJ on three pose choices and LOO:
  rank1     : pose_0 (deployment default, ~3.4Å) — what we reported
  top5_mean : mean features over poses 0-4 (~3.6Å)
  bestrmsd  : the closest-to-crystal pose (oracle pose SELECTION; upper bound if ranking worked)
Also: correlation of per-complex prediction error with rank1 pose RMSD (pose-quality confound).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

GEN = ROOT / "logs/crystal65_n100"
bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
res = json.loads((GEN / "benchmark_results.json").read_text())


def rmsds(cname):
    return (res.get(cname, {}).get("pretrained", {}) or {}).get("ref_rmsds") or []


def feats_mean(pdb, meta, idxs):
    rec = ROOT / meta["pocket_pdb"]
    fs = []
    for i in idxs:
        pose = GEN / f"cr_{pdb}" / "poses" / f"pose_{i}.pdb"
        if pose.exists():
            f = compute_geometry_features(pose, rec)
            if f:
                fs.append(f)
    if not fs:
        return None
    return {k: float(np.mean([f[k] for f in fs])) for k in fs[0]}


def loo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return p


def main():
    data = {"rank1": [], "top5": [], "best": []}
    ys = {"rank1": [], "top5": [], "best": []}
    rank1_rmsd = []
    for pdb, meta in bench.items():
        cname = f"cr_{pdb}"
        rm = rmsds(cname)
        if not rm or not (GEN / cname / "poses").exists():
            continue
        bi = int(np.argmin(rm))
        f1 = feats_mean(pdb, meta, [0])
        f5 = feats_mean(pdb, meta, list(range(5)))
        fb = feats_mean(pdb, meta, [bi])
        if f1:
            data["rank1"].append(f1); ys["rank1"].append(meta["dg_exp"]); rank1_rmsd.append(rm[0])
        if f5:
            data["top5"].append(f5); ys["top5"].append(meta["dg_exp"])
        if fb:
            data["best"].append(fb); ys["best"].append(meta["dg_exp"])
    print("=== geometry+MJ LOO by pose choice (kcal/mol RMSE) ===")
    preds = {}
    for k in ["rank1", "top5", "best"]:
        y = np.array(ys[k])
        X = np.array([[r.get(f, 0.0) for f in GEOMETRY_FEATURES] for r in data[k]])
        p = loo(X, y); preds[k] = (p, y)
        rmse = np.sqrt(((p - y) ** 2).mean())
        print(f"  {k:<10} n={len(y)}  r={pearsonr(p,y).statistic:+.3f}  RMSE={rmse:.2f}")
    # pose-quality confound: does rank1 error grow with rank1 RMSD?
    p, y = preds["rank1"]
    err = np.abs(p - y)
    rr = np.array(rank1_rmsd[:len(err)])
    print(f"\n  corr(|rank1 pred error|, rank1 pose RMSD) = {pearsonr(err, rr).statistic:+.3f}")
    print("  (positive => worse poses give worse affinity => pose ranking is headroom)")
    print(f"\n  pose RMSD scored: rank1 median {np.median(rr):.1f}Å vs best-of-100 median "
          f"{np.median([min(rmsds('cr_'+p)) for p in bench if rmsds('cr_'+p)]):.1f}Å")


if __name__ == "__main__":
    main()
