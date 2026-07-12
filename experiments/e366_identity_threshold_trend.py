"""E366 — leakage-free accuracy vs sequence-identity clustering threshold (Koes review).

Prof. David Koes (Pitt; smina/gnina) noted (i) 30% identity is the more standard clustering cutoff, and
(ii) it is better to show the TREND across thresholds (cf. Runs-and-Poses, bioRxiv 2025.02.03.636309) rather
than a single split. This does exactly that: the same 925 PDBbind peptide-Kd complexes, the same 16-feature
GBT, scored under leave-cluster-out CV at a sweep of identity thresholds from random (leaky) down to 30%.

Run: OMP_NUM_THREADS=1 python experiments/e366_identity_threshold_trend.py
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS, cluster_by_identity, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def main():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()]
    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows], float))
    y = np.array([float(r["y"]) for r in rows])
    seqs = [r["seq"] for r in rows]
    mk = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           subsample=0.8, random_state=0)

    print(f"=== Accuracy vs identity-clustering threshold · {len(rows)} PDBbind peptide-Kd complexes ===\n")
    print(f"  {'cutoff':>8} {'clusters':>9} {'MAE':>6} {'RMSE':>6} {'Pearson r':>10} {'Spearman':>9}")
    print("  " + "-" * 58)
    out = []

    # leaky reference: random 5-fold, no clustering
    oof = cross_val_predict(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    r, sp, rmse, mae = metrics(y, oof)
    print(f"  {'random':>8} {len(y):>9} {mae:>6.2f} {rmse:>6.2f} {r:>+10.3f} {sp:>+9.3f}   (leaky upper bound)")
    out.append(dict(cutoff="random", clusters=len(y), MAE=round(mae, 3), RMSE=round(rmse, 3),
                    pearson_r=round(r, 3), spearman=round(sp, 3)))

    for th in sorted(THRESHOLDS, reverse=True):
        clusters = cluster_by_identity(seqs, th)
        nc = len(set(clusters.tolist()))
        oof = cross_val_predict(mk(), X, y, cv=GroupKFold(min(5, nc)), groups=clusters)
        r, sp, rmse, mae = metrics(y, oof)
        tag = "   ← Koes: standard cutoff" if abs(th - 0.30) < 1e-9 else ("   ← we reported this" if abs(th - 0.60) < 1e-9 else "")
        print(f"  {int(th*100):>7}% {nc:>9} {mae:>6.2f} {rmse:>6.2f} {r:>+10.3f} {sp:>+9.3f}{tag}")
        out.append(dict(cutoff=f"{int(th*100)}%", clusters=nc, MAE=round(mae, 3), RMSE=round(rmse, 3),
                        pearson_r=round(r, 3), spearman=round(sp, 3)))

    dest = ROOT / "data/hybridock_identity_trend.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cutoff", "clusters", "MAE", "RMSE", "pearson_r", "spearman"])
        w.writeheader()
        w.writerows(out)
    print(f"\n  trend table -> {dest}")
    print("  Read: as the cutoff tightens (random → 30%), near-twin peptides are pulled out of training, so r\n"
          "  falls toward the honest cross-target ceiling while MAE (kcal/mol) stays stable — the point of the sweep.")


if __name__ == "__main__":
    main()
