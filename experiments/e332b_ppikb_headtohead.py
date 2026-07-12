"""E332b — ours vs PPI-clone on the INDEPENDENT PPIKB set, leakage-free, full feature stack.

Companion to E331 (PDBbind). Confirms the head-to-head win generalizes to a second, different database — and
that PPIKB's higher absolute MAE is its own label noise (IC50/EC50 mix, same-seq disagreement up to 10.8 kcal),
not our scorer. Both models, SAME 60%-identity clusters, SAME GroupKFold folds.

  OURS  = full stack proxy: desc3d (37 ProtDCal-3D) + pocket_pkf (22 pocket/physics) = 59 features, GBT.
  CLONE = PPI-Affinity's feature class: desc3d (37 ProtDCal-3D) only, StandardScaler + SVR(rbf).

Run: OMP_NUM_THREADS=1 python experiments/e332b_ppikb_headtohead.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity, ID_THRESH, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PPIKB = ROOT / "data/ppikb_features.jsonl"


def _run(rows, tag):
    seen = {}
    for r in rows:
        if r.get("desc3d") and r.get("pocket_pkf") and r.get("y") is not None and r.get("seq"):
            seen[r["pdb"]] = r
    rr = list(seen.values())
    if len(rr) < 30:
        print(f"{tag}: only {len(rr)} — skip"); return
    seqs = [r["seq"] for r in rr]
    y = np.array([float(r["y"]) for r in rr])
    X_ours = np.nan_to_num(np.array([r["desc3d"] + r["pocket_pkf"] for r in rr], float))
    X_clone = np.nan_to_num(np.array([r["desc3d"] for r in rr], float))
    clu = cluster_by_identity(seqs, ID_THRESH)
    nc = len(set(clu.tolist()))
    gbt = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    svr = lambda: Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))])
    print(f"\n{tag}: n={len(rr)}, {nc} clusters, y std={y.std():.2f}")
    for nm, mk, X in [("OURS  full(59)", gbt, X_ours), ("CLONE ProtDCal(37)", svr, X_clone)]:
        rnd = cross_val_predict(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0))
        clc = cross_val_predict(mk(), X, y, cv=GroupKFold(min(5, nc)), groups=clu)
        rr_ = metrics(y, rnd); rc_ = metrics(y, clc)
        print(f"  {nm:18s} random[leaky] r={rr_[0]:+.3f} MAE={rr_[3]:.2f} | clustered[leakage-free] r={rc_[0]:+.3f} RMSE={rc_[2]:.2f} MAE={rc_[3]:.2f}")
    print(f"  mean-baseline MAE={np.mean(np.abs(y - y.mean())):.2f}")


def main():
    rows = [json.loads(l) for l in PPIKB.read_text().splitlines() if l.strip()]
    print("=== E332b PPIKB independent-set head-to-head (leakage-free) ===")
    _run(rows, "ALL PPIKB (mixed Kd/Ki/IC50/EC50 — noisy)")
    _run([r for r in rows if r.get("aff_type") in ("Kd", "KD", "Ki")], "Kd/Ki-ONLY (IC50/EC50 removed)")


if __name__ == "__main__":
    main()
