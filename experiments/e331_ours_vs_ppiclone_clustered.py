"""E331 — ours vs the PPI-Affinity clone under the SAME leakage-free split.

Answers Ram's "what's our comparison to the PPI clone now?" after E330 showed the
random-CV headline was a redundancy mirage. Here BOTH models are evaluated on the
SAME 865 PDBbind peptide-Kd complexes, with the SAME peptide-identity clusters and
the SAME GroupKFold folds, so the comparison is strictly matched.

  OURS  = 16 structural features (GBT), identical config to E330.
  CLONE = PPI-Affinity's feature class: 37 ProtDCal-3D intra-peptide descriptors
          (data/e180_protdcal3d.jsonl) -> StandardScaler + SVR(rbf), the model E236
          validated as a faithful stand-in for PPI's shipped predictions.

Prints random 5-fold (LEAKY) and peptide-clustered 5-fold (LEAKAGE-FREE) for each,
so the mirage is visible on both sides, not just ours.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Reuse the exact clustering used for the E330 headline number.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import ID_THRESH, cluster_by_identity, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "data/pdbbind_peptides.jsonl"
CLONE = ROOT / "data/e180_protdcal3d.jsonl"
STRUCT = ['poc_n', 'poc_f_hyd', 'poc_f_arom', 'poc_net', 'poc_eis', 'bsa_hyd', 'sasa_hb',
          'sasa_sb', 'arom_cc', 'hb_count', 'mj_contact', 'strength_bur', 'rg_per_L',
          'org_density', 'cys_frac', 'mean_burial']


def new_ours():
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, random_state=0)


def new_clone():
    # PPI-Affinity is a poly/RBF-kernel SVR over ProtDCal descriptors (pose-blind).
    return Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))])


def _report(tag, model, X, y, clusters):
    rand = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    n_folds = min(5, len(set(clusters.tolist())))
    clu = cross_val_predict(model, X, y, cv=GroupKFold(n_folds), groups=clusters)
    rr = metrics(y, rand)
    rc = metrics(y, clu)
    print(f"  {tag}")
    print(f"    random 5-fold  [LEAKY]        : r={rr[0]:+.3f}  rho={rr[1]:+.3f}  RMSE={rr[2]:.2f}  MAE={rr[3]:.2f}")
    print(f"    clustered {n_folds}-fold [LEAKAGE-FREE]: r={rc[0]:+.3f}  rho={rc[1]:+.3f}  RMSE={rc[2]:.2f}  MAE={rc[3]:.2f}")
    return {"random_leaky": {"r": rr[0], "rho": rr[1], "rmse": rr[2], "mae": rr[3]},
            "clustered_leakage_free": {"r": rc[0], "rho": rc[1], "rmse": rc[2], "mae": rc[3]}}


def main():
    ours = {json.loads(l)["pdb"].lower(): json.loads(l)
            for l in OURS.read_text().splitlines() if l.strip()}
    clone = {b["pdb"].lower(): b for b in
             (json.loads(l) for l in CLONE.read_text().splitlines() if l.strip())
             if b.get("desc")}
    ids = sorted(set(ours) & set(clone))  # matched set both can score
    print(f"=== E331 ours vs PPI-clone, matched n={len(ids)} PDBbind peptide-Kd ===\n")

    seqs = [ours[i]["seq"] for i in ids]
    y = np.array([ours[i]["y"] for i in ids], float)
    clusters = cluster_by_identity(seqs, ID_THRESH)
    print(f"  peptide-identity clustering (>= {ID_THRESH:.0%}): "
          f"{len(set(clusters.tolist()))} clusters; y mean={y.mean():.2f} std={y.std():.2f}\n")

    X_ours = np.nan_to_num(np.array([[ours[i][k] for k in STRUCT] for i in ids], float))
    X_clone = np.nan_to_num(np.array([clone[i]["desc"] for i in ids], float))

    m_ours = _report("OURS  (16 structural feats, GBT):", new_ours(), X_ours, y, clusters)
    print()
    m_clone = _report("CLONE (PPI-Affinity ProtDCal-3D, SVR):", new_clone(), X_clone, y, clusters)
    r_ours = m_ours["clustered_leakage_free"]["r"]
    r_clone = m_clone["clustered_leakage_free"]["r"]

    print(f"\n  LEAKAGE-FREE verdict:  ours r={r_ours:+.3f}  vs  PPI-clone r={r_clone:+.3f}  "
          f"(Δ={r_ours - r_clone:+.3f})")
    print("  PPI-Affinity published (their data, their split): r=0.554  [NOT leakage-controlled comparably]")

    receipt = {
        "experiment": "E331 ours vs PPI-Affinity clone, matched leakage-free head-to-head",
        "n_complexes": len(ids), "n_clusters": len(set(clusters.tolist())),
        "id_thresh": ID_THRESH, "matched_ids_file": "data/e331_matched_pdbids.json",
        "ours_16struct_GBT": m_ours,
        "ppi_clone_ProtDCal3d_SVR": m_clone,
        "note": ("PPI-clone = faithful reimplementation of PPI-Affinity's method (ProtDCal-3D desc + StandardScaler "
                 "+ SVR-rbf, model E236) scored on the IDENTICAL 60%%-id clustered split as ours; the real "
                 "PPI-Affinity web server has been down since 2022 so the original cannot be queried. "
                 "The clustered/leakage-free r is the reported number."),
        "reproduce": "OMP_NUM_THREADS=1 python experiments/e331_ours_vs_ppiclone_clustered.py",
    }
    out = ROOT / "data/e331_ours_vs_ppiclone.json"
    out.write_text(json.dumps(receipt, indent=2))
    print(f"\n  receipt -> {out}")


if __name__ == "__main__":
    main()
