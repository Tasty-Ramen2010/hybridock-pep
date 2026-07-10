"""E375 — WHY did dual-training (PDBbind + PPIKB) fail to help? Forensic decomposition.

Tests four candidate causes, across both feature families (16 physics/STRUCT geometry + 37 ProtDCal-3D):
  (1) LABEL-SCALE mismatch — do the two DBs put ΔG on the same scale (mean/std)?
  (2) COVARIATE SHIFT — can a classifier tell PDBbind from PPIKB from features alone? (AUC→1 = different regions)
  (3) CONCEPT DRIFT — does each feature relate to ΔG the SAME way in both DBs? (sign flips = they teach opposite lessons)
  (4) TRANSFER — train on one DB, predict the other (leave-dataset-out). ~0 transfer = disjoint signal.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e375_dual_train_forensics.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS  # 16 STRUCT

ROOT = Path(__file__).resolve().parents[1]
GBR = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)


def load():
    pdb = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()]
    d3_pdb = {json.loads(l)["pdb"]: json.loads(l) for l in (ROOT / "data/e180_protdcal3d.jsonl").read_text().splitlines() if l.strip()}
    ppi = [r for r in (json.loads(l) for l in (ROOT / "data/ppikb_kd_clean.jsonl").read_text().splitlines() if l.strip()) if "poc_n" in r]
    return pdb, d3_pdb, ppi


def block(name, Xp, yp, Xk, yk):
    print(f"\n########## FEATURE FAMILY: {name} ##########")
    # (1) label scale
    print(f"(1) label scale:  PDBbind ΔG mean={yp.mean():+.2f} std={yp.std():.2f}   "
          f"PPIKB mean={yk.mean():+.2f} std={yk.std():.2f}")
    # (2) covariate shift — domain classifier
    X = np.vstack([Xp, Xk]); dom = np.r_[np.zeros(len(Xp)), np.ones(len(Xk))]
    auc = cross_val_score(GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
                          X, dom, cv=5, scoring="roc_auc").mean()
    print(f"(2) covariate shift: domain-classifier AUC={auc:.3f}  "
          f"({'SEVERE — features live in different regions' if auc > 0.8 else 'mild' if auc > 0.65 else 'low'})")
    # (3) concept drift — per-feature corr with y in each DB, count sign flips
    keys = FEATS if name.startswith("physics") else [f"d{i}" for i in range(Xp.shape[1])]
    flips = []
    for i, k in enumerate(keys):
        rp = pearsonr(Xp[:, i], yp)[0]; rk = pearsonr(Xk[:, i], yk)[0]
        if abs(rp) > 0.08 and abs(rk) > 0.08 and np.sign(rp) != np.sign(rk):
            flips.append((k, rp, rk))
    print(f"(3) concept drift: {len(flips)} feature(s) with SIGN-FLIPPED corr(feat,ΔG) between DBs "
          f"(feature means opposite things):")
    for k, rp, rk in flips[:6]:
        print(f"      {k:14s} PDBbind r={rp:+.2f}   PPIKB r={rk:+.2f}")
    # (4) transfer — train one, predict other
    mp = GBR().fit(Xp, yp); mk = GBR().fit(Xk, yk)
    tp = pearsonr(mp.predict(Xk), yk)[0]   # PDBbind-trained → PPIKB
    tk = pearsonr(mk.predict(Xp), yp)[0]   # PPIKB-trained → PDBbind
    print(f"(4) transfer:  PDBbind-trained → PPIKB r={tp:+.3f}   |   PPIKB-trained → PDBbind r={tk:+.3f}")
    print(f"    (compare: each DB's OWN in-fold r ~0.3; low transfer ⇒ the two DBs teach different mappings)")


def main():
    pdb, d3_pdb, ppi = load()
    yp = np.array([float(r["y"]) for r in pdb])
    yk = np.array([float(r["y"]) for r in ppi])

    # physics/STRUCT (16) — both have them
    Xp_s = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in pdb], float))
    Xk_s = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in ppi], float))
    block("physics / STRUCT geometry (16)", Xp_s, yp, Xk_s, yk)

    # ProtDCal-3D (37) — PDBbind 'desc' vs PPIKB 'desc3d'
    DIM = 37
    pdb_d = [(r, d3_pdb[r["pdb"]]) for r in pdb if r["pdb"] in d3_pdb and len(d3_pdb[r["pdb"]].get("desc") or []) == DIM]
    ppi_d = [r for r in ppi if r.get("desc3d") and len(r["desc3d"]) == DIM]
    Xp_d = np.nan_to_num(np.array([d["desc"] for _, d in pdb_d], float)); yp_d = np.array([float(r["y"]) for r, _ in pdb_d])
    Xk_d = np.nan_to_num(np.array([r["desc3d"] for r in ppi_d], float)); yk_d = np.array([float(r["y"]) for r in ppi_d])
    block("ProtDCal-3D descriptors (37)", Xp_d, yp_d, Xk_d, yk_d)

    print("\n=== SUMMARY ===")
    print("  Dual-training helps only if the two DBs share BOTH the feature→ΔG mapping AND the ΔG scale.")
    print("  Read the four rows above: high domain-AUC = covariate shift; sign-flips = concept drift;")
    print("  low cross-DB transfer = the extra data teaches a mapping that doesn't apply to PDBbind.")


if __name__ == "__main__":
    main()
