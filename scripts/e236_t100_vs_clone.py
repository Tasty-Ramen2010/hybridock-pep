"""E236 — our FULL model vs the PPI-clone on T100, anchored to PPI-Affinity's published 0.554.
Three comparable numbers on the SAME complexes (all trained on PDBbind-925 MINUS T100, no leakage):
  OURS-FULL  = 16 structural + ProtDCal(seq) + charge-compl + length     (our method, our data)
  PPI-CLONE  = ProtDCal-3D intra-peptide descriptors (their feature class) (their method, our data)
  PPI real   = 0.554 published                                            (their method, their data)
Gap(ours-clone)=feature/method edge;  Gap(clone-0.554)=their training-data edge. Breakdown by length x charge.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _charge_complementarity  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

PPI_PUBLISHED = 0.554
STRUCT = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
          "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def ourfeat(r):
    s = [float(r.get(k, 0) or 0) for k in STRUCT]
    seq = r["seq"]; pn = float(r.get("poc_net", 0) or 0)
    return s + _protdcal_descriptors(seq) + _charge_complementarity(seq, pn) + [float(len(seq))]


def qnet(seq):
    return sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16"


def main():
    t100 = [json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")]
    t100ids = {r["pdb"].lower() for r in t100}
    print(f"=== T100 with full features: n={len(t100)} ===")

    # training: 925 minus T100
    pdb = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl") if json.loads(l)["pdb"].lower() not in t100ids]
    Xo = np.nan_to_num([ourfeat(r) for r in pdb]); yo = np.array([r["y"] for r in pdb])
    ours = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
            l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xo, yo)

    # clone training (ProtDCal-3D) on 925 minus T100
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl")
            if json.loads(l).get("desc") and json.loads(l)["pdb"].lower() not in t100ids]
    Xc = np.nan_to_num([b["desc"] for b in base]); yc = np.array([b["y"] for b in base])
    clone = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                      ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, yc)

    # predict T100 — ours on all; clone where a peptide extract exists
    yv = np.array([float(r["y"]) for r in t100]); Lv = np.array([len(r["seq"]) for r in t100])
    qv = np.array([qnet(r["seq"]) for r in t100])
    po = ours.predict(np.nan_to_num([ourfeat(r) for r in t100]))
    pc = np.full(len(t100), np.nan)
    for i, r in enumerate(t100):
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{r['pdb'].lower()}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is not None:
            pc[i] = clone.predict(np.nan_to_num([e179.descriptors(res, 6.0, 3)]))[0]

    ro, mo = rmae(po, yv); rc, mc = rmae(pc[~np.isnan(pc)], yv[~np.isnan(pc)])
    print(f"\n  OURS-FULL   r={ro:+.3f}  MAE={mo:.2f}   (n={len(yv)})")
    print(f"  PPI-CLONE   r={rc:+.3f}  MAE={mc:.2f}   (n={int((~np.isnan(pc)).sum())}, their method on our data)")
    print(f"  PPI real (published)  r={PPI_PUBLISHED:.3f}")
    # ours on the clone-subset (fair pairing)
    msk = ~np.isnan(pc)
    ro_s, mo_s = rmae(po[msk], yv[msk])
    print(f"  [ours on clone's n={int(msk.sum())} subset]  r={ro_s:+.3f}  MAE={mo_s:.2f}")

    print("\n  by LENGTH (ours):")
    for b in ("short≤8", "med9-12", "long13-16"):
        m = np.array([band(int(L)) == b for L in Lv])
        if m.sum() >= 5:
            r2, m2 = rmae(po[m], yv[m]); rc2, _ = rmae(pc[m & msk], yv[m & msk]) if (m & msk).sum() >= 5 else (float('nan'), 0)
            print(f"    {b:<10} n={int(m.sum()):<3} ours r={r2:+.3f} MAE={m2:.2f}   clone r={rc2:+.3f}")
    print("  by CHARGE (ours):")
    for nm, m in [("charged|q|≥2", np.abs(qv) >= 2), ("neutral|q|≤1", np.abs(qv) <= 1)]:
        if m.sum() >= 5:
            r2, m2 = rmae(po[m], yv[m]); rc2, _ = rmae(pc[m & msk], yv[m & msk]) if (m & msk).sum() >= 5 else (float('nan'), 0)
            print(f"    {nm:<12} n={int(m.sum()):<3} ours r={r2:+.3f} MAE={m2:.2f}   clone r={rc2:+.3f}")


if __name__ == "__main__":
    main()
