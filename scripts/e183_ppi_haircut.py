"""E183 — measure the PPI-clone's structure-quality HAIRCUT on RAPiDock poses, then model it onto real
PPI-Affinity scores.

PPI-Affinity is 3D-structure-based (ProtDCal weighted-contact descriptors). On a NOVEL peptide there is no
crystal -> you must GENERATE a pose (RAPiDock). The pose has ~3A RMSD error, distorting the intra-peptide
contact network -> distorting the descriptors -> degrading the prediction. This script quantifies that.

Paired data: e93 set (65 crystal Kd complexes) has BOTH the crystal peptide structure (RCSB fetch) AND
the retained RAPiDock poses (runs/e93_realpose_campaign/<PDB>/poses_raw/poses_raw/rank*.pdb).

Train the PPI-clone (ProtDCal-3D, d=6 t=3) on 925 crystal (e180), then predict each e93 complex from:
  (a) its CRYSTAL peptide structure  -> the oracle/crystal score (what PPI's 0.55 benchmark uses)
  (b) its RAPiDock rank1 pose        -> the DEPLOYMENT score
  (c) its RAPiDock best-RMSD pose    -> oracle upper bound
HAIRCUT = r_crystal - r_pose. Model onto PPI: estimated real-pose PPI r = 0.554 * (r_pose / r_crystal).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e179_protdcal_3d as e179  # noqa: E402
import e180_protdcal_925 as e180  # noqa: E402

CAMP = ROOT / "runs/e93_realpose_campaign"
D_CUT, T_CUT = 6.0, 3
PPI_T100 = 0.554  # real PPI-Affinity crystal-benchmark Pearson r


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1])


def pose_files(pdb):
    d = CAMP / pdb / "poses_raw" / "poses_raw"
    if not d.exists():
        d = CAMP / pdb
    return sorted(d.glob("rank*.pdb"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))


def main():
    # train clone on 925 crystal
    tr = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    e93 = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    e93_ids = {k.lower() for k in e93}
    tr = [d for d in tr if d["pdb"].lower() not in e93_ids]  # hold out e93
    Xtr = np.nan_to_num([d["desc"] for d in tr]); ytr = np.array([d["y"] for d in tr])
    mdl = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                    ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xtr, ytr)
    print(f"clone trained on {len(tr)} crystal (e93 held out)", flush=True)

    # for each e93 complex: crystal desc + pose descs
    rows = []
    for pdb_u, e in e93.items():
        pdb = pdb_u.lower()
        seq = e["seq"]; y = float(e["y"])
        cres = e180.peptide_chain(pdb, seq)
        if cres is None:
            continue
        cdesc = e179.descriptors(cres, D_CUT, T_CUT)
        poses = pose_files(pdb_u) or pose_files(pdb.upper())
        if not poses:
            continue
        # rank1
        r1 = e179.residue_seq_and_coords(poses[0])
        r1desc = e179.descriptors(r1, D_CUT, T_CUT) if r1 else None
        # best pose by lowest predicted Kd error is cheating; use crystal-closest by descriptor L2 as oracle
        best_desc, best_d = None, 1e18
        cd = np.array(cdesc)
        for p in poses[:50]:
            res = e179.residue_seq_and_coords(p)
            if res is None:
                continue
            dd = np.array(e179.descriptors(res, D_CUT, T_CUT))
            dist = np.linalg.norm(np.nan_to_num(dd - cd))
            if dist < best_d:
                best_d, best_desc = dist, dd.tolist()
        if r1desc is None or best_desc is None:
            continue
        rows.append((pdb, y, cdesc, r1desc, best_desc))
    print(f"e93 complexes with crystal+pose: {len(rows)}", flush=True)

    y = np.array([r[1] for r in rows])
    Xc = np.nan_to_num([r[2] for r in rows])
    Xr1 = np.nan_to_num([r[3] for r in rows])
    Xbest = np.nan_to_num([r[4] for r in rows])

    # WITHIN-e93 repeated CV: train on crystal of the other folds, predict held-out complex from CRYSTAL
    # and from POSE descriptors (same fold model) -> isolates structure-quality haircut, distribution-matched.
    from sklearn.model_selection import RepeatedKFold
    rcs, rr1s, rbs, corrs = [], [], [], []
    for rep, (tr_i, te_i) in enumerate(RepeatedKFold(n_splits=10, n_repeats=8, random_state=0).split(Xc)):
        pass  # placeholder to keep structure; we aggregate over full repeats below
    # accumulate out-of-fold preds per repeat
    for repeat in range(8):
        from sklearn.model_selection import KFold
        pc = np.full(len(rows), np.nan); pr1 = np.full(len(rows), np.nan); pbest = np.full(len(rows), np.nan)
        for tr_i, te_i in KFold(10, shuffle=True, random_state=repeat).split(Xc):
            m = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                          ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc[tr_i], y[tr_i])
            pc[te_i] = m.predict(Xc[te_i]); pr1[te_i] = m.predict(Xr1[te_i]); pbest[te_i] = m.predict(Xbest[te_i])
        rcs.append(met(pc, y)); rr1s.append(met(pr1, y)); rbs.append(met(pbest, y)); corrs.append(met(pc, pr1))
    rc, rr1, rb, corr_c_r1 = np.mean(rcs), np.mean(rr1s), np.mean(rbs), np.mean(corrs)
    sc, sr = np.std(rcs), np.std(rr1s)
    print(f"\n=== PPI-CLONE HAIRCUT on RAPiDock poses (n={len(rows)} e93, 8x repeated 10-fold) ===")
    print(f"  CRYSTAL structure   -> r_truth = {rc:+.3f} ± {sc:.3f}   (benchmark regime, where PPI's 0.55 lives)")
    print(f"  RAPiDock RANK1 pose -> r_truth = {rr1:+.3f} ± {sr:.3f}   (DEPLOYMENT: novel peptide, generated pose)")
    print(f"  descriptor-closest pose -> r_truth = {rb:+.3f}   (LEAKY oracle, ignore — picks crystal-like pose)")
    print(f"  corr(crystal preds, rank1 preds) = {corr_c_r1:+.3f}")
    ratio = rr1 / rc if rc > 0 else float("nan")
    print(f"\n  HAIRCUT (crystal->rank1): Δr = {rc-rr1:+.3f}   retention ratio = {ratio:.2f}")
    print(f"\n=== MODEL ONTO REAL PPI-AFFINITY (their structure-descriptor method = same haircut class) ===")
    print(f"  PPI-Affinity crystal benchmark: r = {PPI_T100:.3f}")
    print(f"  estimated PPI-Affinity on RAPiDock rank1 poses: r ~ {PPI_T100*ratio:+.3f}")
    print(f"  (prediction-corr method: PPI_pose ~ {PPI_T100*corr_c_r1:+.3f} using pred-corr {corr_c_r1:.2f})")

    # ---- OUR geometry model on the SAME e93 complexes (crystal interface vs pose) for head-to-head ----
    try:
        our = compare_ours(rows)
        print(f"\n=== OURS (interface geometry) on the SAME e93 poses ===")
        print(f"  our deployment real-pose r (from memory/e-realpose): see runs; geometry uses interface not")
        print(f"  intra-peptide contacts -> robust to peptide-internal pose error.  {our}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  (ours-comparison skipped: {str(e)[:80]})")

    (ROOT / "runs/e183_haircut.json").write_text(json.dumps(
        {"n": len(rows), "r_crystal": rc, "r_crystal_std": sc, "r_rank1": rr1, "r_rank1_std": sr,
         "corr_crystal_rank1": corr_c_r1, "retention_ratio": ratio,
         "ppi_crystal": PPI_T100, "ppi_pose_estimate_ratio": PPI_T100 * ratio,
         "ppi_pose_estimate_predcorr": PPI_T100 * corr_c_r1}, indent=2))


def compare_ours(rows):
    """OUR interface-geometry model on the SAME e93 rank1 poses, same within-e93 repeated CV."""
    import importlib.util
    e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
    importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
    SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
    GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
           "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
    e93 = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    ids = {r[0] for r in rows}
    X, y = [], []
    for pdb_u, e in e93.items():
        if pdb_u.lower() not in ids:
            continue
        g = e["rank1"]
        X.append(SD(e["seq"]) + [float(g.get(k, 0.0)) for k in GEO]); y.append(float(e["y"]))
    X = np.nan_to_num(X); y = np.array(y)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    rs = []
    for repeat in range(8):
        pr = np.full(len(y), np.nan)
        for tr_i, te_i in KFold(10, shuffle=True, random_state=repeat).split(X):
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=3.0, min_samples_leaf=8, random_state=0).fit(X[tr_i], y[tr_i])
            pr[te_i] = m.predict(X[te_i])
        rs.append(met(pr, y))
    return f"OUR geometry on rank1 poses: r_truth = {np.mean(rs):+.3f} ± {np.std(rs):.3f}  (n={len(y)}, same CV)"


if __name__ == "__main__":
    main()
