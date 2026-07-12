"""E152 — the "AI haircut": production model on CRYSTAL poses vs REAL RAPiDock poses (same complexes).

Deployment reality: at inference we score RAPiDock-generated poses (~3.4 Å), not crystals. How much r do we
lose? Train the 240-feature production model on PDBbind + the98 (EXCLUDE cr65 → no leakage), then score the
65 cr65 complexes two ways:
  - CRYSTAL pose features (data/benchmark_crystal via pooled cr65 rows)
  - REAL RAPiDock pose features (data/e93_realpose_results.json: rank-1 of N=100, and top-5 ensemble mean)
Haircut = r_crystal − r_real. Report rank-1 and top-5 ensemble (memory: 5-pose ensemble is the sweet spot).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py")
e150 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e150)
PROD, POS, NEG = e150.PROD, e150.POS, e150.NEG


def compl(seq, pn):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return [pq * pn, abs(pq) * abs(pn), abs(pq + pn)]


def fvec(geom: dict, seq: str):
    return ([geom[c] for c in PROD] + e150.seq_descriptors(seq)
            + compl(seq, geom.get("poc_net", 0.0)) + [float(len(seq))])


def R(p, y):
    return float(np.corrcoef(p, y)[0, 1])


def main():
    # POOL = PDBbind + the98 + cr65(crystal). cr65 must be IN-distribution to isolate the pose effect
    # (training only on PDBbind+the98 collapses on cr65 = leave-dataset-out wall, not the haircut).
    pdb_rows, the98_rows, cr65_rows = [], [], []
    for r in [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]:
        pdb_rows.append(({c: r[c] for c in PROD}, r["seq"], r["y"]))
    for nm in ["train", "test"]:
        for r in csv.DictReader(open(ROOT / f"data/pooled_benchmark_{nm}.csv")):
            if not r.get("seq"):
                continue
            rec = ({c: float(r[c]) for c in PROD}, r["seq"], float(r["y"]), r["pdb"].split("_")[0].upper())
            (the98_rows if r["dataset"] == "the98" else cr65_rows).append(rec)
    real = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    real_by = {pid.split("_")[0].upper(): e for pid, e in real.items()}

    # build full design matrix; track which rows are cr65 (with a real-pose counterpart)
    rows = [(g, s, y, None) for (g, s, y) in pdb_rows] + [(g, s, y, None) for (g, s, y, _) in the98_rows]
    cr_idx = []
    for (g, s, y, key) in cr65_rows:
        if key in real_by:
            cr_idx.append(len(rows))
        rows.append((g, s, y, key))
    Xc = np.nan_to_num(np.array([fvec(g, s) for (g, s, y, k) in rows], float))
    y = np.array([r[2] for r in rows], float)
    print(f"=== E152 AI haircut — pooled CV, cr65 in-distribution (n={len(rows)}, cr65∩real={len(cr_idx)}) ===\n")

    # grouped 5-fold CV; for cr65 test rows, ALSO predict with real-pose features
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(rows))
    pc = np.full(len(rows), np.nan); pr1 = np.full(len(rows), np.nan); pr5 = np.full(len(rows), np.nan)
    for f in range(5):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=600, max_depth=3, learning_rate=0.03,
                                          l2_regularization=4.0, min_samples_leaf=18, random_state=0).fit(Xc[tr], y[tr])
        te = np.where(fold == f)[0]
        pc[te] = m.predict(Xc[te])
        for i in te:
            key = rows[i][3]
            if key in real_by:
                e = real_by[key]
                pr1[i] = m.predict(np.nan_to_num(np.array([fvec(e["rank1"], e["seq"])], float)))[0]
                t5 = e.get("top5")
                if isinstance(t5, list) and t5:
                    pr5[i] = float(np.mean([m.predict(np.nan_to_num(np.array([fvec(g, e["seq"])], float)))[0] for g in t5]))
                else:
                    pr5[i] = pr1[i]
    idx = np.array(cr_idx)
    yv = y[idx]
    rc = R(pc[idx], yv)
    print(f"  {'pose source':<26}{'r':>8}{'MAE':>7}  haircut")
    print(f"  {'CRYSTAL (oracle)':<26}{rc:>+8.3f}{np.mean(np.abs(pc[idx]-yv)):>7.2f}   —")
    for lbl, p in [("REAL RAPiDock rank-1", pr1[idx]), ("REAL RAPiDock top-5 ens", pr5[idx])]:
        rr = R(p, yv)
        print(f"  {lbl:<26}{rr:>+8.3f}{np.mean(np.abs(p-yv)):>7.2f}   {rr-rc:+.3f}")
    print("\n  haircut = real − crystal. The 240-feat model leans on POSE-INDEPENDENT sequence descriptors")
    print("  (ProtDCal), so the haircut should be SMALLER than the old geometry-only model (~0.12).")


if __name__ == "__main__":
    main()
