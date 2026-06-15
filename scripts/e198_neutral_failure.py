"""E198 — deep dive: WHICH neutral (|q|<=1) complexes do we fail on? Per-complex error vs complex properties,
to find the failure mode and target a fix. Crystal-925 neutral, clustered-CV production model.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
       "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]
SSK = ["helix", "sheet", "ppii", "turn"]
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        if abs(pq) > 1:
            continue  # NEUTRAL only
        s = ss.get(pid, {})
        rows.append({"pid": pid, "seq": r["seq"], "y": float(r["y"]), "L": r["length"], "pn": float(r["poc_net"]),
                     "geo": {k: float(r.get(k, 0)) for k in GEO}, "ps": ps,
                     "pkf": [float(np.mean([SCALES[s2].get(c, 0) for c in ps])) for s2 in SN],
                     "pep_hyd": float(np.mean([SCALES["hopp"].get(c, 0) for c in r["seq"]])),
                     "pep_arom": float(np.mean([SCALES["arom"].get(c, 0) for c in r["seq"]])),
                     "helix": float(s.get("helix", 0)), "sheet": float(s.get("sheet", 0))})
    print(f"NEUTRAL crystal complexes: n={len(rows)}\n", flush=True)
    y = np.array([r["y"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    X = np.nan_to_num([SD(r["seq"]) + r["pkf"] + [r["geo"][k] for k in GEO] + [float(len(r["seq"]))] for r in rows])
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    err = np.abs(pred - y)
    print(f"overall neutral r={np.corrcoef(pred, y)[0,1]:+.3f}  mean|err|={err.mean():.2f}\n")

    # correlate |error| with complex properties → what makes a neutral complex hard?
    props = {
        "length": np.array([r["L"] for r in rows]),
        "pocket_hyd(hopp)": np.array([r["pkf"][SN.index("hopp")] for r in rows]),
        "pocket_arom": np.array([r["geo"]["poc_f_arom"] for r in rows]),
        "pocket_size(poc_n)": np.array([r["geo"]["poc_n"] for r in rows]),
        "peptide_hyd": np.array([r["pep_hyd"] for r in rows]),
        "peptide_arom": np.array([r["pep_arom"] for r in rows]),
        "helix_frac": np.array([r["helix"] for r in rows]),
        "sheet_frac": np.array([r["sheet"] for r in rows]),
        "bsa_hyd": np.array([r["geo"]["bsa_hyd"] for r in rows]),
        "mean_burial": np.array([r["geo"]["mean_burial"] for r in rows]),
        "affinity_mag|y|": np.abs(y),
        "hyd_MISMATCH": np.abs(np.array([r["pep_hyd"] for r in rows]) - np.array([r["pkf"][SN.index("hopp")] for r in rows])),
    }
    print("=== what correlates with our ERROR on neutral (high = drives failure) ===")
    cors = sorted(((np.corrcoef(v, err)[0, 1], k) for k, v in props.items()), key=lambda x: -abs(x[0]))
    for c, k in cors:
        print(f"  corr(|err|, {k:<20}) = {c:+.3f}")

    # split neutral into sub-bins by the top error-driver, report r per bin
    top = cors[0][1]; tv = props[top]
    print(f"\n=== r within neutral sub-bins by top driver '{top}' ===")
    for lo, hi, nm in [(-1e9, np.percentile(tv, 33), "low"), (np.percentile(tv, 33), np.percentile(tv, 66), "mid"),
                       (np.percentile(tv, 66), 1e9, "high")]:
        mk = (tv >= lo) & (tv < hi)
        if mk.sum() > 5:
            print(f"  {nm:<5} {top} [{lo:.2f},{hi:.2f}) n={mk.sum():<4} r={np.corrcoef(pred[mk], y[mk])[0,1]:+.3f}  mean|err|={err[mk].mean():.2f}")

    # the worst-predicted neutral complexes
    print("\n=== 10 WORST-predicted neutral complexes ===")
    order = np.argsort(-err)[:10]
    for i in order:
        r = rows[i]
        print(f"  {r['pid']} L={r['L']:<3} y={y[i]:+.1f} pred={pred[i]:+.1f} |err|={err[i]:.1f} "
              f"helix={r['helix']:.2f} sheet={r['sheet']:.2f} pep_hyd={r['pep_hyd']:+.2f} pock_hyd={r['pkf'][SN.index('hopp')]:+.2f}")


if __name__ == "__main__":
    main()
