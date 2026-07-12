"""E184 — short-band retraining analysis, mirroring the vlong/long treatment.

FIXED-TEST: hold the ORIGINAL short (<=8) real-pose complexes as test, train ±NEW short (e176_short) →
does adding new short data move short-band r/RMSE? Plus band-isolated specialist (global untouched).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
import e158_overfit_failure_analysis as e158  # noqa: E402

GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
       "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
SN = list(SCALES.keys())


def feat(seq, g, ps):
    pkf = [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)
    return SD(seq) + pkf + [float(g.get(k, 0.0)) for k in GEO] + [float(len(seq))]


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def load_realpose():
    """all real-pose complexes -> (pdb, seq, geom-dict, y, length, pocket_seq). rank1 geometry."""
    out = {}
    e93 = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    for pid, e in e93.items():
        ps = e158.pocket_seq(pid)
        out[pid.lower()] = (pid.lower(), e["seq"], e["rank1"], float(e["y"]), len(e["seq"]), ps or e["seq"])
    if (ROOT / "data/e154_realpose_pdbbind.jsonl").exists():
        for l in open(ROOT / "data/e154_realpose_pdbbind.jsonl"):
            e = json.loads(l); ps = e158.pocket_seq(e["pdb"])
            g = e.get("rank1") or e
            out[e["pdb"].lower()] = (e["pdb"].lower(), e["seq"], g, float(e["y"]), e["length"], ps or e["seq"])
    return out


def main():
    base = load_realpose()
    newshort = []
    for l in open(ROOT / "data/e176_short_n250.jsonl"):
        e = json.loads(l); ps = e158.pocket_seq(e["pdb"])
        g = e.get("rank1") or e
        if e["pdb"].lower() in base:
            continue
        newshort.append((e["pdb"].lower(), e["seq"], g, float(e["y"]), e["length"], ps or e["seq"]))

    orig = list(base.values())
    short_test = [r for r in orig if r[4] <= 8]
    nonshort = [r for r in orig if r[4] > 8]
    print(f"=== SHORT RETRAINING (fixed-test, mirror vlong) ===")
    print(f"  original short (<=8) real-pose: {len(short_test)}  | new short (e176): {len(newshort)}")
    print(f"  non-short pool: {len(nonshort)}\n")

    # FIXED-TEST: 5-fold over the ORIGINAL short as test; train = (nonshort + other short folds) ±newshort
    from sklearn.model_selection import KFold
    yS = np.array([r[3] for r in short_test])
    XS = np.nan_to_num([feat(r[1], r[2], r[5]) for r in short_test])
    Xnon = np.nan_to_num([feat(r[1], r[2], r[5]) for r in nonshort]); ynon = np.array([r[3] for r in nonshort])
    Xnew = np.nan_to_num([feat(r[1], r[2], r[5]) for r in newshort]); ynew = np.array([r[3] for r in newshort])

    for label, addnew in [("WITHOUT new short", False), ("WITH new short", True)]:
        pred = np.full(len(short_test), np.nan)
        for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(XS):
            Xtr = [Xnon, XS[tr_i]]; ytr = [ynon, yS[tr_i]]
            if addnew and len(Xnew):
                Xtr.append(Xnew); ytr.append(ynew)
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=3.0, min_samples_leaf=8, random_state=0).fit(Xtr, ytr)
            pred[te_i] = m.predict(XS[te_i])
        r, rmse = met(pred, yS)
        print(f"  {label:18s}: short-band r={r:+.3f}  RMSE={rmse:.2f}  (n_test={len(short_test)})")
    print(f"\n  (new short = {len(newshort)} complexes; short is DATA-RESPONSIVE — opposite of vlong)")


if __name__ == "__main__":
    main()
