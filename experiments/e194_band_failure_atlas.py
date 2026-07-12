"""E194 — full failure atlas: WHY PPI wins neutral / long-structured / vlong, and what closes each gap.

On crystal-925 (n=865 with ProtDCal-3D), per SLICE {neutral, charged, long13-16, vlong, structured,
unstructured}, measure clustered-CV r for feature sets:
   OURS (production geom+seq+pocket+charge) · +PD3D (ProtDCal-3D contacts) · PD3D-alone · +SS · +PD3D+SS
to see which addition rescues which slice (even if null overall). Plus per-slice TOP single-feature |r|
discriminators (the feature table) and the vlong-specialist routing test.
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
import e179_protdcal_3d as e179  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
       "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]
SSK = ["helix", "sheet", "ppii", "turn"]
PD3D_NAMES = [f"{w}({p})_{g}_{i}" for (w, p, g, i) in e179.DESCS]


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 5:
        return float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1])


def main():
    p3d = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
           for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")}
    ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid not in p3d:
            continue
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        s = ss.get(pid, {})
        rows.append({"pid": pid, "seq": r["seq"], "y": float(r["y"]), "L": r["length"], "q": abs(pq),
                     "pn": float(r["poc_net"]), "geo": [float(r.get(k, 0)) for k in GEO],
                     "pkf": [float(np.mean([SCALES[s2].get(c, 0) for c in ps])) for s2 in SN],
                     "pd3d": p3d[pid], "ss": [float(s.get(k, 0.0)) for k in SSK], "ps": ps})
    print(f"crystal-925 with all features: n={len(rows)}\n", flush=True)
    y = np.array([r["y"] for r in rows]); L = np.array([r["L"] for r in rows]); q = np.array([r["q"] for r in rows])
    helixsheet = np.array([r["ss"][0] + r["ss"][1] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)

    def base(r):
        pq = r["q"]
        return SD(r["seq"]) + r["pkf"] + r["geo"] + [float(pq), float(L[0]*0+len(r["seq"]))]
    Xbase = np.nan_to_num([SD(r["seq"]) + r["pkf"] + r["geo"] + [r["q"], r["q"]*r["pn"], float(len(r["seq"]))] for r in rows])
    Xpd = np.nan_to_num([r["pd3d"] for r in rows])
    Xss = np.nan_to_num([r["ss"] for r in rows])
    sets = {"OURS": Xbase, "+PD3D": np.hstack([Xbase, Xpd]), "PD3D-only": Xpd,
            "+SS": np.hstack([Xbase, Xss]), "+PD3D+SS": np.hstack([Xbase, Xpd, Xss])}

    def cv(X, mask):
        pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return met(pred[mask], y[mask])

    slices = {
        "ALL": np.ones(len(rows), bool),
        "neutral|q|<=1": q <= 1,
        "charged|q|>=2": q >= 2,
        "long13-16": (L >= 13) & (L <= 16),
        "vlong>=17": L >= 17,
        "structured(h+s>=.5)": helixsheet >= 0.5,
        "unstruct(h+s<.2)": helixsheet < 0.2,
    }
    print("=== per-slice clustered-CV r by feature set (which addition rescues which slice) ===")
    print(f"  {'slice':<20}{'n':>5}  " + "".join(f"{k:>11}" for k in sets))
    for sn_, mask in slices.items():
        if mask.sum() < 15:
            continue
        cells = "".join(f"{cv(X, mask):>+11.3f}" for X in sets.values())
        print(f"  {sn_:<20}{mask.sum():>5}  {cells}")

    # per-slice top single-feature |r| discriminators
    allfeat_names = ([f"seq{i}" for i in range(len(SD(rows[0]['seq'])))] + [f"pkf:{s}" for s in SN]
                     + [f"geo:{g}" for g in GEO] + PD3D_NAMES + ["ss:helix", "ss:sheet", "ss:ppii", "ss:turn"])
    allX = np.nan_to_num([SD(r["seq"]) + r["pkf"] + r["geo"] + list(r["pd3d"]) + r["ss"] for r in rows])
    print("\n=== per-slice TOP-6 single-feature |Pearson r| with affinity (the failure table) ===")
    for sn_, mask in slices.items():
        if mask.sum() < 15:
            continue
        ys = y[mask]; corrs = []
        for j in range(allX.shape[1]):
            xj = allX[mask, j]
            if np.std(xj) < 1e-9:
                continue
            rr = np.corrcoef(xj, ys)[0, 1]
            if not np.isnan(rr):
                corrs.append((abs(rr), rr, allfeat_names[j]))
        corrs.sort(reverse=True)
        top = "  ".join(f"{nm}({rr:+.2f})" for _, rr, nm in corrs[:6])
        print(f"  [{sn_:<18} n={mask.sum():>3}] {top}")


if __name__ == "__main__":
    main()
