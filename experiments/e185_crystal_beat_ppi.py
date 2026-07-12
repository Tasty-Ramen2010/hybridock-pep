"""E185 — can we BEAT PPI on crystal by ADDING ProtDCal-3D (PPI's own winning feature class) to our model?

Idea: PPI wins crystal-oracle absolute Kd because its ProtDCal weighted-contact descriptors capture
intra-peptide structure our cheap geometry misses. We HAVE that engine now (e179/e180). So fuse:
   OURS (seq ProtDCal + interface geometry + pocket + charge)  ⊕  ProtDCal-3D (contact descriptors)
and see if the union beats either alone on crystal-925, overall + per band + CHARGED.

Honest clustered-CV (greedy_cluster 0.7) so no redundancy mirage. Reports r and RMSE.
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
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG, PROD = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG, e150.PROD
import e158_overfit_failure_analysis as e158  # noqa: E402
SN = list(SCALES.keys())


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def our_feat(seq, r, ps):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq); pn = float(r.get("poc_net", 0))
    pkf = [float(np.mean([SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)
    return SD(seq) + pkf + [float(r.get(c, 0)) for c in PROD] + [pq * pn, abs(pq) * abs(pn), abs(pq + pn), float(len(seq))]


def main():
    # ProtDCal-3D descriptors for the 925 (e180)
    p3d = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
           for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")}
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid not in p3d:
            continue
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        g = {c: float(r.get(c, 0)) for c in PROD}; g["poc_net"] = float(r["poc_net"])
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        rows.append((pid, r["seq"], g, float(r["y"]), r["length"], ps, p3d[pid], abs(pq)))
    print(f"crystal-925 with ProtDCal-3D + ours: n={len(rows)}", flush=True)
    y = np.array([r[3] for r in rows]); L = np.array([r[4] for r in rows]); chg = np.array([r[7] for r in rows])
    grp, _ = e158.greedy_cluster([r[5] for r in rows], 0.7)

    Xour = np.nan_to_num([our_feat(r[1], r[2], r[5]) for r in rows])
    X3d = np.nan_to_num([r[6] for r in rows])
    Xfus = np.hstack([Xour, X3d])

    def cv(X):
        pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    print("\n=== crystal-925 clustered-CV (honest) — beat-PPI test ===")
    print(f"  {'model':<22}{'r':>8}{'RMSE':>8}")
    res = {}
    for name, X in [("OURS (geom+seq+pocket)", Xour), ("ProtDCal-3D alone", X3d), ("FUSION ours+3D", Xfus)]:
        p = cv(X); r, rmse = met(p, y); res[name] = p
        print(f"  {name:<22}{r:>+8.3f}{rmse:>8.2f}")
    print("  PPI-Affinity crystal benchmark ref:  r≈0.55 (T100), our prior crystal best ≈0.52")

    print("\n=== per-band (fusion vs ours) ===")
    bands = {"short<=8": L <= 8, "med9-12": (L >= 9) & (L <= 12), "long13-16": (L >= 13) & (L <= 16), "vlong>=17": L >= 17}
    for bn, mask in bands.items():
        ro, _ = met(res["OURS (geom+seq+pocket)"][mask], y[mask])
        rf, _ = met(res["FUSION ours+3D"][mask], y[mask])
        print(f"  {bn:<12} n={mask.sum():<4} ours={ro:+.3f}  fusion={rf:+.3f}  Δ={rf-ro:+.3f}")

    print("\n=== CHARGED subset (|net charge|>=2) — the known floor ===")
    for thr in (1, 2, 3):
        m = chg >= thr
        ro, _ = met(res["OURS (geom+seq+pocket)"][m], y[m])
        rf, _ = met(res["FUSION ours+3D"][m], y[m])
        print(f"  |q|>={thr} n={m.sum():<4} ours={ro:+.3f}  fusion={rf:+.3f}  Δ={rf-ro:+.3f}")


if __name__ == "__main__":
    main()
