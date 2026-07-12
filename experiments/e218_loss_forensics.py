"""E218 — deep loss forensics for the three problem bands (≤12, vlong, charged): WHY we lose, by how much,
on which exact peptides, with full feature correlation. Production 262-feat model, crystal-925 clustered-CV
(most data, all features). For each band: r/MAE, feature-vs-error drivers, top-10 worst-predicted peptides.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, SIZE_IDX, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())
GEO = GEOMETRY_KEYS
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}


def vec(r, ps):
    g = {k: float(r.get(k, 0.0)) for k in GEO}; g["pocket_seq"] = ps
    x = build_feature_vector(g, r["seq"])
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        s = ss.get(pid, {})
        pq = sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"])
        rows.append({"pid": pid, "seq": r["seq"], "y": float(r["y"]), "L": r["length"], "q": abs(pq),
                     "x": vec(r, ps), "ps": ps, "pn": float(r["poc_net"]),
                     "pep_hyd": float(np.mean([_SCALES["kd"].get(c, 0) for c in r["seq"]])),
                     "pock_hyd": float(np.mean([_SCALES["kd"].get(c, 0) for c in ps])),
                     "pep_chg": float(np.mean([_SCALES["charge"].get(c, 0) for c in r["seq"]])),
                     "helix": float(s.get("helix", 0)), "sheet": float(s.get("sheet", 0)),
                     "n_pos": sum(c in "KR" for c in r["seq"]), "n_neg": sum(c in "DE" for c in r["seq"])})
    X = np.nan_to_num([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    L = np.array([r["L"] for r in rows]); q = np.array([r["q"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        regs = {j: LinearRegression().fit(L[tr].reshape(-1, 1), X[tr][:, j]) for j in SIZE_IDX}
        Xtr = X[tr].copy(); Xte = X[te].copy()
        for j, lr in regs.items():
            Xtr[:, j] -= lr.predict(L[tr].reshape(-1, 1)); Xte[:, j] -= lr.predict(L[te].reshape(-1, 1))
        pred[te] = e202._hgb().fit(Xtr, y[tr]).predict(Xte)
    err = np.abs(pred - y)

    # candidate explanatory properties
    props = {
        "|affinity y|": np.abs(y), "length": L.astype(float), "net_charge|q|": q.astype(float),
        "pep_hyd": np.array([r["pep_hyd"] for r in rows]), "pock_hyd": np.array([r["pock_hyd"] for r in rows]),
        "hyd_mismatch": np.array([abs(r["pep_hyd"] - r["pock_hyd"]) for r in rows]),
        "n_pos": np.array([r["n_pos"] for r in rows]), "n_neg": np.array([r["n_neg"] for r in rows]),
        "helix": np.array([r["helix"] for r in rows]), "sheet": np.array([r["sheet"] for r in rows]),
        "poc_net": np.array([r["pn"] for r in rows]),
        "charge_clash(pq*pn>0)": np.array([1.0 if (r["q"] * 0 + (r["n_pos"] - r["n_neg"])) * r["pn"] > 0 else 0.0 for r in rows]),
    }

    def report(name, mask):
        ry = y[mask]; rp = pred[mask]; re = err[mask]
        r = np.corrcoef(rp, ry)[0, 1] if mask.sum() > 4 else float("nan")
        print(f"\n===== {name}  (n={mask.sum()}) =====")
        print(f"  r={r:+.3f}  MAE={re.mean():.2f}  (label std={ry.std():.2f}, mean|y|={np.abs(ry).mean():.2f})")
        cors = sorted(((np.corrcoef(v[mask], re)[0, 1], k) for k, v in props.items()
                       if np.std(v[mask]) > 1e-9), key=lambda x: -abs(x[0]))
        print("  error-drivers corr(|err|, prop):  " + "  ".join(f"{k}={c:+.2f}" for c, k in cors[:5]))
        order = np.argsort(-re)[:10]
        idx = np.where(mask)[0]
        print("  worst-predicted:")
        for i in order:
            g = rows[idx[i]]
            print(f"    {g['pid']} L={g['L']:<3} q={g['n_pos']-g['n_neg']:+d} y={ry[i]:+.1f} pred={rp[i]:+.1f} "
                  f"|err|={re[i]:.1f}  pepHyd={g['pep_hyd']:+.2f} pockHyd={g['pock_hyd']:+.2f} hel={g['helix']:.2f} sht={g['sheet']:.2f}")

    print(f"OVERALL crystal-925: r={np.corrcoef(pred,y)[0,1]:+.3f} MAE={err.mean():.2f}")
    report("BAND ≤12", L <= 12)
    report("  sub: short ≤8", L <= 8)
    report("  sub: med 9-12", (L >= 9) & (L <= 12))
    report("BAND vlong ≥17", L >= 17)
    report("BAND charged |q|≥2", q >= 2)
    report("  sub: charged ≤12", (q >= 2) & (L <= 12))


if __name__ == "__main__":
    main()
