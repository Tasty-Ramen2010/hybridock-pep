"""E216 — build the production routed CRYSTAL artifact: main size-fix model (all 925, used for L<17) + a
vlong sub-model (925 + PPIKB-vlong augmentation, used for L>=17). The router fires ONLY for vlong so every
other band is byte-identical to the current production crystal model. Saves data/affinity_crystal_sizefix.joblib
with an embedded vlong_model + vlong_size_regs + vlong_threshold.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, SIZE_IDX  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402

FEATURE_NAMES = (GEOMETRY_KEYS + [f"pd_{i}" for i in range(220)] + ["q_compl", "abs_q_match", "q_neutralize", "length"]
                 + [f"poc_pd_{i}" for i in range(22)])


def vec(g0, seq, pocket_seq):
    g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = pocket_seq or ""
    x = build_feature_vector(g, seq)
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def regs_of(X, L):
    return {int(j): LinearRegression().fit(L.reshape(-1, 1), X[:, j]) for j in SIZE_IDX}


def apply_regs(X, L, regs):
    X = X.copy()
    for j, lr in regs.items():
        X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
    return X


def serialize(regs):
    return {int(j): [float(lr.coef_[0]), float(lr.intercept_)] for j, lr in regs.items()}


def main():
    # main: all 925
    tr = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        tr.append((vec(r, r["seq"], ps), float(r["y"]), r["length"]))
    X = np.nan_to_num([t[0] for t in tr]); y = np.array([t[1] for t in tr]); L = np.array([t[2] for t in tr])
    rg = regs_of(X, L); main_model = e202._hgb().fit(apply_regs(X, L, rg), y)

    def ppikb_aug(lo, hi):
        """925 + PPIKB band-augmented sub-model (geom-complete, E212). Used for lo<=L<=hi only."""
        aug = []
        for ln in (ROOT / "data/e212_ppikb_geom.jsonl").read_text().splitlines():
            e = json.loads(ln)
            if not e.get("geom") or not (lo <= e["length"] <= hi):
                continue
            g = {k: float(e["geom"].get(k, 0.0)) for k in GEOMETRY_KEYS}
            x = build_feature_vector(g, e["seq"])
            x = (x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))).copy()
            if e.get("pocket_pkf") and len(e["pocket_pkf"]) == 22:
                x[240:262] = np.array(e["pocket_pkf"], float)
            aug.append((np.nan_to_num(x), float(e["y"]), e["length"]))
        Xa = np.vstack([X, np.array([a[0] for a in aug])]); ya = np.concatenate([y, np.array([a[1] for a in aug])])
        La = np.concatenate([L, np.array([a[2] for a in aug])])
        rga = regs_of(Xa, La)
        return e202._hgb().fit(apply_regs(Xa, La, rga), ya), rga, len(aug)

    # band specialists (E215/E238 validated): long 13-16 (+0.026 T100), vlong >=17 (+0.107 T100).
    long_model, rgl, n_long = ppikb_aug(13, 16)
    vlong_model, rgv, n_vlong = ppikb_aug(17, 999)

    bundle = {"model": main_model, "size_regs": serialize(rg), "feature_order": FEATURE_NAMES,
              "n_train": len(tr), "size_fix": True, "pocket_protdcal": True, "pose_type": "crystal",
              "long_model": long_model, "long_size_regs": serialize(rgl), "long_band": [13, 16],
              "vlong_model": vlong_model, "vlong_size_regs": serialize(rgv), "vlong_threshold": 17,
              "vlong_n_train": len(tr) + n_vlong, "long_n_train": len(tr) + n_long}
    out = ROOT / "data/affinity_crystal_sizefix.joblib"
    joblib.dump(bundle, out)
    print(f"saved routed crystal: main n={len(tr)}, +long13-16 sub (+{n_long} PPIKB), "
          f"+vlong>=17 sub (+{n_vlong} PPIKB)")


if __name__ == "__main__":
    main()
