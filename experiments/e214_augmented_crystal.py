"""E214 — SHIP: retrain the production crystal model (262-feat) with PPIKB long/vlong augmentation (geometry
now computed, e212). Build full 262-feat vectors for the 74 geometry-complete PPIKB long/vlong complexes,
add to the 925 training, validate on T100 (held out) — does long/vlong climb? If yes, save the augmented
crystal artifact.
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


def rmae(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def vec(g0, seq, pocket_seq):
    g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = pocket_seq or ""
    x = build_feature_vector(g, seq)
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}

    # ---- 925 training ----
    tr925 = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        if r["pdb"].lower() in man:
            continue
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        tr925.append((vec(r, r["seq"], ps), float(r["y"]), r["length"]))

    # ---- PPIKB long/vlong augmentation (geometry from e212) ----
    aug = []
    for ln in (ROOT / "data/e212_ppikb_geom.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if not e.get("geom"):
            continue
        # reconstruct pocket_seq is unavailable; use pocket_pkf path → build_feature_vector needs pocket_seq.
        # We have pocket_pkf precomputed (e188); inject directly by faking a pocket_seq is wrong. Instead
        # build base-240 from geom+seq, then append the cached pocket_pkf as the 22-block.
        g = {k: float(e["geom"].get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, e["seq"])  # 262 but pocket block = zeros (no pocket_seq)
        x = x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))
        if e.get("pocket_pkf") and len(e["pocket_pkf"]) == 22:
            x[240:262] = np.array(e["pocket_pkf"], float)  # inject cached pocket-ProtDCal
        aug.append((np.nan_to_num(x), float(e["y"]), e["length"]))
    print(f"925 train: {len(tr925)} | PPIKB long/vlong aug (geom-complete): {len(aug)}")

    # ---- T100 test ----
    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        seq = d["seq"]
        ps = e158.pocket_seq(pid) or ""
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        test.append((vec(d, seq, ps), float(m["dg_exp"]), len(seq), ship,
                     abs(sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq))))
    Xte = np.nan_to_num([t[0] for t in test]); y = np.array([t[1] for t in test])
    L = np.array([t[2] for t in test]); ship = np.array([t[3] for t in test]); q = np.array([t[4] for t in test])

    def train_eval(rows, label, save=None):
        X = np.nan_to_num([r[0] for r in rows]); yt = np.array([r[1] for r in rows]); Lt = np.array([r[2] for r in rows])
        regs = {j: LinearRegression().fit(Lt.reshape(-1, 1), X[:, j]) for j in SIZE_IDX}

        def ap(XX, LL):
            XX = XX.copy()
            for j, lr in regs.items():
                XX[:, j] = XX[:, j] - lr.predict(LL.reshape(-1, 1))
            return XX
        m = e202._hgb().fit(ap(X, Lt), yt)
        p = m.predict(ap(Xte, L))
        nv = L < 17
        print(f"  {label:<26} T100: overall={rmae(p,y,np.ones(len(y),bool))[0]:+.3f}  "
              f"fair(no-vlong)={rmae(p,y,nv)[0]:+.3f}  long={rmae(p,y,(L>=13)&(L<=16))[0]:+.3f}  "
              f"vlong={rmae(p,y,L>=17)[0]:+.3f}  charged={rmae(p,y,q>=2)[0]:+.3f}")
        if save:
            joblib.dump({"model": m, "size_regs": {int(j): [float(lr.coef_[0]), float(lr.intercept_)] for j, lr in regs.items()},
                         "feature_order": FEATURE_NAMES, "n_train": len(rows), "size_fix": True,
                         "pocket_protdcal": True, "ppikb_augmented": True, "pose_type": "crystal"}, ROOT / "data" / save)
            print(f"    saved {save}")

    print("=== AUGMENTED production crystal (262-feat, size-fix) on T100 ===")
    train_eval(tr925, "925 only (current prod)")
    train_eval(tr925 + aug, "925 + PPIKB long/vlong", save="affinity_crystal_augmented.joblib")
    nv = L < 17
    print(f"  {'PPI shipped':<26} T100: overall={rmae(ship,y,np.ones(len(y),bool))[0]:+.3f}  "
          f"fair(no-vlong)={rmae(ship,y,nv)[0]:+.3f}  long={rmae(ship,y,(L>=13)&(L<=16))[0]:+.3f}  "
          f"vlong={rmae(ship,y,L>=17)[0]:+.3f}  charged={rmae(ship,y,q>=2)[0]:+.3f}")


if __name__ == "__main__":
    main()
