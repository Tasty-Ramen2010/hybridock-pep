"""E208 — FAIR T100 benchmark: exclude the vlong (>=17) complexes whose PPI numbers are redundancy-inflated
(n=16 BioLiP, in PPI's own T949 training; FEP-bound for us). Standing benchmark from now on. Reports our
262-feat pocket model (size-fix, held out of 925) vs PPI shipped, overall + by length + charge.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, SIZE_IDX  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402

EXCLUDE_VLONG = True  # the standing fair-benchmark rule


def rmae(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def vec(d, pid):
    g = {k: float(d.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = e158.pocket_seq(pid) or ""
    x = build_feature_vector(g, d["seq"])
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        seq = d["seq"]
        if EXCLUDE_VLONG and len(seq) >= 17:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        test.append({"x": vec(d, pid), "y": float(m["dg_exp"]), "ship": ship,
                     "L": len(seq), "q": abs(sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq))})
    tid = set(man)
    tr = [r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))
          if r["pdb"].lower() not in tid and e158.pocket_seq(r["pdb"]) is not None]
    Xtr = np.nan_to_num([vec(r, r["pdb"].lower()) for r in tr]); ytr = np.array([float(r["y"]) for r in tr])
    Ltr = np.array([r["length"] for r in tr])
    regs = {j: LinearRegression().fit(Ltr.reshape(-1, 1), Xtr[:, j]) for j in SIZE_IDX}

    def ap(X, L):
        X = X.copy()
        for j, lr in regs.items():
            X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
        return X
    model = e202._hgb().fit(ap(Xtr, Ltr), ytr)
    Xte = np.nan_to_num([t["x"] for t in test]); L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])
    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])
    ours = model.predict(ap(Xte, L))

    print(f"=== FAIR T100 benchmark (vlong EXCLUDED, n={len(test)}) — 262-feat pocket model vs PPI ===")
    print(f"  {'slice':<16}{'n':>4}  {'OURS r':>7}{'PPI r':>7}{'Δr':>7}   {'OURS MAE':>9}{'PPI MAE':>9}  verdict")
    for nm, mk in [("OVERALL", np.ones(len(test), bool)), ("short<=8", L <= 8), ("med 9-12", (L >= 9) & (L <= 12)),
                   ("long 13-16", (L >= 13) & (L <= 16)), ("neutral|q|<=1", q <= 1),
                   ("charged|q|>=2", q >= 2), ("v.charged|q|>=3", q >= 3)]:
        if mk.sum() < 4:
            continue
        ro, mo = rmae(ours, y, mk); rp, mp = rmae(ship, y, mk)
        v = "WE WIN r" if (not np.isnan(ro) and ro > rp) else f"lose {ro-rp:+.2f}"
        if not np.isnan(mo) and mo < mp:
            v += " | win MAE"
        print(f"  {nm:<16}{int(mk.sum()):>4}  {ro:>+7.3f}{rp:>+7.3f}{ro-rp:>+7.3f}   {mo:>9.2f}{mp:>9.2f}  {v}")


if __name__ == "__main__":
    main()
