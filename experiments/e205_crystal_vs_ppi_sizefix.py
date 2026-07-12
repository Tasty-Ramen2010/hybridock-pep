"""E205 — NEW size-fixed crystal model vs PPI on T100, r + MAE by LENGTH and CHARGE: where do we still lose,
and by how much, after the size-fix? Trains the crystal model (size-fix) on 925 minus T100, predicts T100,
compares to PPI's shipped predictions per slice.
"""
from __future__ import annotations

import importlib.util
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


def rmae(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def fit_regs(X, L):
    return {j: LinearRegression().fit(L.reshape(-1, 1), X[:, j]) for j in SIZE_IDX}


def apply_regs(X, L, regs):
    X = X.copy()
    for j, lr in regs.items():
        X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
    return X


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}

    def vec(d):
        g = {k: float(d.get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, d["seq"])
        return x[:240]

    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        seq = d["seq"]; pq = abs(sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq))
        test.append({"x": vec(d), "y": float(m["dg_exp"]), "ship": ship, "L": len(seq), "q": pq})
    tid = {p for p in man}  # hold out all T100 ids
    tr = [r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")) if r["pdb"].lower() not in tid]
    Xtr = np.array([vec(r) for r in tr]); ytr = np.array([float(r["y"]) for r in tr])
    Ltr = np.array([r["length"] for r in tr])
    regs = fit_regs(Xtr, Ltr)
    model = e202._hgb().fit(apply_regs(Xtr, Ltr, regs), ytr)

    Xte = np.array([t["x"] for t in test]); L = np.array([t["L"] for t in test]); q = np.array([t["q"] for t in test])
    ours = model.predict(apply_regs(Xte, L, regs))
    y = np.array([t["y"] for t in test]); ship = np.array([t["ship"] for t in test])

    print(f"=== NEW size-fixed CRYSTAL model vs PPI on T100 (n={len(test)}, held out of 925) ===")
    print(f"  {'slice':<16}{'n':>4}  {'OURS r':>7}{'PPI r':>7}{'Δr':>7}   {'OURS MAE':>9}{'PPI MAE':>9}{'ΔMAE':>7}  verdict")
    slices = [("OVERALL", np.ones(len(test), bool)),
              ("short<=8", L <= 8), ("med 9-12", (L >= 9) & (L <= 12)),
              ("long 13-16", (L >= 13) & (L <= 16)), ("vlong>=17", L >= 17),
              ("neutral|q|<=1", q <= 1), ("charged|q|>=2", q >= 2), ("v.charged|q|>=3", q >= 3)]
    for nm, mk in slices:
        if mk.sum() < 4:
            continue
        ro, mo = rmae(ours, y, mk); rp, mp = rmae(ship, y, mk)
        v = "WE WIN r" if (not np.isnan(ro) and ro > rp) else f"lose {ro-rp:+.2f}"
        if not np.isnan(mo) and mo < mp:
            v += " | win MAE"
        print(f"  {nm:<16}{int(mk.sum()):>4}  {ro:>+7.3f}{rp:>+7.3f}{ro-rp:>+7.3f}   {mo:>9.2f}{mp:>9.2f}{mo-mp:>+7.2f}  {v}")


if __name__ == "__main__":
    main()
