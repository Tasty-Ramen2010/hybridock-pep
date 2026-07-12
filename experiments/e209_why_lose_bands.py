"""E209 — answer Ram's 3 questions:
  Q1: why does CRYSTAL-T100 short lose (0.165) despite the AI campaign? (campaign = DEPLOYMENT not crystal)
  Q2: why do we lose neutral + long? (home-field: split T100 into PDBbind-overlap vs BioLiP-only)
  Q3: would per-band CALIBRATION fix it? (math: Pearson r is invariant under affine recalibration → test)
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


def rmae(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def vec(d, pid):
    g = {k: float(d.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = e158.pocket_seq(pid) or ""
    x = build_feature_vector(d if False else g, d["seq"])
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    our_pdbs = set(have)
    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        seq = d["seq"]
        if len(seq) >= 17:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        test.append({"x": vec(d, pid), "y": float(m["dg_exp"]), "ship": ship, "L": len(seq),
                     "q": abs(sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)),
                     "in925": pid in our_pdbs})
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
    in925 = np.array([t["in925"] for t in test])
    ours = model.predict(ap(Xte, L))

    # Q2: home-field — split each losing band into PDBbind-overlap vs BioLiP-only
    print("=== Q2: HOME-FIELD split (in our 925 = fair ground · BioLiP-only = PPI's turf) ===")
    print(f"  {'band':<16}{'subset':<14}{'n':>4}{'OURS':>8}{'PPI':>8}")
    for nm, bm in [("short<=8", L <= 8), ("long13-16", (L >= 13) & (L <= 16)), ("neutral|q|<=1", q <= 1)]:
        for sn, sm in [("PDBbind-overlap", in925), ("BioLiP-only", ~in925)]:
            mk = bm & sm
            if mk.sum() >= 4:
                ro, _ = rmae(ours, y, mk); rp, _ = rmae(ship, y, mk)
                print(f"  {nm:<16}{sn:<14}{int(mk.sum()):>4}{ro:>+8.3f}{rp:>+8.3f}")

    # Q3: per-band affine CALIBRATION — does it change r? (it cannot; demonstrate) and MAE?
    print("\n=== Q3: per-band CALIBRATION (fit a,b per band: ŷ=a·pred+b) — r vs MAE ===")
    print(f"  {'band':<16}{'n':>4}  {'r raw':>7}{'r calib':>8}   {'MAE raw':>8}{'MAE calib':>10}")
    for nm, mk in [("short<=8", L <= 8), ("med9-12", (L >= 9) & (L <= 12)),
                   ("long13-16", (L >= 13) & (L <= 16)), ("neutral|q|<=1", q <= 1), ("charged|q|>=2", q >= 2)]:
        if mk.sum() < 4:
            continue
        p, yy = ours[mk], y[mk]
        # LOO affine calibration (honest): fit a,b on the rest, apply to held-out
        cal = np.full(len(p), np.nan)
        for i in range(len(p)):
            tr_i = np.arange(len(p)) != i
            lr = LinearRegression().fit(p[tr_i].reshape(-1, 1), yy[tr_i])
            cal[i] = lr.predict(p[i:i+1].reshape(-1, 1))[0]
        r_raw, m_raw = rmae(ours, y, mk)
        r_cal = float(np.corrcoef(cal, yy)[0, 1]); m_cal = float(np.mean(np.abs(cal - yy)))
        print(f"  {nm:<16}{int(mk.sum()):>4}  {r_raw:>+7.3f}{r_cal:>+8.3f}   {m_raw:>8.2f}{m_cal:>10.2f}")
    print("  → r is ~UNCHANGED by affine calibration (math fact); only MAE moves. Calibration fixes BIAS not RANKING.")


if __name__ == "__main__":
    main()
