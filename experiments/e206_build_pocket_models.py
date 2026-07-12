"""E206 — rebuild both scoring artifacts WITH the pocket-ProtDCal feature (E205 come-back lever, +0.13 r on
T100, wins charged). Feature vector is now 262 (240 base + 22 pocket-composition), via build_feature_vector
with pocket_seq in the geometry dict. Crystal keeps the size-fix; AI stays no-size-fix.
  data/affinity_crystal_sizefix.joblib  (262-feat, size-fix)
  data/affinity_ai_nofix.joblib         (262-feat, no size-fix)
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
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, SIZE_IDX  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402

NFEAT = 262
FEATURE_NAMES = (GEOMETRY_KEYS + [f"pd_{i}" for i in range(220)] + ["q_compl", "abs_q_match", "q_neutralize", "length"]
                 + [f"poc_pd_{i}" for i in range(22)])


def _hgb():
    return e202._hgb()


def fit_regs(X, L):
    return {j: LinearRegression().fit(L.reshape(-1, 1), X[:, j]) for j in SIZE_IDX}


def apply_regs(X, L, regs):
    X = X.copy()
    for j, lr in regs.items():
        X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
    return X


def vec(g0, seq, pid):
    g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = e158.pocket_seq(pid) or ""
    x = build_feature_vector(g, seq)
    return x[:NFEAT] if x.shape[0] >= NFEAT else np.pad(x, (0, NFEAT - x.shape[0]))


def load_crystal():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        rows.append((vec(r, r["seq"], r["pdb"]), float(r["y"]), r["length"], ps,
                     abs(sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"]))))
    return rows


def load_ai():
    rows = []
    e93 = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    for pid, e in e93.items():
        rows.append((vec(e["rank1"], e["seq"], pid), float(e["y"]), len(e["seq"]), e["seq"],
                     abs(sum(c in "KR" for c in e["seq"]) - sum(c in "DE" for c in e["seq"]))))
    for fn in ["e154_realpose_pdbbind.jsonl", "e176_long_n100.jsonl", "e176_short_n250.jsonl", "e176_vlong_n250.jsonl"]:
        p = ROOT / "data" / fn
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            g0 = e.get("rank1") or e
            rows.append((vec(g0, e["seq"], e["pdb"]), float(e["y"]), e["length"], e["seq"],
                         abs(sum(c in "KR" for c in e["seq"]) - sum(c in "DE" for c in e["seq"]))))
    seen = {}; out = []
    for r in rows:
        k = (r[3], round(r[1], 2))
        if k not in seen:
            seen[k] = 1; out.append(r)
    return out


def R(p, y, m):
    p, y = p[m], y[m]; ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def build(tag, rows, out, size_fix):
    X = np.nan_to_num(np.array([r[0] for r in rows])); y = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
    q = np.array([r[4] for r in rows]); grp, _ = e158.greedy_cluster([r[3] for r in rows], 0.7)
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        Xtr, Xte = X[tr], X[te]
        regs = fit_regs(Xtr, L[tr]) if size_fix else None
        if size_fix:
            Xtr, Xte = apply_regs(Xtr, L[tr], regs), apply_regs(Xte, L[te], regs)
        pred[te] = _hgb().fit(Xtr, y[tr]).predict(Xte)
    cv_r = R(pred, y, np.ones(len(y), bool))
    print(f"{tag} (size_fix={size_fix}) grouped-CV n={len(rows)}: overall={cv_r:+.3f} "
          f"short={R(pred,y,L<=8):+.3f} med={R(pred,y,(L>=9)&(L<=12)):+.3f} long={R(pred,y,(L>=13)&(L<=16)):+.3f} "
          f"vlong={R(pred,y,L>=17):+.3f} charged={R(pred,y,q>=2):+.3f} neutral={R(pred,y,q<=1):+.3f}")
    regs_full = fit_regs(X, L) if size_fix else None
    Xfit = apply_regs(X, L, regs_full) if size_fix else X
    mfull = _hgb().fit(Xfit, y)
    bundle = {"model": mfull, "feature_order": FEATURE_NAMES, "n_train": len(rows), "cv_r": cv_r,
              "size_fix": size_fix, "pocket_protdcal": True, "pose_type": tag}
    if size_fix:
        bundle["size_regs"] = {int(j): [float(lr.coef_[0]), float(lr.intercept_)] for j, lr in regs_full.items()}
    joblib.dump(bundle, out)
    print(f"  saved {out.name} ({NFEAT}-feat, pocket-ProtDCal)")


def main():
    build("crystal", load_crystal(), ROOT / "data/affinity_crystal_sizefix.joblib", size_fix=True)
    build("ai_realpose", load_ai(), ROOT / "data/affinity_ai_nofix.joblib", size_fix=False)


if __name__ == "__main__":
    main()
