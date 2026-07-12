"""E222 — verify/debunk the E221 leads:
  1. ΔG/length r=0.869 — REAL or 1/L artifact? Back-transform (predict ΔG/L → ×L → ΔG) and check.
  2. Is the residual predictable by a NONLINEAR model from all features? (linear said ⊥; test GBT.)
  3. WITHIN-RECEPTOR skill — for receptors with >=4 peptides, do our features RANK peptides correctly?
     (this is the FEP-free signal; receptor baseline cancels.)
  4. The real hidden variable: receptor one-hot vs our pocket features — quantify the recoverable ceiling.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import linregress, spearmanr  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        g = {k: float(r.get(k, 0.0)) for k in GEOMETRY_KEYS}; g["pocket_seq"] = ps
        x = build_feature_vector(g, r["seq"])
        x = x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))
        rows.append({"ps": ps, "y": float(r["y"]), "L": r["length"], "x": x, "pdb": r["pdb"].lower()})
    n = len(rows); y = np.array([r["y"] for r in rows]); L = np.array([r["L"] for r in rows])
    X = np.nan_to_num([r["x"] for r in rows]); grp = np.array(e158.greedy_cluster([r["ps"] for r in rows], 0.7)[0])

    def cv(target, Xuse=None):
        Xuse = X if Xuse is None else Xuse
        p = np.full(n, np.nan)
        for tr, te in GroupKFold(5).split(Xuse, target, grp):
            p[te] = e202._hgb().fit(Xuse[tr], target[tr]).predict(Xuse[te])
        return p

    # 1. ΔG/L debunk
    print("=== 1. ΔG/length r=0.869 — REAL or 1/L artifact? ===")
    p_eff = cv(y / L)
    r_eff = np.corrcoef(p_eff, y / L)[0, 1]
    dg_from_eff = p_eff * L  # back-transform to ΔG
    r_back = np.corrcoef(dg_from_eff, y)[0, 1]
    p_raw = cv(y); r_raw = np.corrcoef(p_raw, y)[0, 1]
    # baseline: predict ΔG/L using ONLY 1/L (no real features)
    invL = (1.0 / L).reshape(-1, 1)
    p_invL = cv(y / L, Xuse=invL); r_invL = np.corrcoef(p_invL, y / L)[0, 1]
    print(f"  predict ΔG/L (full feats):     r={r_eff:+.3f}")
    print(f"  predict ΔG/L using ONLY 1/L:   r={r_invL:+.3f}  ← if ≈ same, it's the 1/L artifact")
    print(f"  back-transform (ΔG/L ×L → ΔG): r={r_back:+.3f}  vs direct ΔG r={r_raw:+.3f}")
    print(f"  VERDICT: {'ARTIFACT (1/L dominates; back-transform = direct)' if abs(r_back-r_raw)<0.05 else 'REAL gain'}\n")

    # 2. residual predictable nonlinearly?
    pred = cv(y); resid = y - pred
    p_resid = cv(resid)
    print(f"=== 2. is the residual predictable by GBT from all features? ===")
    print(f"  GBT(features → residual) grouped-CV r={np.corrcoef(p_resid,resid)[0,1]:+.3f}  "
          f"→ {'LEFTOVER SIGNAL EXISTS' if np.corrcoef(p_resid,resid)[0,1]>0.15 else 'residual is NOISE/FEP (no learnable signal)'}\n")

    # 3. within-receptor skill (FEP-free, receptor baseline cancels)
    fam = defaultdict(list)
    for i in range(n):
        fam[grp[i]].append(i)
    multi = [c for c in fam if len(fam[c]) >= 4 and np.std(y[fam[c]]) > 0.3]
    print(f"=== 3. WITHIN-RECEPTOR ranking skill ({len(multi)} receptors w/ >=4 peptides, affinity spread) ===")
    # leave-one-receptor-out, then within-receptor spearman
    p = np.full(n, np.nan)
    from sklearn.model_selection import LeaveOneGroupOut
    for tr, te in LeaveOneGroupOut().split(X, y, grp):
        p[te] = e202._hgb().fit(X[tr], y[tr]).predict(X[te])
    taus = [spearmanr(p[fam[c]], y[fam[c]]).statistic for c in multi]
    taus = [t for t in taus if not np.isnan(t)]
    print(f"  mean within-receptor Spearman τ = {np.mean(taus):+.3f} (median {np.median(taus):+.3f})  "
          f"→ {'we DO rank within receptor' if np.mean(taus)>0.15 else 'WEAK even within receptor = features dont separate peptides'}")
    print(f"  (this is the receptor-baseline-free skill; >0 = real peptide discrimination)\n")

    # 4. receptor one-hot ceiling (multi-receptors only, where identity is defined)
    print("=== 4. receptor identity ceiling (multi-peptide receptors) ===")
    midx = np.array([i for c in multi for i in fam[c]])
    if len(midx) > 20:
        ym = y[midx]; gm = grp[midx]
        loo_rm = np.array([(y[grp == grp[i]].sum() - y[i]) / max((grp == grp[i]).sum() - 1, 1) for i in midx])
        # our model on this subset
        pm = p[midx]
        Xm = np.hstack([X[midx], loo_rm.reshape(-1, 1)])
        po = np.full(len(midx), np.nan)
        for tr, te in GroupKFold(min(5, len(set(gm)))).split(Xm, ym, gm):
            po[te] = e202._hgb().fit(Xm[tr], ym[tr]).predict(Xm[te])
        print(f"  on multi-receptor subset (n={len(midx)}): ours r={np.corrcoef(pm,ym)[0,1]:+.3f}  "
              f"ours+true-recmean r={np.corrcoef(po,ym)[0,1]:+.3f}")
        print(f"  receptor-mean alone (LOO) r={np.corrcoef(loo_rm,ym)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
