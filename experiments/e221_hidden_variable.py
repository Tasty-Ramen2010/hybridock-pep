"""E221 — the DEEPEST dive into the regression-to-the-mean: is there a hidden variable?

5 structured tests on crystal-925:
  A. VARIANCE DECOMPOSITION — how much affinity variance is BETWEEN-receptor vs WITHIN-receptor? If most is
     between-receptor and our features are peptide-dominated, we predict the global mean → shrinkage.
  B. RECEPTOR-MEAN ORACLE — if we KNEW each receptor's mean affinity, how much does that alone explain?
     (= the ceiling a perfect receptor-identity feature would give.) Plus: do our pocket features recover it?
  C. RESIDUAL-VS-EVERYTHING — after our prediction, correlate the SIGNED + ABSOLUTE residual with EVERY
     feature (geometry, ProtDCal, pocket, charge, SS, anchor) to find leftover signal we're not using.
  D. PER-RESIDUE / NORMALIZED TARGET — is ΔG/length or ligand-efficiency more learnable (less shrinkage)?
  E. ORACLE FEATURE CEILING — train on a feature set augmented with the true receptor-mean; what's max r?
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
from scipy.stats import linregress  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import (build_feature_vector, GEOMETRY_KEYS, SIZE_IDX,  # noqa: E402
                                                  _protdcal_descriptors, _SCALES)
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
SN = list(_SCALES.keys())
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        g = {k: float(r.get(k, 0.0)) for k in GEOMETRY_KEYS}; g["pocket_seq"] = ps
        x = build_feature_vector(g, r["seq"])
        x = x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))
        s = ss.get(pid, {})
        rows.append({"pid": pid, "seq": r["seq"], "ps": ps, "y": float(r["y"]), "L": r["length"],
                     "x": x, "helix": float(s.get("helix", 0)), "sheet": float(s.get("sheet", 0))})
    n = len(rows)
    y = np.array([r["y"] for r in rows]); L = np.array([r["L"] for r in rows])
    X = np.nan_to_num([r["x"] for r in rows])
    # receptor groups = pocket-sequence clusters (each cluster ≈ a receptor/family)
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    grp = np.array(grp)
    print(f"crystal-925: n={n}, receptor-clusters(0.7)={len(set(grp))}, mean cluster size={n/len(set(grp)):.1f}\n")

    # ---------- A. VARIANCE DECOMPOSITION ----------
    cl_means = {c: y[grp == c].mean() for c in set(grp)}
    between = np.array([cl_means[c] for c in grp])
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_between = ((between - y.mean()) ** 2).sum()
    print("=== A. VARIANCE DECOMPOSITION (affinity y) ===")
    print(f"  between-receptor variance: {ss_between/ss_tot:.1%}   within-receptor: {1-ss_between/ss_tot:.1%}")
    multi = [c for c in set(grp) if (grp == c).sum() >= 3]
    within_std = np.mean([y[grp == c].std() for c in multi])
    print(f"  mean within-receptor affinity std: {within_std:.2f} kcal/mol (over {len(multi)} receptors w/ >=3 peptides)")
    print(f"  global affinity std: {y.std():.2f}  → if we only nailed receptor-mean, r_ceiling≈{np.sqrt(ss_between/ss_tot):.2f}\n")

    # ---------- B. RECEPTOR-MEAN ORACLE ----------
    # leave-one-out receptor mean (no self-leak)
    loo_recmean = np.array([(y[grp == grp[i]].sum() - y[i]) / max((grp == grp[i]).sum() - 1, 1) for i in range(n)])
    r_recmean = np.corrcoef(loo_recmean, y)[0, 1]
    print("=== B. RECEPTOR-MEAN ORACLE (knowing the receptor's avg affinity, peptide-blind) ===")
    print(f"  LOO receptor-mean alone predicts y at r={r_recmean:+.3f}  (this is pure receptor identity, NO peptide info)")
    # how well do our POCKET features recover the receptor mean?
    pkf_idx = slice(240, 262)  # pocket-ProtDCal block
    Xpoc = X[:, 240:262]
    pred_rm = np.full(n, np.nan)
    for tr, te in GroupKFold(5).split(Xpoc, loo_recmean, grp):
        pred_rm[te] = e202._hgb().fit(Xpoc[tr], loo_recmean[tr]).predict(Xpoc[te])
    print(f"  our pocket features predict the receptor-mean at r={np.corrcoef(pred_rm,loo_recmean)[0,1]:+.3f} "
          f"(grouped-CV) → {'GOOD' if np.corrcoef(pred_rm,loo_recmean)[0,1]>0.5 else 'WEAK = we miss receptor identity'}\n")

    # ---------- C. our model + RESIDUAL vs EVERYTHING ----------
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        regs = {j: LinearRegression().fit(L[tr].reshape(-1, 1), X[tr][:, j]) for j in SIZE_IDX}
        Xtr = X[tr].copy(); Xte = X[te].copy()
        for j, lr in regs.items():
            Xtr[:, j] -= lr.predict(L[tr].reshape(-1, 1)); Xte[:, j] -= lr.predict(L[te].reshape(-1, 1))
        pred[te] = e202._hgb().fit(Xtr, y[tr]).predict(Xte)
    resid = y - pred  # signed: + means we UNDER-predicted (true stronger than we said)
    slope = linregress(pred, y).slope
    print(f"=== C. our model r={np.corrcoef(pred,y)[0,1]:+.3f}, shrink-slope={slope:.2f} ===")
    print("  what does the SIGNED residual (y−pred; + = we under-predicted) still correlate with?")
    cand = {
        "receptor_mean(LOO)": loo_recmean, "y_itself(ceiling)": y, "length": L.astype(float),
        "pocket_pred_recmean": pred_rm,
        "pep_hyd": np.array([np.mean([_SCALES["kd"].get(c, 0) for c in r["seq"]]) for r in rows]),
        "pock_n": X[:, GEOMETRY_KEYS.index("poc_n")], "bsa_hyd": X[:, GEOMETRY_KEYS.index("bsa_hyd")],
        "mean_burial": X[:, GEOMETRY_KEYS.index("mean_burial")], "mj_contact": X[:, GEOMETRY_KEYS.index("mj_contact")],
        "helix": np.array([r["helix"] for r in rows]), "sheet": np.array([r["sheet"] for r in rows]),
        "hb_count": X[:, GEOMETRY_KEYS.index("hb_count")], "org_density": X[:, GEOMETRY_KEYS.index("org_density")],
    }
    cors = sorted(((np.corrcoef(v, resid)[0, 1], k) for k, v in cand.items() if np.std(v) > 1e-9), key=lambda x: -abs(x[0]))
    for c, k in cors:
        flag = " ← LEFTOVER SIGNAL" if (abs(c) > 0.2 and k not in ("y_itself(ceiling)",)) else ""
        print(f"    corr(residual, {k:<22}) = {c:+.3f}{flag}")

    # ---------- D. NORMALIZED TARGET ----------
    print("\n=== D. is a NORMALIZED target less shrunk? ===")
    for tname, yt in [("ΔG (raw)", y), ("ΔG/length (efficiency)", y / L), ("ΔG/sqrt(L)", y / np.sqrt(L))]:
        p = np.full(n, np.nan)
        for tr, te in GroupKFold(5).split(X, yt, grp):
            p[te] = e202._hgb().fit(X[tr], yt[tr]).predict(X[te])
        print(f"  {tname:<24} r={np.corrcoef(p,yt)[0,1]:+.3f}  shrink-slope={linregress(p,yt).slope:.2f}")

    # ---------- E. ORACLE FEATURE CEILING ----------
    print("\n=== E. ORACLE: add the TRUE receptor-mean as a feature — max r if we PERFECTLY knew receptor ===")
    Xo = np.hstack([X, loo_recmean.reshape(-1, 1)])
    po = np.full(n, np.nan)
    for tr, te in GroupKFold(5).split(Xo, y, grp):
        po[te] = e202._hgb().fit(Xo[tr], y[tr]).predict(Xo[te])
    print(f"  model + perfect-receptor-mean: r={np.corrcoef(po,y)[0,1]:+.3f}  shrink-slope={linregress(po,y).slope:.2f}")
    print(f"  (vs our current {np.corrcoef(pred,y)[0,1]:+.3f}) → the gap = how much a perfect receptor feature would add")


if __name__ == "__main__":
    main()
