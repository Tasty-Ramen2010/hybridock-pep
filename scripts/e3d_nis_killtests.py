"""E3d — try to KILL the cross-family NIS signal three ways.

E3c said nis_p_frac is significant cross-family at the family-mean level. Before
believing it (it reverses a prior conclusion), attack it:

  1. PERMUTATION test: shuffle family-mean ΔG labels 20000x, recompute the
     length-residualized r; exact p = P(|r_perm| >= |r_obs|). Robust at small n.
  2. THRESHOLD sensitivity: re-cluster families at Jaccard 0.2/0.3/0.4/0.5 and at
     a strict whole-receptor grouping; signal must persist.
  3. LENGTH-LEAK check: corr(nis_p, length) — if large, the linear residualization
     may not have removed a nonlinear size effect; also report r after rank
     (Spearman) residualization which is nonparametric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]


def kmer_groups(seqs, threshold=0.3, k=3):
    ks = [{s[i:i+k] for i in range(max(0, len(s)-k+1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - threshold).fit_predict(D)


def resid(x, z):
    if np.std(z) == 0:
        return x - x.mean()
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def collapse(rows, mask, feat, threshold=0.3):
    sub = [r for r, mm in zip(rows, mask) if mm]
    g = kmer_groups([r["seq"] for r in sub], threshold)
    fy, fL, fv = {}, {}, {}
    for i, gi in enumerate(g):
        fy.setdefault(gi, []).append(sub[i]["y"])
        fL.setdefault(gi, []).append(sub[i]["L"])
        fv.setdefault(gi, []).append(sub[i].get(feat, np.nan))
    ks = sorted(fy)
    Y = np.array([np.mean(fy[k]) for k in ks])
    L = np.array([np.mean(fL[k]) for k in ks])
    V = np.array([np.nanmean(fv[k]) for k in ks])
    return Y, L, V


def perm_p(V, Y, L, n=20000, seed=0):
    vr, yr = resid(V, L), resid(Y, L)
    r_obs = pearsonr(vr, yr).statistic
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        yp = resid(Y[rng.permutation(len(Y))], L)
        if abs(pearsonr(vr, yp).statistic) >= abs(r_obs):
            cnt += 1
    return r_obs, (cnt + 1) / (n + 1)


def main():
    rows = json.loads(Path("/tmp/e3_features.json").read_text())
    kd = np.array([r["aff"] == "Kd" for r in rows])

    for split, mask in [("ALL", np.ones(len(rows), bool)), ("Kd", kd)]:
        print(f"\n########## {split} (n={int(mask.sum())}) ##########")
        # 1. permutation test for nis_p_frac
        Y, L, V = collapse(rows, mask, "nis_p_frac", 0.3)
        r_obs, p = perm_p(V, Y, L)
        print(f"[1] nis_p_frac family-mean lenresid r = {r_obs:+.3f}  "
              f"permutation p = {p:.4f}  ({len(Y)} families)")

        # 2. threshold sensitivity
        print("[2] threshold sensitivity (nis_p_frac lenresid r):")
        for th in (0.2, 0.3, 0.4, 0.5):
            Y2, L2, V2 = collapse(rows, mask, "nis_p_frac", th)
            r2 = pearsonr(resid(V2, L2), resid(Y2, L2)).statistic
            print(f"      Jaccard {th}: {len(Y2)} families  r = {r2:+.3f}")

        # 3. length-leak + nonparametric
        Yr, Lr, Vr = collapse(rows, mask, "nis_p_frac", 0.3)
        rL = pearsonr(Vr, Lr).statistic
        # Spearman (rank) residualization: rank-resid both on length-rank
        sr = spearmanr(resid(rankdata(Vr), rankdata(Lr)),
                       resid(rankdata(Yr), rankdata(Lr))).statistic
        print(f"[3] corr(nis_p, length) = {rL:+.3f}   "
              f"rank-residualized cross-family r = {sr:+.3f}")


if __name__ == "__main__":
    main()
