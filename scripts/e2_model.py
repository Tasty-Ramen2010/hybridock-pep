"""E2 model — push the HONEST affinity ceiling with orthogonal size-free axes.

Rigor:
  * every structural feature is length-residualized INSIDE each training fold
    (test fold uses train-fold regression coeffs) -> no size leakage (Trap #1).
  * all signs fixed A-PRIORI from physics/PRODIGY (never fit on this data).
  * GroupKFold by sequence family (whole family held out) -> no co-crystal
    leakage (Trap #3).
  * nested forward-selection picks features on TRAIN folds only (Trap #5).
  * equal-weight z-blend has no free parameters -> cannot overfit n=34.

Models: BSA / NIS / z-blend(orthogonal) / nested-NNLS / full-NNLS(cautionary).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]

# orient multiplier: makes oriented column POSITIVELY correlated with dG
# (dG negative=strong). +1 = feature means weaker binding, -1 = stronger.
ORIENT = {
    "bsa": -1, "nis_p_frac": -1, "nis_c_frac": +1,
    "hb_density": -1, "sb_density": -1, "ic_charged_frac": -1, "ic_apolar_frac": +1,
    "mean_hyd_contact": -1, "nis_apolar_area": +1, "nis_charged_area": +1,
    "nis_polar_area": -1, "buried_unsat_polar_frac": +1,
}
STRUCT = list(ORIENT)  # all length-residualized


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


def fit_resid(x, L):
    A = np.column_stack([np.ones_like(L), L])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return c


def apply_resid(x, L, c):
    return x - (c[0] + c[1] * L)


def prep_fold(Xtr, Ltr, Xte, Lte):
    """length-residualize + standardize each col using train stats only.
    returns oriented columns (positively corr w/ dG)."""
    out_tr, out_te = [], []
    for j, f in enumerate(STRUCT):
        c = fit_resid(Xtr[:, j], Ltr)
        rtr = apply_resid(Xtr[:, j], Ltr, c)
        rte = apply_resid(Xte[:, j], Lte, c)
        mu, sd = rtr.mean(), rtr.std() + 1e-9
        out_tr.append(ORIENT[f] * (rtr - mu) / sd)
        out_te.append(ORIENT[f] * (rte - mu) / sd)
    return np.column_stack(out_tr), np.column_stack(out_te)


def grouped_predict(X, L, y, g, model="zblend", subset=None, inner_select=False):
    pred = np.zeros_like(y)
    picks = []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
    for tr, te in gkf.split(X, y, g):
        Ztr, Zte = prep_fold(X[tr], L[tr], X[te], L[te])
        ytr = y[tr]
        if subset is not None:
            cols = [STRUCT.index(f) for f in subset]
            Ztr, Zte = Ztr[:, cols], Zte[:, cols]
        if model == "zblend":
            pred[te] = Zte.mean(1) * ytr.std() + ytr.mean()  # scale to dG range
        elif model == "nnls":
            sel = list(range(Ztr.shape[1]))
            if inner_select:
                sel = forward_select(Ztr, ytr, g[tr])
                picks.append([(subset or STRUCT)[i] for i in sel])
            w, _ = nnls(Ztr[:, sel], ytr - ytr.mean())
            pred[te] = Zte[:, sel] @ w + ytr.mean()
    return pearsonr(pred, y).statistic, picks


def forward_select(Z, y, g):
    inner = GroupKFold(n_splits=min(3, len(np.unique(g))))

    def oof_r(cols):
        p = np.zeros_like(y)
        for tr, te in inner.split(Z, y, g):
            w, _ = nnls(Z[tr][:, cols], y[tr] - y[tr].mean())
            p[te] = Z[te][:, cols] @ w + y[tr].mean()
        return pearsonr(p, y).statistic

    chosen, best = [], -1.0
    pool = list(range(Z.shape[1]))
    while pool:
        cand = sorted(((oof_r(chosen + [c]), c) for c in pool), reverse=True)
        if cand[0][0] <= best + 0.01:
            break
        best = cand[0][0]
        chosen.append(cand[0][1])
        pool.remove(cand[0][1])
    return chosen or [0]


def main():
    rows = json.loads(Path("/tmp/e2_features.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    seqs = np.array([sm.get(r["pdb"].upper(), "X") for r in rows])

    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], float)

    y_all, L_all = col("y"), col("L")
    Xall = np.column_stack([col(f) for f in STRUCT])
    kd = np.array([r["aff"] == "Kd" for r in rows])

    orth = ["nis_p_frac", "nis_c_frac", "hb_density", "mean_hyd_contact",
            "buried_unsat_polar_frac"]

    for label, mask in [("ALL (Kd+Ki)", np.ones(len(rows), bool)), ("Kd-only", kd)]:
        ok = mask & np.all(np.isfinite(Xall), 1) & np.isfinite(y_all)
        y, L, X = y_all[ok], L_all[ok], Xall[ok]
        g = kmer_groups(list(seqs[ok]), 0.3)
        print(f"\n=== {label}  (n={len(y)}, fam={len(np.unique(g))}) ===")

        r_bsa, _ = grouped_predict(X, L, y, g, "nnls", subset=["bsa"])
        r_nis, _ = grouped_predict(X, L, y, g, "nnls", subset=["nis_p_frac", "nis_c_frac"])
        r_zb, _ = grouped_predict(X, L, y, g, "zblend", subset=orth)
        r_zb_all, _ = grouped_predict(X, L, y, g, "zblend", subset=None)
        r_nest, picks = grouped_predict(X, L, y, g, "nnls", inner_select=True)
        r_full, _ = grouped_predict(X, L, y, g, "nnls")

        print(f"  BSA (resid)            {r_bsa:+.3f}")
        print(f"  NIS (resid)            {r_nis:+.3f}")
        print(f"  z-blend 5-orthogonal   {r_zb:+.3f}")
        print(f"  z-blend ALL 12         {r_zb_all:+.3f}")
        print(f"  nested-NNLS (selected) {r_nest:+.3f}")
        print(f"  full-NNLS 12 (caution) {r_full:+.3f}")
        from collections import Counter
        flat = Counter(f for p in picks for f in p)
        print(f"  nested picks (freq): {dict(flat.most_common())}")

        # bootstrap CI on the best of {zblend5, nested}
        best_name = max([("zblend5", r_zb), ("nested", r_nest)], key=lambda t: t[1])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            idx = rng.integers(0, len(y), len(y))
            try:
                gg = kmer_groups([seqs[ok][i] for i in idx], 0.3)
                if best_name[0] == "zblend5":
                    r, _ = grouped_predict(X[idx], L[idx], y[idx], gg, "zblend", subset=orth)
                else:
                    r, _ = grouped_predict(X[idx], L[idx], y[idx], gg, "nnls", inner_select=True)
                bs.append(r)
            except Exception:
                pass
        bs = np.array(bs)
        print(f"  >> best = {best_name[0]} {best_name[1]:+.3f}  "
              f"95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")


if __name__ == "__main__":
    main()
