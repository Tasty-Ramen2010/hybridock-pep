"""E3c — gold-standard cross-family test: collapse each family to its MEAN.

The earlier 'NIS dead (0.065)' came from ONE noisy n=20 NNLS draw; E3b's bootstrap
said NIS survives but its CI could be tight just because most families are
singletons. This removes all ambiguity: average each feature and ΔG WITHIN each
family -> one fully-independent point per family -> correlate across family means,
length-residualized. Jackknife-over-families CI. This is the most honest
cross-family correlation the dataset can produce.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
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


def jackknife_r(v, y):
    """Pearson r + leave-one-out jackknife 95% CI."""
    r = pearsonr(v, y).statistic
    n = len(v)
    js = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        if np.std(v[m]) == 0:
            continue
        js.append(pearsonr(v[m], y[m]).statistic)
    js = np.array(js)
    se = np.sqrt((n - 1) / n * np.sum((js - js.mean()) ** 2)) if len(js) else np.nan
    return r, r - 1.96 * se, r + 1.96 * se


def main():
    rows = json.loads(Path("/tmp/e3_features.json").read_text())

    def famcollapse(mask):
        sub = [r for r, mm in zip(rows, mask) if mm]
        seqs = [r["seq"] for r in sub]
        g = kmer_groups(seqs, 0.3)
        feats = ["ent_sc", "ent_chi", "ent_tot_per_res", "frac_flexible",
                 "bsa", "nis_p_frac", "nis_c_frac", "vina"]
        fam_y, fam_L, fam_f = {}, {}, {f: {} for f in feats}
        for i, gi in enumerate(g):
            fam_y.setdefault(gi, []).append(sub[i]["y"])
            fam_L.setdefault(gi, []).append(sub[i]["L"])
            for f in feats:
                fam_f[f].setdefault(gi, []).append(sub[i].get(f, np.nan))
        fams = sorted(fam_y)
        Y = np.array([np.mean(fam_y[k]) for k in fams])
        L = np.array([np.mean(fam_L[k]) for k in fams])
        return fams, Y, L, {f: np.array([np.nanmean(fam_f[f][k]) for k in fams]) for f in feats}

    kd = np.array([r["aff"] == "Kd" for r in rows])
    for split, mask in [("ALL", np.ones(len(rows), bool)), ("Kd", kd)]:
        fams, Y, L, F = famcollapse(mask)
        print(f"\n=== {split}: {len(fams)} independent families "
              f"(collapsed from n={int(mask.sum())}) ===")
        Yr = resid(Y, L)
        print(f"{'feature':<16}{'raw_r':>8}{'lenresid_r':>12}{'  95% CI (jackknife)':>22}")
        for f, v in F.items():
            if np.std(v) == 0 or np.any(~np.isfinite(v)):
                # drop NaN families pairwise
                ok = np.isfinite(v)
                if ok.sum() < 6:
                    continue
                vv, yy, ll = v[ok], Y[ok], L[ok]
            else:
                vv, yy, ll = v, Y, L
            raw = pearsonr(vv, yy).statistic
            r, lo, hi = jackknife_r(resid(vv, resid(ll, np.zeros_like(ll)) if False else ll),
                                    resid(yy, ll))
            flag = "  <== REAL" if lo * hi > 0 and abs(r) >= 0.3 else ""
            print(f"{f:<16}{raw:>8.3f}{r:>12.3f}   [{lo:+.2f},{hi:+.2f}]{flag}")


if __name__ == "__main__":
    main()
