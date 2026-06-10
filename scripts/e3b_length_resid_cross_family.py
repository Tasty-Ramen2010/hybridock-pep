"""E3b — the decisive test for physical entropy: length-residualized cross-family.

E3 showed entropy SUMS hit one-per-family r~0.4, but they scale with length and
length itself correlates +0.43 with ΔG here — so the raw-sum cross-family signal
may just be length re-discovered. The composition (per-residue) forms were null.

This test fully separates length from composition ACROSS families:
  for each bootstrap, draw one peptide per family, then regress BOTH the feature
  and ΔG on peptide length WITHIN that cross-family subset, and correlate the
  residuals. If entropy composition carries real cross-family signal, it survives;
  if it was length, it collapses to ~0.

Run on ALL and Kd-only, all entropy features + a few structural references.
"""
from __future__ import annotations

import json
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
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def lenresid_one_per_family(y, v, L, g, n_boot=3000, seed=0):
    fams = {}
    for i, gi in enumerate(g):
        fams.setdefault(gi, []).append(i)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = np.array([rng.choice(m) for m in fams.values()])
        if len(idx) < 5:
            continue
        vv, yy, LL = v[idx], y[idx], L[idx]
        if np.std(vv) == 0 or np.std(LL) == 0:
            continue
        rs.append(pearsonr(resid(vv, LL), resid(yy, LL)).statistic)
    rs = np.array(rs)
    return rs.mean(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def main():
    rows = json.loads(Path("/tmp/e3_features.json").read_text())

    def col(k):
        return np.array([r[k] for r in rows], float)

    y, L = col("y"), col("L")
    seqs = [r["seq"] for r in rows]
    g = kmer_groups(seqs, 0.3)
    kd = np.array([r["aff"] == "Kd" for r in rows])

    feats = ["ent_sc", "ent_bb", "ent_tot", "ent_chi",
             "ent_sc_per_res", "ent_tot_per_res", "frac_flexible",
             "bsa", "nis_p_frac", "nis_c_frac"]

    print("LENGTH-RESIDUALIZED ONE-PER-FAMILY (true cross-family, size-free)")
    for split, mask in [("ALL", np.ones(len(rows), bool)), ("Kd", kd)]:
        print(f"\n-- {split} (n={int(mask.sum())}, fam={len(set(g[mask]))}) --")
        ym, Lm, gm = y[mask], L[mask], g[mask]
        print(f"{'feature':<18}{'lenresid_1perFam_r':>20}{'  95% CI':>18}")
        for f in feats:
            v = col(f)[mask]
            if np.std(v) == 0:
                continue
            r, lo, hi = lenresid_one_per_family(ym, v, Lm, gm)
            flag = "  <== REAL" if lo * hi > 0 and abs(r) >= 0.25 else ""
            print(f"{f:<18}{r:>20.3f}   [{lo:+.2f},{hi:+.2f}]{flag}")


if __name__ == "__main__":
    main()
