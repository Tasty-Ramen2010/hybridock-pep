"""E9b — rigorous analysis of MD-ensemble MM-GBSA+IE (does sampling break the wall?).

Tests the e9 outputs the same honest way as everything else:
  - cross-family (family-mean, length-residualized, permutation p)
  - within-family (pooled within-binding-group correlation)
  - compares ⟨E_int⟩ (ensemble enthalpy), dg_pred (+IE), and the IE term alone
  - vs the STATIC single-pose MM-GBSA baseline (data/benchmark_crystal_scored_baseline.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]


def kmer_groups(seqs, th=0.3, k=3):
    ks = [{s[i:i+k] for i in range(max(0, len(s)-k+1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - th).fit_predict(D)


def resid(x, z):
    z = np.asarray(z, float)
    if np.std(z) == 0:
        return x - x.mean()
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def perm_p(V, Y, L, n=20000, seed=0):
    vr, yr = resid(V, L), resid(Y, L)
    r = pearsonr(vr, yr).statistic
    rng = np.random.default_rng(seed)
    c = sum(abs(pearsonr(vr, resid(Y[rng.permutation(len(Y))], L)).statistic) >= abs(r)
            for _ in range(n))
    return r, (c + 1) / (n + 1)


def main():
    out = json.loads(Path("/tmp/e9_results.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    seqmap = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    for o in out:
        o["seq"] = seqmap.get(o["pdb"].upper(), "X")
    out = [o for o in out if np.isfinite(o.get("dg_pred", np.nan)) and o["seq"] != "X"]
    print(f"E9b: {len(out)} complexes with finite MD-ensemble results")

    seqs = [o["seq"] for o in out]
    g = kmer_groups(seqs, 0.3)
    y = np.array([o["y"] for o in out])
    L = np.array([o["L"] for o in out])
    kd = np.array([o["aff"] == "Kd" for o in out])

    feats = ["e_int_mean", "minus_tds_ie", "dg_pred"]

    def fammean(mask, key):
        idx = np.where(mask)[0]
        gg = g[idx]
        d = {}
        for j, gi in enumerate(gg):
            d.setdefault(gi, []).append(idx[j])
        ks = sorted(d)
        Y = np.array([np.mean([y[i] for i in d[k]]) for k in ks])
        Lm = np.array([np.mean([L[i] for i in d[k]]) for k in ks])
        V = np.array([np.mean([out[i][key] for i in d[k]]) for k in ks])
        return Y, Lm, V

    for label, mask in [("ALL", np.ones(len(out), bool)), ("Kd", kd)]:
        print(f"\n=== {label} (n={int(mask.sum())}) ===")
        # raw (no family collapse)
        yy, LL = y[mask], L[mask]
        print("  RAW pearson (no controls):")
        for f in feats:
            v = np.array([o[f] for o, m in zip(out, mask) if m])
            print(f"    {f:<16} r={pearsonr(v, yy).statistic:+.3f}")
        print("  CROSS-FAMILY (family-mean, length-residualized, perm p):")
        for f in feats:
            Y, Lm, V = fammean(mask, f)
            if len(Y) >= 6 and np.std(V) > 0:
                r, p = perm_p(V, Y, Lm)
                flag = "  <== SIG correct-sign" if (p < 0.05 and r < 0) else (
                    "  <== SIG BACKWARDS" if p < 0.05 else "")
                print(f"    {f:<16} r={r:+.3f}  p={p:.4f}  ({len(Y)} fam){flag}")

    # within-family (pooled within-group, only groups with n>=4)
    print("\n=== WITHIN-FAMILY (variant ranking regime) ===")
    d = {}
    for j, gi in enumerate(g):
        d.setdefault(gi, []).append(j)
    for f in feats:
        wr = []
        for k, idxs in d.items():
            if len(idxs) >= 4:
                v = np.array([out[i][f] for i in idxs])
                yv = np.array([y[i] for i in idxs])
                if np.std(v) > 0:
                    wr.append(pearsonr(v, yv).statistic)
        if wr:
            print(f"  {f:<16} mean within-family r={np.mean(wr):+.3f} "
                  f"(median {np.median(wr):+.3f}, {len(wr)} groups n>=4)")


if __name__ == "__main__":
    main()
