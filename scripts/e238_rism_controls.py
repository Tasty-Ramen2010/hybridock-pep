"""E238 — controls for the E230 3D-RISM "breaks the wall" claim (r=+0.428, n=44, plain LOO Ridge).

Three things the bare LOO number does NOT rule out, each a documented past-mirage shape on this project:

  (1) PERMUTATION NULL  — at n=44 with 5 features, how often does *shuffled* y clear the 0.30 bar /
      reach 0.428 by chance? Gives an honest p-value for the headline.
  (2) FAMILY-GROUPED CV — single-linkage cluster receptors by 5-mer Jaccard sequence identity, then
      leave-one-CLUSTER-out. If r collapses, the 0.428 was near-duplicate-receptor leakage.
  (3) ESM ORTHOGONALITY — on the 19 receptors with ESM2-150M embeddings: does RISM beat ESM, and does
      RISM still predict the part of the baseline ESM CANNOT (LOO residuals)? If it collapses on the
      residuals, RISM is re-encoding sequence, not adding hydration physics.

Run in score-env:  python3 scripts/e238_rism_controls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RISM = ROOT / "data" / "e230_rism.jsonl"
ESM = ROOT / "data" / "e231_pilot_esm.npz"
MANIFEST = ROOT / "data" / "e228_pilot_manifest.json"
FEATS = ["n_pocket", "n_sites", "max_g", "mean_g", "exchem"]
RNG = np.random.default_rng(0)


def loo_ridge_r(X, y, alpha=2.0):
    """Leave-one-out Ridge, return Pearson r of OOF predictions vs y."""
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        pred[i] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i + 1]))[0]
    return float(np.corrcoef(pred, y)[0, 1]), pred


def grouped_cv_r(X, y, groups, alpha=2.0):
    """Leave-one-group-out OOF predictions -> Pearson r."""
    n = len(y)
    pred = np.empty(n)
    for gid in np.unique(groups):
        te = groups == gid
        tr = ~te
        if tr.sum() < 3:
            pred[te] = y[tr].mean() if tr.sum() else y.mean()
            continue
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return float(np.corrcoef(pred, y)[0, 1]), pred


def kmer_set(seq, k=5):
    seq = seq.replace("/", "")
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def cluster_by_identity(seqs, thr=0.30):
    """Single-linkage clusters: edge if 5-mer Jaccard >= thr. Catches near-duplicate receptors."""
    n = len(seqs)
    ks = [kmer_set(s) for s in seqs]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            u = len(ks[i] | ks[j]) or 1
            if len(ks[i] & ks[j]) / u >= thr:
                parent[find(i)] = find(j)
    remap, out = {}, []
    for i in range(n):
        r = find(i)
        out.append(remap.setdefault(r, len(remap)))
    return np.array(out)


def main():
    rows = [json.loads(l) for l in RISM.read_text().splitlines()]
    pdbs = [r["rep_pdb"] for r in rows]
    y = np.array([r["y_mean"] for r in rows])
    X = np.array([[r.get(f, np.nan) for f in FEATS] for r in rows], float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    n = len(rows)

    man = {r["peptides"][0]["pdb"]: r["rec_seq"] for r in json.load(open(MANIFEST))["receptors"]}
    seqs = [man.get(p, "") for p in pdbs]

    print(f"=== E238 controls on E230 3D-RISM (n={n}, baseline std={y.std():.2f}) ===\n")

    # --- headline reproduce ---
    r_obs, _ = loo_ridge_r(X, y)
    print(f"[headline]  LOO-Ridge r = {r_obs:+.3f}   (bar 0.30)\n")

    # --- (1) permutation null ---
    NPERM = 2000
    null = np.empty(NPERM)
    for b in range(NPERM):
        null[b], _ = loo_ridge_r(X, RNG.permutation(y))
    p_ge_obs = (null >= r_obs).mean()
    p_ge_bar = (null >= 0.30).mean()
    print(f"[1 perm null] {NPERM} shuffles:")
    print(f"    null r mean={null.mean():+.3f} std={null.std():.3f} 95th pct={np.percentile(null,95):+.3f}")
    print(f"    P(null >= observed {r_obs:.3f}) = {p_ge_obs:.4f}   <- the honest p-value")
    print(f"    P(null >= 0.30 bar)            = {p_ge_bar:.4f}   <- how often noise 'clears the bar'")
    print(f"    VERDICT: {'SURVIVES (p<0.05)' if p_ge_obs < 0.05 else 'NOT SIGNIFICANT — likely small-n mirage'}\n")

    # --- (2) family-grouped CV ---
    for thr in (0.20, 0.30, 0.40):
        groups = cluster_by_identity(seqs, thr)
        ng = len(np.unique(groups))
        r_g, _ = grouped_cv_r(X, y, groups)
        biggest = np.bincount(groups).max()
        print(f"[2 grouped@{thr:.2f}] {ng} clusters (largest={biggest}): leave-cluster-out r = {r_g:+.3f}")
    print(f"    -> if these track {r_obs:.2f}, not leakage; if they collapse toward 0, the LOO was duplicate-leakage\n")

    # --- (3) ESM orthogonality ---
    d = np.load(ESM, allow_pickle=True)
    epdbs = list(d["pdbs"])
    emb = {p: d["emb"][i] for i, p in enumerate(epdbs)}
    idx = [i for i, p in enumerate(pdbs) if p in emb]
    if len(idx) < 8:
        print(f"[3 ESM] only {len(idx)} overlap — skip"); return
    Xr = X[idx]
    E = np.array([emb[pdbs[i]] for i in idx], float)
    ysub = y[idx]
    print(f"[3 ESM orthogonality] n={len(idx)} receptors with ESM2-150M")
    r_rism_sub, _ = loo_ridge_r(Xr, ysub)
    r_esm, esm_pred = loo_ridge_r(E, ysub, alpha=20.0)
    print(f"    RISM-only (this subset) r = {r_rism_sub:+.3f}")
    print(f"    ESM-only             r = {r_esm:+.3f}")
    resid = ysub - esm_pred  # part of baseline ESM cannot explain
    r_resid, _ = loo_ridge_r(Xr, resid)
    print(f"    RISM -> ESM residual r = {r_resid:+.3f}   <- the orthogonal-information test")
    print(f"    VERDICT: {'RISM adds hydration signal BEYOND sequence' if r_resid > 0.20 else 'RISM mostly RE-ENCODES sequence — not new physics'}")


if __name__ == "__main__":
    main()
