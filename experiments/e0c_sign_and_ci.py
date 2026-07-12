"""E0c — is the nis+vina win HONEST physics or the backwards-Vina artifact?

Three checks:
  1. SIGN: fit on the full set, print standardized coefficients. Physically,
     more-negative Vina => more-negative ΔG => POSITIVE weight on vina. The
     verdict's artifact is a NEGATIVE vina weight (backwards slope).
  2. HONEST variant: constrain vina weight >= 0 (physically correct sign) via
     non-negative ridge on the sign-flipped feature; see if grouped-oof holds.
  3. CI + robustness: bootstrap the grouped-oof Pearson, and vary the family
     threshold, to put error bars on the 0.55.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]


def kmer_set(seq, k=3):
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


def jaccard_groups(seqs, threshold=0.3):
    n = len(seqs)
    ks = [kmer_set(s) for s in seqs]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - threshold,
    ).fit_predict(D)


def load():
    rows = json.loads(Path("/tmp/e0_features.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    seqs = [sm.get(r["pdb"].upper(), "X") for r in rows]
    return rows, np.array(seqs)


def col(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], float)


def grouped_oof_ridge(X, y, g, alpha=1.0, k=5):
    gkf = GroupKFold(n_splits=min(k, len(np.unique(g))))
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return pred


def main():
    rows, seqs = load()
    feats = ["nis_c_frac", "nis_p_frac", "vina"]
    kd = np.array([r["aff"] == "Kd" for r in rows])

    for name, mask in [("ALL", np.ones(len(rows), bool)), ("Kd-only", kd)]:
        y = col(rows, "y")[mask]
        X = np.column_stack([col(rows, f)[mask] for f in feats])
        ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        seqm = seqs[mask][ok]
        g = jaccard_groups(list(seqm), 0.3)

        # ---- 1. SIGN of standardized coefficients (full-set fit) ----
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=1.0).fit((X - mu) / sd, y)
        print(f"\n=== {name} (n={len(y)}, fam={len(np.unique(g))}) ===")
        print("  standardized ridge weights:")
        for f, w in zip(feats, m.coef_):
            note = ""
            if f == "vina":
                note = "  (NEG = backwards artifact; POS = physical)"
            print(f"    {f:<14}{w:+.3f}{note}")

        # ---- 2. HONEST variant: force physically-correct vina sign ----
        # physical: positive weight on vina (more negative vina -> lower dG).
        # Use NNLS on standardized features with vina kept, nis free via +/- split.
        Xs = (X - mu) / sd
        # build design where every feature has a non-negative coeff by splitting
        # nis features into +/- columns, but vina ONLY allowed its physical (+) dir
        cols, labels = [], []
        # vina physical direction: want coef>=0 on vina meaning predict more
        # negative dG for more negative vina -> include +vina, target y.
        cols.append(Xs[:, 2]); labels.append("vina(+only)")
        for idx, fn in [(0, "nis_c"), (1, "nis_p")]:
            cols.append(Xs[:, idx]); labels.append(f"+{fn}")
            cols.append(-Xs[:, idx]); labels.append(f"-{fn}")
        A = np.column_stack(cols)
        # center y for nnls intercept handling
        yc = y - y.mean()
        w, _ = nnls(A, yc)
        print("  HONEST (vina sign-constrained physical) weights:")
        for lb, wi in zip(labels, w):
            if wi > 1e-6:
                print(f"    {lb:<14}{wi:+.3f}")
        # grouped-oof of the honest model
        def honest_oof():
            gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
            pred = np.zeros_like(y)
            for tr, te in gkf.split(X, y, g):
                mu2, sd2 = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xtr = (X[tr] - mu2) / sd2
                Xte = (X[te] - mu2) / sd2
                cset = lambda M: np.column_stack(
                    [M[:, 2], M[:, 0], -M[:, 0], M[:, 1], -M[:, 1]]
                )
                wtr, _ = nnls(cset(Xtr), y[tr] - y[tr].mean())
                pred[te] = cset(Xte) @ wtr + y[tr].mean()
            return pred
        ph = honest_oof()
        print(f"  HONEST grouped-oof r = {pearsonr(ph, y).statistic:.3f}")

        # ---- 3. bootstrap CI + threshold robustness on the free model ----
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(300):
            idx = rng.integers(0, len(y), len(y))
            try:
                p = grouped_oof_ridge(X[idx], y[idx], jaccard_groups(list(seqm[idx]), 0.3))
                boots.append(pearsonr(p, y[idx]).statistic)
            except Exception:
                pass
        boots = np.array(boots)
        base_p = grouped_oof_ridge(X, y, g)
        base_r = pearsonr(base_p, y).statistic
        print(f"  free nis+vina grouped-oof r = {base_r:.3f}  "
              f"[boot 95% CI {np.percentile(boots,2.5):.2f}, {np.percentile(boots,97.5):.2f}]")
        for th in (0.2, 0.4, 0.5):
            gg = jaccard_groups(list(seqm), th)
            p = grouped_oof_ridge(X, y, gg)
            print(f"    threshold={th}: fam={len(np.unique(gg))}  "
                  f"oof_r={pearsonr(p,y).statistic:.3f}")


if __name__ == "__main__":
    main()
