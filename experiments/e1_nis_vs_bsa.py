"""E1 — does NIS add signal ORTHOGONAL to BSA, or is it the same wall?

Per SCORING_FINDINGS_HANDOFF.md: BSA is the best correctly-signed affinity
feature (size-controlled r ~ +0.28; held +0.35), and combining interface terms
HURTS because they're collinear. So the only question that matters for NIS:

    does NIS + BSA beat BSA-alone under FAMILY-GROUPED held-out CV,
    with both features entering at their physically-correct sign?

All features oriented a-priori (never sign-fit on this data):
  BSA       -> stronger binding  (corr with dG should be +, bigger buried = ... )
              NOTE: dG negative=strong, so "more BSA -> more negative dG" = NEG corr.
              We orient so the oriented column is POSITIVELY signed for NNLS.
  nis_p_frac (polar non-interface frac)   : E0 sign, more -> stronger (neg corr w/ dG)
  nis_c_frac (charged non-interface frac) : E0 sign, more -> weaker  (pos corr w/ dG)

Outputs:
  - raw, spearman, size-controlled (partial|L) for BSA & NIS
  - collinearity corr(NIS_signal, BSA)
  - grouped-oof Pearson for: BSA / NIS / NIS+BSA  (NNLS, correct-sign)
  - bootstrap CI on the delta (NIS+BSA) - (BSA)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold

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
        distance_threshold=1.0 - threshold,
    ).fit_predict(D)


def residualize(x, z):
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def grouped_oof(cols, y, g):
    X = np.column_stack(cols)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, g):
        mu = y[tr].mean()
        w, _ = nnls(X[tr], y[tr] - mu)
        pred[te] = X[te] @ w + mu
    return pearsonr(pred, y).statistic, pred


def main():
    rows = json.loads(Path("/tmp/e0_features.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    seqs = np.array([sm.get(r["pdb"].upper(), "X") for r in rows])

    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], float)

    y_all, L_all = col("y"), col("L")
    bsa_all = col("bsa")
    nis_p_all, nis_c_all = col("nis_p_frac"), col("nis_c_frac")
    kd = np.array([r["aff"] == "Kd" for r in rows])

    for label, mask in [("ALL (Kd+Ki)", np.ones(len(rows), bool)),
                        ("Kd-only", kd)]:
        y, L = y_all[mask], L_all[mask]
        bsa = bsa_all[mask]
        nis_p, nis_c = nis_p_all[mask], nis_c_all[mask]
        ok = np.isfinite(y) & np.isfinite(bsa) & np.isfinite(nis_p)
        y, L, bsa, nis_p, nis_c = y[ok], L[ok], bsa[ok], nis_p[ok], nis_c[ok]
        seqm = list(seqs[mask][ok])
        g = kmer_groups(seqm, 0.3)

        print(f"\n=== {label}  (n={len(y)}, fam={len(np.unique(g))}) ===")

        # --- descriptive: raw / spearman / size-controlled ---
        def desc(name, v):
            raw = pearsonr(v, y).statistic
            sp = spearmanr(v, y).statistic
            pl = pearsonr(residualize(v, L), residualize(y, L)).statistic
            print(f"  {name:<14} raw={raw:+.3f}  spear={sp:+.3f}  partial|L={pl:+.3f}")
        desc("BSA", bsa)
        desc("nis_p_frac", nis_p)
        desc("nis_c_frac", nis_c)

        # --- collinearity: is NIS just BSA? ---
        print(f"  collinearity corr(nis_p, BSA) = {pearsonr(nis_p, bsa).statistic:+.3f}")
        print(f"  collinearity corr(nis_c, BSA) = {pearsonr(nis_c, bsa).statistic:+.3f}")

        # --- oriented columns for NNLS (coeff>=0, physical sign) ---
        def z(v):
            return (v - v.mean()) / (v.std() + 1e-9)
        # dG: negative=strong. orient each so "more of it -> more negative dG"
        # => oriented column should be NEGATIVELY correlated... we feed NNLS the
        # column whose POSITIVE coeff reduces dG. Easiest: orient by sign of raw r.
        def orient(v):
            s = np.sign(pearsonr(v, y).statistic) or 1.0
            return s * z(v)  # now positively correlated with y; NNLS coeff>=0
        bsa_o = orient(bsa)
        nisp_o = orient(nis_p)
        nisc_o = orient(nis_c)

        r_bsa, _ = grouped_oof([bsa_o], y, g)
        r_nis, _ = grouped_oof([nisp_o, nisc_o], y, g)
        r_both, _ = grouped_oof([bsa_o, nisp_o, nisc_o], y, g)
        print(f"  grouped-oof  BSA      = {r_bsa:+.3f}")
        print(f"  grouped-oof  NIS      = {r_nis:+.3f}")
        print(f"  grouped-oof  NIS+BSA  = {r_both:+.3f}   (delta vs BSA = {r_both-r_bsa:+.3f})")

        # --- bootstrap CI on the delta ---
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(400):
            idx = rng.integers(0, len(y), len(y))
            try:
                gg = kmer_groups([seqm[i] for i in idx], 0.3)
                rb, _ = grouped_oof([bsa_o[idx]], y[idx], gg)
                rcomb, _ = grouped_oof([bsa_o[idx], nisp_o[idx], nisc_o[idx]], y[idx], gg)
                deltas.append(rcomb - rb)
            except Exception:
                pass
        deltas = np.array(deltas)
        print(f"  delta 95% CI = [{np.percentile(deltas,2.5):+.3f}, "
              f"{np.percentile(deltas,97.5):+.3f}]  "
              f"(P(delta>0)={np.mean(deltas>0):.2f})")


if __name__ == "__main__":
    main()
