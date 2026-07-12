"""E0b — does the NIS signal survive FAMILY-grouped held-out CV?

E0 found NIS composition (nis_c_frac, nis_p_frac) carries ΔG signal that
survives length-residualization. But the crystal set has near-duplicate
co-crystal families (e.g. 30x the same receptor). If a family has both a
characteristic NIS composition AND a characteristic ΔG, an ungrouped fit
leaks the label — exactly what killed v1.3/v1.4/per-family.

This script:
  1. clusters peptides into families by k-mer Jaccard (same as per_family),
  2. runs GroupKFold ridge so no family is split across train/test,
  3. reports OUT-OF-FOLD Pearson r + sign + RMSE for several models.

A model only counts if out-of-fold r >= 0.5 with the physically correct sign
and is stable across grouping granularity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]


def kmer_set(seq: str, k: int = 3) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


def jaccard_groups(seqs: list[str], threshold: float = 0.3) -> np.ndarray:
    n = len(seqs)
    ksets = [kmer_set(s) for s in seqs]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ksets[i], ksets[j]
            u = len(a | b)
            sim = len(a & b) / u if u else 0.0
            D[i, j] = D[j, i] = 1.0 - sim
    cl = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - threshold,
    )
    return cl.fit_predict(D)


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float = 1.0):
    n_groups = len(np.unique(groups))
    k = min(5, n_groups)
    gkf = GroupKFold(n_splits=k)
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    r = pearsonr(pred, y).statistic
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return r, rmse


def main() -> None:
    rows = json.loads(Path("/tmp/e0_features.json").read_text())
    seqs = []
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    seqmap = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    for r in rows:
        seqs.append(seqmap.get(r["pdb"].upper(), "X"))

    groups = jaccard_groups(seqs, threshold=0.3)
    print(f"n={len(rows)}  families={len(np.unique(groups))}")

    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], float)

    y = col("y")

    feature_sets = {
        "vina (artifact baseline)": ["vina"],
        "nis_c+nis_p": ["nis_c_frac", "nis_p_frac"],
        "nis + length": ["nis_c_frac", "nis_p_frac", "L"],
        "nis + bsa": ["nis_c_frac", "nis_p_frac", "bsa"],
        "nis + ic_charged_frac": ["nis_c_frac", "nis_p_frac", "ic_charged_frac"],
        "nis + vina": ["nis_c_frac", "nis_p_frac", "vina"],
        "full physics": ["vina", "dh", "n_contact", "bsa",
                          "ic_charged_frac", "ic_apolar_frac",
                          "nis_c_frac", "nis_p_frac"],
    }

    for split_name, mask in [
        ("ALL (Kd+Ki)", np.ones(len(rows), bool)),
        ("Kd only", np.array([r["aff"] == "Kd" for r in rows])),
    ]:
        print(f"\n=== {split_name}  (n={int(mask.sum())}, "
              f"families={len(np.unique(groups[mask]))}) ===")
        print(f"{'model':<26}{'oof_r':>9}{'rmse':>8}")
        ym = y[mask]
        gm = groups[mask]
        for name, feats in feature_sets.items():
            X = np.column_stack([col(f)[mask] for f in feats])
            ok = np.all(np.isfinite(X), 1) & np.isfinite(ym)
            if ok.sum() < 10:
                continue
            r, rmse = grouped_oof(X[ok], ym[ok], gm[ok])
            flag = "  <==" if r >= 0.5 else ""
            print(f"{name:<26}{r:>9.3f}{rmse:>8.2f}{flag}")


if __name__ == "__main__":
    main()
