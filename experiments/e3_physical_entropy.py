"""E3 — does PHYSICAL per-residue conformational entropy carry cross-family signal?

User's thesis: peptide ΔG = ΔH − TΔS, and using *correct per-residue physical*
entropy constants (not a single fitted α) may separate binders where size-confounded
features cannot. The orthogonality bet: two peptides of equal LENGTH but different
COMPOSITION lose different entropy on binding (Gly/Ala-rich lose less than
Lys/Arg/Gln-rich). That composition signal is potentially orthogonal to interface
size — the confound that walls every other feature.

Published note (Zhang & Liu, PLoS Comp Biol 2006): side-chain conf. entropy scales
~linearly with chain length → the RAW sum is size; we therefore test the
length-residualized / per-residue COMPOSITION form, and the only honest metric is
ONE-PER-FAMILY (cross-family) Pearson, bootstrap-CI'd.

Side-chain conformational entropy scale TΔS_sc (kcal/mol, 298 K, loss on full
burial) — widely-tabulated Abagyan–Totrov / Doig–Sternberg values. We also test a
second (flexibility-rank) scale for robustness; if both give ~0 one-per-family,
the null is scale-independent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]

# Side-chain conformational entropy TΔS_sc, kcal/mol @298K (Abagyan-Totrov/Doig-
# Sternberg compilation). Higher = more entropy lost when the side chain is buried.
TDS_SC = {
    "A": 0.00, "R": 2.13, "N": 0.81, "D": 0.61, "C": 0.55, "Q": 2.02, "E": 1.65,
    "G": 0.00, "H": 0.99, "I": 0.75, "L": 0.75, "K": 2.21, "M": 1.53, "F": 0.76,
    "P": 0.00, "S": 0.55, "T": 0.48, "W": 0.97, "Y": 0.99, "V": 0.50,
}
# Backbone/main-chain entropy loss TΔS_bb (kcal/mol) — D'Aquino 1996 style; Gly
# most flexible (largest loss), Pro least (locked). Coarse but ordering is robust.
TDS_BB = {aa: 1.0 for aa in TDS_SC}
TDS_BB["G"] = 1.6
TDS_BB["P"] = 0.2
# Robustness scale 2: number of side-chain rotatable bonds (pure geometry).
N_CHI = {
    "A": 0, "R": 4, "N": 2, "D": 2, "C": 1, "Q": 3, "E": 3, "G": 0, "H": 2,
    "I": 2, "L": 2, "K": 4, "M": 3, "F": 2, "P": 0, "S": 1, "T": 1, "W": 2,
    "Y": 2, "V": 1,
}


def seq_entropy_features(seq: str) -> dict:
    seq = seq.upper()
    L = max(1, len(seq))
    ent_sc = sum(TDS_SC.get(a, 1.0) for a in seq)
    ent_bb = sum(TDS_BB.get(a, 1.0) for a in seq)
    ent_tot = ent_sc + ent_bb
    n_chi = sum(N_CHI.get(a, 0) for a in seq)
    return dict(
        ent_sc=ent_sc, ent_bb=ent_bb, ent_tot=ent_tot, ent_chi=float(n_chi),
        ent_sc_per_res=ent_sc / L, ent_tot_per_res=ent_tot / L,
        ent_chi_per_res=n_chi / L,
        # fraction of flexible (high-entropy) residues — pure composition
        frac_flexible=sum(1 for a in seq if N_CHI.get(a, 0) >= 3) / L,
    )


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


def residualize(x, z):
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def one_per_family(y, v, g, seqs, n_boot=2000, seed=0):
    """Cross-family Pearson: one representative per family, bootstrapped over which."""
    fams = {}
    for i, gi in enumerate(g):
        fams.setdefault(gi, []).append(i)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = [rng.choice(members) for members in fams.values()]
        if len(idx) < 4 or np.std(v[idx]) == 0:
            continue
        rs.append(pearsonr(v[idx], y[idx]).statistic)
    rs = np.array(rs)
    return rs.mean(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def main():
    rows = json.loads(Path("/tmp/e3_rows.json").read_text())
    rows = [r for r in rows if r.get("seq")]
    for r in rows:
        r.update(seq_entropy_features(r["seq"]))
    Path("/tmp/e3_features.json").write_text(json.dumps(rows))

    def col(k):
        return np.array([r[k] for r in rows], float)

    y = col("y")
    L = col("L")
    seqs = [r["seq"] for r in rows]
    g = kmer_groups(seqs, 0.3)
    kd = np.array([r["aff"] == "Kd" for r in rows])

    feats = ["ent_sc", "ent_bb", "ent_tot", "ent_chi", "ent_sc_per_res",
             "ent_tot_per_res", "ent_chi_per_res", "frac_flexible"]

    print(f"n={len(rows)} families={len(set(g))}")
    print(f"\n{'feature':<18}{'raw_r':>8}{'spear':>8}{'partial|L':>11}"
          f"{'1perFam_r':>11}{'  95% CI':>20}")
    for split, mask in [("ALL", np.ones(len(rows), bool)), ("Kd", kd)]:
        print(f"-- {split} (n={int(mask.sum())}) --")
        ym, Lm, gm = y[mask], L[mask], g[mask]
        sm = [s for s, mm in zip(seqs, mask) if mm]
        for f in feats:
            v = col(f)[mask]
            if np.std(v) == 0:
                continue
            raw = pearsonr(v, ym).statistic
            sp = spearmanr(v, ym).statistic
            pl = pearsonr(residualize(v, Lm), residualize(ym, Lm)).statistic
            r1, lo, hi = one_per_family(ym, v, gm, sm)
            flag = "  <==" if abs(r1) >= 0.3 and lo * hi > 0 else ""
            print(f"{f:<18}{raw:>8.3f}{sp:>8.3f}{pl:>11.3f}{r1:>11.3f}"
                  f"   [{lo:+.2f},{hi:+.2f}]{flag}")


if __name__ == "__main__":
    main()
