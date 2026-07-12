"""E21b — attack the shared blind spot (helices) + light ESM ensemble.

Findings from e21: we & Vina BOTH fail on amphipathic helices; we have no SS term.
Tests:
  1. Add a helicity term: (a) sequence helical-propensity, (b) actual backbone phi/psi
     helix fraction from the pose. Does it fix the worst (helical) complexes?
  2. ESM as LIGHT SECONDARY ensemble member (z-blend, NOT feature concat which overfit).
  3. Best combined: geometry + helix, ensembled with Vina (+ optional light ESM).
All LOO-honest. crystal-65 oracle poses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEO = POCK + IFACE

# Pace-Scholtz helix propensity (kcal/mol; lower = more helix-favorable). Use -value so higher=more helical.
HELIX_PROP = {"A": 0.00, "L": 0.21, "R": 0.21, "M": 0.24, "K": 0.26, "Q": 0.39, "E": 0.40,
              "I": 0.41, "W": 0.49, "S": 0.50, "Y": 0.53, "F": 0.54, "H": 0.61, "V": 0.61,
              "N": 0.65, "T": 0.66, "C": 0.68, "D": 0.69, "G": 1.00, "P": 3.16}


def seq_helicity(seq):
    """Mean helix favorability (higher = more helical) + amphipathic moment proxy."""
    if not seq:
        return 0.0, 0.0
    h = np.mean([-HELIX_PROP.get(a, 0.5) for a in seq])
    # hydrophobic moment (i, i+3.6 periodicity) — amphipathicity
    KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,
          "I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    ang = 100 * np.pi / 180
    mx = sum(KD.get(a,0)*np.cos(i*ang) for i,a in enumerate(seq))
    my = sum(KD.get(a,0)*np.sin(i*ang) for i,a in enumerate(seq))
    moment = np.sqrt(mx*mx+my*my)/len(seq)
    return h, moment


def loo_pred(X, y):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pred


def rr(p, y):
    return pearsonr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean()))


def main():
    cr = json.loads(Path("/tmp/e19_cr.json").read_text())
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    esm = json.loads(Path("/tmp/esm_affinity.json").read_text()) if Path("/tmp/esm_affinity.json").exists() else {}
    rows = []
    for r in cr:
        b = bench.get(r["pdb"].upper())
        if not b:
            continue
        h, mom = seq_helicity(r["seq"])
        r = dict(r); r["vina"] = b["vina_docked"]; r["helix"] = h; r["amphi"] = mom
        rows.append(r)
    y = np.array([r["y"] for r in rows])
    Xg = np.array([[r.get(f, 0.0) for f in GEO] for r in rows], float)
    vina = np.array([r["vina"] for r in rows], float)
    helix = np.array([r["helix"] for r in rows])
    amphi = np.array([r["amphi"] for r in rows])

    print("=== raw correlations of new terms ===")
    for nm, v in [("helix_prop", helix), ("amphi_moment", amphi)]:
        print(f"  corr({nm}, ΔG) = {pearsonr(v,y).statistic:+.3f}")

    print("\n=== LOO models (r / RMSE) ===")
    tests = {
        "geometry (ours)": Xg,
        "+ helix": np.column_stack([Xg, helix]),
        "+ amphi moment": np.column_stack([Xg, amphi]),
        "+ helix + amphi": np.column_stack([Xg, helix, amphi]),
    }
    preds = {}
    for nm, X in tests.items():
        p = loo_pred(X, y); preds[nm] = p
        r, rmse = rr(p, y)
        print(f"  {nm:<22}{r:+.3f}  RMSE {rmse:.2f}")

    pred_vina = loo_pred(vina.reshape(-1, 1), y)
    # ESM light: PCA3 ridge LOO as a weak member
    pred_esm = None
    if esm and all(r["seq"] in esm for r in rows):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        Xe = np.array([esm[r["seq"]] for r in rows])
        pe = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            pca = PCA(n_components=5).fit(Xe[tr]); m = Ridge(alpha=10).fit(pca.transform(Xe[tr]), y[tr])
            pe[i] = m.predict(pca.transform(Xe[i:i+1]))[0]
        pred_esm = pe
        print(f"  ESM-PCA5 ridge (weak)  {pearsonr(pe,y).statistic:+.3f}")

    print("\n=== ENSEMBLES (z-blend of LOO predictors) ===")
    def z(p): return (p - p.mean()) / p.std()
    best = preds["+ helix + amphi"]
    zb, zv = z(best), z(pred_vina)
    for w in (0.4, 0.5, 0.6, 0.7):
        ze = w * zb + (1 - w) * zv
        print(f"  {w:.1f}*(geo+helix+amphi) + {1-w:.1f}*vina   r={pearsonr(ze,y).statistic:+.3f}")
    if pred_esm is not None:
        ze2 = 0.45 * zb + 0.45 * zv + 0.10 * z(pred_esm)
        print(f"  0.45*geo+helix + 0.45*vina + 0.10*ESM(light)  r={pearsonr(ze2,y).statistic:+.3f}")
        ze3 = 0.5 * zb + 0.35 * zv + 0.15 * z(pred_esm)
        print(f"  0.50*geo+helix + 0.35*vina + 0.15*ESM(light)  r={pearsonr(ze3,y).statistic:+.3f}")
    print(f"\n  guess-mean RMSE={y.std():.2f}  | targets: ours 0.576, vina 0.527, lit 0.62")


if __name__ == "__main__":
    main()
