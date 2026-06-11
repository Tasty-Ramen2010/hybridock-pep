"""E39 — free-state conformational entropy feature (the term invisible to a bound static pose).

Physics: a peptide LOSES conformational entropy when it freezes on binding → +TΔS penalty
(unfavorable). The penalty per residue = (free-state entropy) × (how ordered it becomes).
Pre-rigid peptides (poly-Pro, disulfide, β-branched) pay LITTLE; floppy ones (Gly-rich,
disordered) pay a lot. This is a FREE-STATE / SEQUENCE property — not in the bound pose.

Cheap sequence-derived estimators (instant):
  sc_entropy   : Σ_i n_chi(aa_i)        side-chain rotamer entropy lost (∝ #χ angles)
  sc_ent_bur   : sc_entropy × (buried_fraction)  — only frozen residues pay
  bb_rigidity  : Pro_frac − Gly_frac    backbone pre-organization (Pro rigid, Gly floppy)
  flex_mean    : mean Bhaskaran-Ponnuswamy flexibility (high = floppy = more penalty)
  disorder     : mean TOP-IDP disorder propensity

Tests whether adding free-state entropy to the universal intensive features improves
cross-dataset GENERALIZATION (cr<->98) — the term that should fix length without using length.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Number of side-chain rotatable (χ) angles per residue ≈ conformational entropy lost on freezing.
N_CHI = {"A": 0, "G": 0, "P": 0, "S": 1, "C": 1, "T": 1, "V": 1, "D": 2, "N": 2, "L": 2,
         "I": 2, "F": 2, "Y": 2, "H": 2, "W": 2, "M": 3, "E": 3, "Q": 3, "K": 4, "R": 4}
# Bhaskaran-Ponnuswamy flexibility (high = flexible). Approx normalized.
FLEX = {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.50,
        "G": 0.54, "H": 0.32, "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.30, "F": 0.31,
        "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31, "Y": 0.42, "V": 0.39}
# TOP-IDP disorder propensity (Campen 2008); high = disorder-promoting.
TOPIDP = {"A": 0.06, "R": 0.18, "N": 0.01, "D": 0.19, "C": 0.02, "Q": 0.32, "E": 0.74,
          "G": 0.17, "H": 0.30, "I": -0.49, "L": -0.33, "K": 0.59, "M": -0.40, "F": -0.70,
          "P": 0.99, "S": 0.34, "T": 0.06, "W": -0.88, "Y": -0.30, "V": -0.54}


def seq_entropy_features(seq, buried_frac=1.0):
    if not seq:
        return {}
    L = len(seq)
    sc = sum(N_CHI.get(a, 2) for a in seq)
    return dict(
        sc_entropy=sc / L,                                   # mean side-chain entropy (intensive)
        sc_ent_total=sc,                                     # total (extensive — pays more if longer+floppy)
        sc_ent_bur=sc / L * buried_frac,                     # entropy that actually freezes
        bb_rigidity=(seq.count("P") - seq.count("G")) / L,   # Pro rigid(+), Gly floppy(−)
        flex_mean=float(np.mean([FLEX.get(a, 0.45) for a in seq])),
        disorder=float(np.mean([TOPIDP.get(a, 0.0) for a in seq])),
        pro_frac=seq.count("P") / L,
    )


def main():
    inten = json.loads(Path("/tmp/e31_intensive.json").read_text())
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    cr_rows = json.loads(Path("/tmp/e19_cr.json").read_text())
    seqs98 = json.loads(Path("/tmp/e39_seqs98.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())

    # crystal-65: seq + buried fraction (use f_hyd_iface as buried proxy)
    cr = []
    for r, it in zip(cr_rows, inten["cr"]):
        seq = bench[r["pdb"].upper()]["peptide_seq"]
        bf = min(1.0, it.get("f_hyd_iface", 0.5))
        cr.append(dict(it, **seq_entropy_features(seq, bf)))
    b98 = []
    for (k, v), it in zip(e28.items(), inten["b98"]):
        seq = seqs98.get(k, "")
        bf = min(1.0, it.get("f_hyd_iface", 0.5))
        b98.append(dict(it, **seq_entropy_features(seq, bf)))
    ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])

    UNI = ["bsa_hyd", "mj_per_contact", "f_hyd_iface", "frac_pol_satisfied"]
    ENT = ["sc_entropy", "sc_ent_total", "sc_ent_bur", "bb_rigidity", "flex_mean", "disorder", "pro_frac"]

    print("=== free-state entropy features: sign-consistency (cr vs 98) ===")
    print(f"  {'feature':<14}{'crystal-65':>12}{'the-98':>10}{'universal?':>12}")
    keep = []
    for f in ENT:
        rc = pearsonr([r[f] for r in cr], ycr).statistic
        r9 = pearsonr([r[f] for r in b98], y98).statistic
        ok = rc * r9 > 0 and min(abs(rc), abs(r9)) > 0.1
        if ok:
            keep.append(f)
        print(f"  {f:<14}{rc:>+12.3f}{r9:>+10.3f}{('YES' if ok else 'flip/weak'):>12}")
    print(f"  universal entropy features: {keep}")

    def mat(rows, feats):
        return np.array([[r.get(f, 0.0) for f in feats] for r in rows])

    def transfer(tr, te, feats):
        Xtr, ytr = mat(tr, feats), np.array([r["y"] for r in tr])
        Xte, yte = mat(te, feats), np.array([r["y"] for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return pearsonr(np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w, yte).statistic

    def loo(rows, feats):
        y = np.array([r["y"] for r in rows]); X = mat(rows, feats); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())

    pool = cr + b98
    print("\n=== does free-state entropy improve GENERALIZATION? (transfer + pool LOO) ===")
    print(f"  {'feature set':<40}{'cr->98':>9}{'98->cr':>9}{'pool LOO':>10}")
    sets = {"universal intensive (4) [baseline]": UNI}
    if keep:
        sets["+ universal entropy"] = UNI + keep
    sets["+ all entropy"] = UNI + ENT
    sets["+ sc_ent_bur only"] = UNI + ["sc_ent_bur"]
    sets["+ bb_rigidity only"] = UNI + ["bb_rigidity"]
    sets["+ disorder only"] = UNI + ["disorder"]
    for nm, fs in sets.items():
        a = transfer(cr, b98, fs); b = transfer(b98, cr, fs); pr, _ = loo(pool, fs)
        print(f"  {nm:<40}{a:>+9.3f}{b:>+9.3f}{pr:>+10.3f}")
    print(f"\n  baseline transfer +0.24/+0.37, pool 0.421. >> if entropy lifts transfer, it's the missing term")


if __name__ == "__main__":
    main()
