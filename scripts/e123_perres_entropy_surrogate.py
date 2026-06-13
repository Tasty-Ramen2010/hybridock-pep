"""E123 — context-aware per-residue entropy SURROGATE (Ram's entropy model, prototype).

Trains an ML model to predict per-residue free-state dihedral entropy from LOCAL SEQUENCE CONTEXT
(the residue + its neighbors + position), distilling the expensive MD into a fast no-MD surrogate.
This is the foundation of the binding-entropy decomposition: entropy_lost_on_binding = Σ predicted
per-residue entropy over the CONTACTING residues.

Features per residue i (context window ±2):
  - one-hot residue identity (20) at positions i-2..i+2
  - intrinsic flexibility / side-chain config-entropy of i and neighbors
  - terminal flags (N/C-term are floppier), position fraction, peptide length
Target: MD per-residue dihedral entropy (nats). Grouped CV by PEPTIDE (no residue from a test peptide
in training) → honest learnability. Compares to the single-residue-flexibility baseline (e119's proxy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
AAIDX = {a: i for i, a in enumerate(AA)}
FLEX = {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.50, "G": 0.54, "H": 0.32,
        "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.30, "F": 0.31, "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31,
        "Y": 0.42, "V": 0.39}
SCENT = {"A": 0, "G": 0, "P": 0, "S": 3.5, "C": 3.5, "T": 3.5, "V": 1.7, "D": 5, "N": 5, "I": 5, "L": 5.2,
         "E": 7.1, "Q": 7.1, "M": 8, "F": 5.5, "Y": 5.9, "W": 5.9, "H": 6.2, "K": 9, "R": 9.3}
WIN = 2  # context window ±2


def res_features(seq, i):
    L = len(seq)
    f = []
    for off in range(-WIN, WIN + 1):
        j = i + off
        oneh = [0.0] * 20
        flex = 0.45
        scent = 5.0
        if 0 <= j < L and seq[j] in AAIDX:
            oneh[AAIDX[seq[j]]] = 1.0
            flex = FLEX[seq[j]]
            scent = SCENT[seq[j]]
        f += oneh + [flex, scent]
    f += [i / max(1, L - 1), float(L), 1.0 if i <= 1 else 0.0, 1.0 if i >= L - 2 else 0.0]  # position/terminal
    return f


def load(path):
    X, y, grp, fl = [], [], [], []
    for gi, ln in enumerate(Path(path).read_text().splitlines()):
        r = json.loads(ln)
        seq, ent = r["seq"], r["per_res_entropy"]
        for i, e in enumerate(ent):
            if e is None or i >= len(seq):
                continue
            X.append(res_features(seq, i))
            y.append(e)
            grp.append(gi)
            fl.append(FLEX.get(seq[i], 0.45))
    return np.array(X), np.array(y), np.array(grp), np.array(fl)


def grouped_cv(X, y, grp, k=5):
    ug = np.unique(grp)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    folds = {g: i % k for i, g in enumerate(ug)}
    fa = np.array([folds[g] for g in grp])
    pred = np.full(len(y), np.nan)
    for f in range(k):
        tr = fa != f
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.05,
                                          l2_regularization=1.0, min_samples_leaf=15, random_state=0).fit(X[tr], y[tr])
        pred[fa == f] = m.predict(X[fa == f])
    return pred


def main():
    path = ROOT / "data" / "sfree_proto.jsonl"
    X, y, grp, fl = load(path)
    npep = len(np.unique(grp))
    print(f"=== E123 per-residue entropy surrogate ({len(y)} residues from {npep} peptides) ===\n")
    if npep < 8:
        print("  too few peptides yet — wait for e122 MD.")
        return
    pred = grouped_cv(X, y, grp)
    ok = ~np.isnan(pred)
    r = pearsonr(pred[ok], y[ok])[0]
    rmse = float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
    base_r = pearsonr(fl[ok], y[ok])[0]  # single-residue flexibility baseline (e119 proxy)
    print(f"  per-residue entropy, grouped 5-fold CV (no test-peptide residue in train):")
    print(f"     context-ML surrogate  r={r:+.3f}  RMSE={rmse:.3f}")
    print(f"     single-flex baseline  r={base_r:+.3f}   (e119 scalar proxy)")
    print(f"     Δ = {r - base_r:+.3f}  → {'CONTEXT HELPS, surrogate learns entropy' if r - base_r > 0.05 else 'context adds little'}")
    # by residue type: does it capture Gly(floppy) vs Pro(rigid)?
    print("\n  mean MD entropy by residue type (sanity — Gly/Ser high, Pro/Ile low expected):")
    seqs = [json.loads(l) for l in path.read_text().splitlines()]
    byaa = {}
    for r_ in seqs:
        for i, e in enumerate(r_["per_res_entropy"]):
            if e is not None and i < len(r_["seq"]):
                byaa.setdefault(r_["seq"][i], []).append(e)
    for a in sorted(byaa, key=lambda a: -np.mean(byaa[a])):
        if len(byaa[a]) >= 4:
            print(f"     {a}: {np.mean(byaa[a]):.2f}  (n={len(byaa[a])})")
    print("\n  reading: surrogate r>0.4 and > baseline ⇒ per-residue entropy IS learnable from context →")
    print("  launch full 922 run, then deploy: entropy_lost = Σ surrogate(contacting residues).")


if __name__ == "__main__":
    main()
