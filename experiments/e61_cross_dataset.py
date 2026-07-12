"""E61 — the honest test of Ram's correction: does it SCALE ACROSS DATASETS (not just within-98 LOO)?

e60 showed within-the-98 a length/blend correction lifts LOO Pearson 0.15->0.41. But the lifters
(L, mj, bsa) are EXTENSIVE size features that flip sign across datasets (Simpson). Decisive test:
train on ONE dataset, predict the OTHER (the-98 <-> crystal-65). A correction that survives this is
real; one that collapses/flips is the within-distribution mirage.

Shared features (both caches): mmgbsa(dg_single), eint, eint_std, minus_tds, cf(charged_frac), L.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def pep_len(pdb_path):
    from Bio.PDB import PDBParser
    P = PDBParser(QUIET=True)
    return sum(1 for r in P.get_structure("p", str(pdb_path))[0].get_residues() if r.id[0] == " ")


def _ok(v, *keys):
    return all(k in v and v[k] is not None and not (isinstance(v[k], float) and np.isnan(v[k])) for k in keys)


def load_98():
    d = json.loads(Path("/tmp/e49b_the98.json").read_text())
    rows = []
    for k, v in d.items():
        if not _ok(v, "dg_single", "e_int_mean", "minus_tds", "L", "y"):
            continue
        rows.append(dict(mmgbsa=v["dg_single"], eint=v["e_int_mean"], eint_std=v["e_int_std"],
                         mtds=v["minus_tds"], cf=v["cf"], L=v["L"], y=v["y"],
                         eint_perL=v["e_int_mean"] / max(1, v["L"]),
                         mmgbsa_perL=v["dg_single"] / max(1, v["L"])))
    return rows


def load_65():
    d = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text())
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    rows = []
    for k, v in d.items():
        L = None
        if k.upper() in bench:
            try:
                L = pep_len(bench[k.upper()]["peptide_pdb"])
            except Exception:
                L = None
        if L is None or not _ok(v, "dg_single", "e_int_mean", "minus_tds", "y"):
            continue
        rows.append(dict(mmgbsa=v["dg_single"], eint=v["e_int_mean"], eint_std=v["e_int_std"],
                         mtds=v["minus_tds"], cf=v["cf"], L=L, y=v["y"],
                         eint_perL=v["e_int_mean"] / max(1, L),
                         mmgbsa_perL=v["dg_single"] / max(1, L)))
    return rows


def fit_predict(tr, te, cols):
    X = np.array([[r[c] for c in cols] for r in tr], dtype=float)
    y = np.array([r["y"] for r in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    Xe = np.array([[r[c] for c in cols] for r in te], dtype=float)
    pred = np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w
    ye = np.array([r["y"] for r in te])
    return pearsonr(pred, ye)[0], w[1:]  # corr + standardized weights


def main():
    r98 = load_98()
    r65 = load_65()
    print(f"=== E61 cross-dataset transfer.  the-98 n={len(r98)}  crystal-65 n={len(r65)} ===\n")

    sets = {
        "mmgbsa": ["mmgbsa"],
        "mmgbsa+L": ["mmgbsa", "L"],
        "mmgbsa+eint (model-ensemble)": ["mmgbsa", "eint"],
        "mmgbsa+eint+L": ["mmgbsa", "eint", "L"],
        "eint_perL (INTENSIVE)": ["eint_perL"],
        "mmgbsa_perL+eint_perL (INTENSIVE blend)": ["mmgbsa_perL", "eint_perL"],
        "mmgbsa+eint+cf+mtds": ["mmgbsa", "eint", "cf", "mtds"],
    }
    print(f"{'feature set':<42}{'train98→test65':>16}{'train65→test98':>16}  weights(on L?)")
    for nm, cols in sets.items():
        try:
            p_98_65, w1 = fit_predict(r98, r65, cols)
            p_65_98, w2 = fit_predict(r65, r98, cols)
            lw = ""
            if "L" in cols:
                li = cols.index("L")
                lw = f"L: {w1[li]:+.2f}(98) vs {w2[li]:+.2f}(65)"
                if w1[li] * w2[li] < 0:
                    lw += "  FLIPS!"
            print(f"  {nm:<40}{p_98_65:>+16.3f}{p_65_98:>+16.3f}  {lw}")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm}: {str(e)[:50]}")

    # show the length confound directly
    print("\n=== the length confound, directly ===")
    for nm, rows in [("the-98", r98), ("crystal-65", r65)]:
        L = np.array([r["L"] for r in rows]); y = np.array([r["y"] for r in rows])
        print(f"  corr(L, exp ΔG) on {nm:<12} = {pearsonr(L, y)[0]:+.3f}  (n={len(rows)})")
    print("  >> opposite signs = the length term that lifts within-98 ANTI-helps crystal-65 = mirage.")
    print("  >> a feature whose train98→65 AND train65→98 are BOTH positive = real, transferable.")


if __name__ == "__main__":
    main()
