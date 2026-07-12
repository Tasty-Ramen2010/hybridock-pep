"""E9e — (a) 60ps vs 1ns convergence, (b) entropy-corrected LIE 3-model test.

User's hypothesis: LIE/MM-GBSA misses the configurational entropy term that should
cancel the size-scaling of the interaction energy. Add a per-residue entropy
penalty (contact-state + AA-type + SS aware) and see if it beats plain length
normalization.

Models compared (all on the 60ps ⟨E_int⟩ = e_int_mean):
  M0: e_int_mean alone                         (baseline, backwards/size)
  M1: e_int_mean + λ·N_res                      (= linear length normalization)
  M2: e_int_mean + Σ per-residue entropy        (composition: contact/AA/SS aware)
λ / scale fit by leave-one-FAMILY-out so no in-sample inflation.

Entropy penalty per residue i (kcal/mol, positive = entropy COST on binding):
  contact residue:     TΔS_sc(AA) + TΔS_bb(AA)        (fully frozen)
  non-contact residue: 0.5 · (TΔS_sc(AA) + TΔS_bb)    (tail retains ~half)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e3_physical_entropy import TDS_SC, TDS_BB  # noqa: E402

from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402

P = PDBParser(QUIET=True)
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


def entropy_penalty(pep_pdb, poc_pdb, contact_cut=5.5, noncontact_factor=0.5):
    pep = [r for r in P.get_structure("p", pep_pdb)[0].get_residues() if r.id[0] == " "]
    poc_atoms = [a for r in P.get_structure("q", poc_pdb)[0].get_residues()
                 if r.id[0] == " " for a in r if a.element != "H"]
    if not pep or not poc_atoms:
        return None
    ns = NeighborSearch(poc_atoms)
    total = 0.0
    for rp in pep:
        aa = AA3to1.get(rp.resname.upper(), "A")
        s = TDS_SC.get(aa, 1.0) + TDS_BB.get(aa, 1.0)
        contacted = any(ns.search(a.coord, contact_cut) for a in rp if a.element != "H")
        total += s if contacted else noncontact_factor * s
    return total


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


def lofo_cv(X, y, g):
    """leave-one-family-out CV predictions for design matrix X (with intercept added)."""
    pred = np.zeros_like(y)
    for fam in set(g):
        te = g == fam
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = w[0] + X[te] @ w[1:]
    return pred


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    # ---- (a) convergence 60ps vs 1ns ----
    ps = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e9_results.json").read_text())}
    ns = json.loads(Path("/tmp/e9d_1ns.json").read_text())
    print("=== 60ps vs 1ns convergence (same complexes) ===")
    print(f"{'pdb':<6}{'exp':>7}{'60_dg':>8}{'1ns_dg':>8}{'Δabs':>7}{'60_IE':>7}{'1ns_IE':>7}")
    dabs = []
    for r in ns:
        p = ps.get(r["pdb"].upper())
        if not p:
            continue
        d = r["dg_pred"] - p["dg_pred"]
        dabs.append(abs(d))
        print(f"{r['pdb']:<6}{r['y']:>7.1f}{p['dg_pred']:>8.1f}{r['dg_pred']:>8.1f}"
              f"{d:>7.1f}{p['minus_tds_ie']:>7.1f}{r['minus_tds_ie']:>7.1f}")
    print(f"mean |Δ dg_pred| 60ps->1ns = {np.mean(dabs):.1f} kcal/mol "
          f"(absolute energy NOT converged at 60ps)")

    # ---- (b) entropy-corrected LIE on the full 60ps set ----
    rows = json.loads(Path("/tmp/e9_results.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
    recs = []
    for r in rows:
        if not np.isfinite(r.get("dg_pred", np.nan)):
            continue
        meta = e0.get(r["pdb"].upper())
        if not meta or not meta.get("pep_pdb"):
            continue
        ent = entropy_penalty(meta["pep_pdb"], meta["poc_pdb"])
        if ent is None:
            continue
        recs.append(dict(pdb=r["pdb"], y=r["y"], L=r["L"], aff=r["aff"],
                        e_int=r["e_int_mean"], ent=ent,
                        seq=sm.get(r["pdb"].upper(), "X")))
    recs = [r for r in recs if r["seq"] != "X"]
    print(f"\n=== entropy-corrected LIE: {len(recs)} complexes ===")
    y = np.array([r["y"] for r in recs])
    L = np.array([r["L"] for r in recs])
    E = np.array([r["e_int"] for r in recs])
    S = np.array([r["ent"] for r in recs])
    g = kmer_groups([r["seq"] for r in recs], 0.3)
    kd = np.array([r["aff"] == "Kd" for r in recs])

    def report(name, X, mask):
        Xm, ym, gm = X[mask], y[mask], g[mask]
        if len(set(gm)) < 4:
            return
        pred = lofo_cv(Xm, ym, gm)
        r = pearsonr(pred, ym).statistic
        print(f"    {name:<34} CV r={r:+.3f}  RMSE={rmse(ym,pred):.2f}")

    for label, mask in [("ALL", np.ones(len(recs), bool)), ("Kd", kd)]:
        print(f"  -- {label} (n={int(mask.sum())}, fam={len(set(g[mask]))}) "
              f"baseline(mean) RMSE={np.std(y[mask]):.2f} --")
        report("M0: E_int alone", E.reshape(-1, 1), mask)
        report("M1: E_int + N_res", np.column_stack([E, L]), mask)
        report("M2: E_int + entropy(contact/AA)", np.column_stack([E, S]), mask)
        report("M2b: E_int + entropy + N_res", np.column_stack([E, S, L]), mask)
        # is entropy just length? correlation
    print(f"\n  corr(entropy_penalty, N_res) = {pearsonr(S, L).statistic:+.3f}  "
          f"(if ~1.0, composition adds nothing beyond length)")


if __name__ == "__main__":
    main()
