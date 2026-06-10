"""E8 — does interface H-BOND COUNT replicate on crystal-65?

PEPBI (31 binding groups) showed interface H-bond count significant cross-family
(r=-0.41, p=0.026). If the SAME feature, computed geometrically, also works on the
independent crystal-65 set, it is a genuinely robust cross-family affinity signal —
the first to replicate on two independent datasets. This is the validation that
NIS failed.

Geometric H-bond: peptide N/O atom within 3.5 Å of a receptor N/O atom. Report
total count AND density (per contact residue), family-mean, length-residualized,
permutation-tested.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402

P = PDBParser(QUIET=True)


def hbonds(pep_pdb, poc_pdb, hb_cut=3.5, contact_cut=5.5):
    pep = [r for r in P.get_structure("p", pep_pdb)[0].get_residues() if r.id[0] == " "]
    poc_atoms = [a for r in P.get_structure("q", poc_pdb)[0].get_residues()
                 if r.id[0] == " " for a in r if a.element != "H"]
    if not pep or not poc_atoms:
        return None
    ns = NeighborSearch(poc_atoms)
    n_hb = 0
    n_contact = 0
    for rp in pep:
        contacted = False
        for a in rp:
            if a.element == "H":
                continue
            near = ns.search(a.coord, contact_cut)
            if near:
                contacted = True
            if a.element in ("N", "O"):
                if any(b.element in ("N", "O") and np.linalg.norm(a.coord - b.coord) <= hb_cut
                       for b in ns.search(a.coord, hb_cut)):
                    n_hb += 1
        n_contact += contacted
    return n_hb, n_contact


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


def resid(x, z):
    z = np.asarray(z, float)
    if np.std(z) == 0:
        return x - x.mean()
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def perm_p(V, Y, L, n=20000, seed=0):
    vr, yr = resid(V, L), resid(Y, L)
    r = pearsonr(vr, yr).statistic
    rng = np.random.default_rng(seed)
    c = sum(abs(pearsonr(vr, resid(Y[rng.permutation(len(Y))], L)).statistic) >= abs(r)
            for _ in range(n))
    return r, (c + 1) / (n + 1)


def main():
    rows = json.loads(Path("/tmp/e3_features.json").read_text())  # has seq, pep_pdb, poc_pdb
    for r in rows:
        res = hbonds(r["pep_pdb"], r["poc_pdb"])
        if res:
            r["n_hb"], r["n_hb_contact"] = res
            r["hb_density"] = res[0] / max(1, res[1])
    rows = [r for r in rows if "n_hb" in r]
    print(f"crystal-65: H-bonds computed on {len(rows)} complexes")

    kd = np.array([r["aff"] == "Kd" for r in rows])
    for split, mask in [("ALL", np.ones(len(rows), bool)), ("Kd", kd)]:
        sub = [r for r, m in zip(rows, mask) if m]
        g = kmer_groups([r["seq"] for r in sub], 0.3)
        bgs = {}
        for i, gi in enumerate(g):
            bgs.setdefault(gi, []).append(sub[i])
        ks = sorted(bgs)
        Y = np.array([np.mean([x["y"] for x in bgs[k]]) for k in ks])
        L = np.array([np.mean([x["L"] for x in bgs[k]]) for k in ks])
        print(f"\n=== crystal-65 {split}: {len(ks)} families ===")
        for fk in ("n_hb", "hb_density"):
            V = np.array([np.mean([x[fk] for x in bgs[k]]) for k in ks])
            r, p = perm_p(V, Y, L)
            flag = "  <== SIG" if p < 0.05 else ""
            print(f"  {fk:<14} family-mean lenresid r={r:+.3f}  perm p={p:.4f}{flag}")


if __name__ == "__main__":
    main()
