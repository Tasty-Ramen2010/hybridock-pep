"""E7 — independent NIS replication on PEPBI (clean structures, ITC Kd).

Last night's fresh-family replication was invalid (88% degenerate extraction).
PEPBI ships clean predicted structures with a reliable chain convention
(A=protein, B=peptide) and ITC Kd/ΔG, across ~31 binding groups. This is the
apples-to-apples cross-DATASET replication of the crystal-65 NIS signal.

Tests:
  - cross-binding-group family-mean, length-residualized r + permutation p
  - within-binding-group nis_p vs ΔG (the variant-ranking regime we ship)
"""
from __future__ import annotations

import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402

P = PDBParser(QUIET=True)
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
STRUCT_DIR = "/tmp/pepbi/struct"
XLSX = "/tmp/pepbi/PEPBI.xlsx"


def _cls(rn: str) -> str:
    rn = rn.upper()
    return "C" if rn in CHARGED else ("P" if rn in POLAR else "A")


def nis(pdb: str, pep_chain: str = "B"):
    s = P.get_structure("x", pdb)[0]
    if pep_chain not in [c.id for c in s]:
        return None
    pep = [r for r in s[pep_chain] if r.id[0] == " "]
    rec = [a for c in s if c.id != pep_chain for r in c if r.id[0] == " "
           for a in r if a.element != "H"]
    if not pep or not rec:
        return None
    ns = NeighborSearch(rec)
    npolar = ncharged = nnis = 0
    for rp in pep:
        if any(ns.search(a.coord, 5.5) for a in rp if a.element != "H"):
            continue
        nnis += 1
        c = _cls(rp.resname)
        npolar += c == "P"
        ncharged += c == "C"
    if nnis == 0:
        return None
    return npolar / nnis, ncharged / nnis


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


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob(f"{STRUCT_DIR}/**/*.pdb", recursive=True)}
    print(f"structure files indexed: {len(files)}")

    wb = openpyxl.load_workbook(XLSX, read_only=True)
    rows = list(wb["PEPBI Data"].iter_rows(values_only=True))
    hdr = rows[1]

    def ci(n):
        return hdr.index(n)

    c_bg, c_nm, c_kd, c_dg, c_L = (ci("Binding Group"), ci("PEPBI Complex Name"),
                                   ci("KD (M)"), ci("ΔG (kcal/mol)"), ci("Peptide Length"))
    recs, miss, degen = [], 0, 0
    for r in rows[2:]:
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            miss += 1
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        res = nis(files[nm])
        if res is None:
            degen += 1
            continue
        recs.append(dict(bg=r[c_bg], dg=dg, L=num(r[c_L]) or 0,
                         nis_p=res[0], nis_c=res[1]))
    nps = [r["nis_p"] for r in recs]
    print(f"NIS computed on {len(recs)} complexes (missing {miss}, degenerate {degen})")
    print(f"nis_p distribution: median={np.median(nps):.2f} mean={np.mean(nps):.2f}")

    bgs = {}
    for r in recs:
        bgs.setdefault(r["bg"], []).append(r)
    ks = sorted(bgs)
    Y = np.array([np.mean([x["dg"] for x in bgs[k]]) for k in ks])
    L = np.array([np.mean([x["L"] for x in bgs[k]]) for k in ks])
    Vp = np.array([np.mean([x["nis_p"] for x in bgs[k]]) for k in ks])
    Vc = np.array([np.mean([x["nis_c"] for x in bgs[k]]) for k in ks])
    print(f"\nINDEPENDENT cross-binding-group replication: {len(ks)} groups")
    for nm, V in [("nis_p_frac", Vp), ("nis_c_frac", Vc)]:
        r, p = perm_p(V, Y, L)
        print(f"  {nm}: family-mean lenresid r={r:+.3f}  perm p={p:.4f}")

    wr = []
    for k in ks:
        g = bgs[k]
        if len(g) >= 4:
            v = np.array([x["nis_p"] for x in g])
            y = np.array([x["dg"] for x in g])
            if np.std(v) > 0:
                wr.append((k, len(g), pearsonr(v, y).statistic))
    if wr:
        print(f"\nWITHIN-binding-group nis_p vs ΔG (variant ranking, n>=4):")
        for k, n_, r in wr:
            print(f"  {str(k)[:40]:<42} n={n_} r={r:+.3f}")
        print(f"  mean within-group r = {np.mean([r for *_, r in wr]):+.3f}  "
              f"({len(wr)} groups)")


if __name__ == "__main__":
    main()
