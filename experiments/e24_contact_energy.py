"""E24 — per-contact ENERGY features (Ram's hotspot idea) + SASA variants.

The ceiling is hotspot energetics: geometry counts contacts, not their ENERGY.
Test whether per-contact chemistry recovers the strong-short failures (1T7R/1T76/4LSJ).

Features (all instant, no GPU — physics gate before the ESM run):
  mj_contact  : Σ over peptide-receptor residue contacts of Miyazawa-Jernigan energy
                (Trp-Trp strongly favorable, Lys-Lys unfavorable — real per-contact E)
  mj_min      : most-favorable single contact (the dominant hotspot)
  sasa_frac   : buried peptide SASA / TOTAL free peptide SASA  (fraction buried)
  sasa_per_res: buried peptide SASA / n_residues             (Ram's per-residue SASA)

Reports r AND RMSE (kcal/mol) for: geometry, +each feature, and specifically the
prediction error on the 3 hotspot failures. If mj_contact helps -> run ESM per-contact.
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
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from e18_hybrid_features import AA3to1  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEO = POCK + IFACE

# Miyazawa-Jernigan contact energies (1996, Table 3, upper-triangle e_ij in kT units).
# More NEGATIVE = more favorable contact. Order: CMFILVWYAGTSNQDEHRKP
_MJ_ORDER = "CMFILVWYAGTSNQDEHRKP"
# Compact MJ matrix (symmetric); values from Miyazawa & Jernigan 1996 (rounded).
_MJ_RAW = """
-5.44 -4.99 -5.80 -5.50 -5.83 -4.96 -4.95 -4.16 -3.57 -3.16 -3.11 -2.86 -2.59 -2.85 -2.41 -2.27 -3.60 -2.57 -1.95 -3.07
-4.99 -5.46 -6.56 -6.02 -6.41 -5.13 -5.55 -4.91 -3.94 -3.39 -3.51 -3.03 -2.95 -3.30 -2.74 -2.61 -3.98 -3.12 -2.48 -3.45
-5.80 -6.56 -7.26 -6.84 -7.28 -6.29 -6.16 -5.66 -4.81 -4.13 -4.28 -4.02 -3.75 -4.10 -3.48 -3.56 -4.77 -3.98 -3.36 -4.25
-5.50 -6.02 -6.84 -6.54 -7.04 -6.05 -5.78 -5.25 -4.58 -3.78 -4.03 -3.52 -3.24 -3.67 -3.17 -3.27 -4.14 -3.63 -3.01 -3.76
-5.83 -6.41 -7.28 -7.04 -7.37 -6.48 -6.14 -5.67 -4.91 -4.16 -4.34 -3.92 -3.74 -4.04 -3.40 -3.59 -4.54 -4.03 -3.37 -4.20
-4.96 -5.13 -6.29 -6.05 -6.48 -5.52 -5.18 -4.62 -4.04 -3.38 -3.46 -3.05 -2.83 -3.07 -2.48 -2.67 -3.58 -3.07 -2.49 -3.32
-4.95 -5.55 -6.16 -5.78 -6.14 -5.18 -5.06 -4.66 -3.82 -3.42 -3.22 -2.99 -3.07 -3.11 -2.84 -2.99 -3.98 -3.41 -2.69 -3.73
-4.16 -4.91 -5.66 -5.25 -5.67 -4.62 -4.66 -4.17 -3.36 -3.01 -3.01 -2.78 -2.76 -2.97 -2.76 -2.79 -3.52 -3.16 -2.60 -3.19
-3.57 -3.94 -4.81 -4.58 -4.91 -4.04 -3.82 -3.36 -2.72 -2.31 -2.32 -2.01 -1.84 -1.89 -1.70 -1.51 -2.41 -1.83 -1.31 -2.07
-3.16 -3.39 -4.13 -3.78 -4.16 -3.38 -3.42 -3.01 -2.31 -2.24 -2.08 -1.82 -1.74 -1.66 -1.59 -1.22 -2.15 -1.72 -1.15 -1.87
-3.11 -3.51 -4.28 -4.03 -4.34 -3.46 -3.22 -3.01 -2.32 -2.08 -2.12 -1.96 -1.88 -1.90 -1.80 -1.74 -2.42 -1.90 -1.31 -1.90
-2.86 -3.03 -4.02 -3.52 -3.92 -3.05 -2.99 -2.78 -2.01 -1.82 -1.96 -1.67 -1.58 -1.49 -1.63 -1.48 -2.11 -1.62 -1.05 -1.57
-2.59 -2.95 -3.75 -3.24 -3.74 -2.83 -3.07 -2.76 -1.84 -1.74 -1.88 -1.58 -1.68 -1.71 -1.68 -1.51 -2.08 -1.64 -1.21 -1.53
-2.85 -3.30 -4.10 -3.67 -4.04 -3.07 -3.11 -2.97 -1.89 -1.66 -1.90 -1.49 -1.71 -1.54 -1.46 -1.42 -1.98 -1.80 -1.29 -1.73
-2.41 -2.74 -3.48 -3.17 -3.40 -2.48 -2.84 -2.76 -1.70 -1.59 -1.80 -1.63 -1.68 -1.46 -1.21 -1.02 -2.32 -2.29 -1.68 -1.33
-2.27 -2.61 -3.56 -3.27 -3.59 -2.67 -2.99 -2.79 -1.51 -1.22 -1.74 -1.48 -1.51 -1.42 -1.02 -0.91 -2.15 -2.27 -1.80 -1.26
-3.60 -3.98 -4.77 -4.14 -4.54 -3.58 -3.98 -3.52 -2.41 -2.15 -2.42 -2.11 -2.08 -1.98 -2.32 -2.15 -3.05 -1.94 -1.35 -2.25
-2.57 -3.12 -3.98 -3.63 -4.03 -3.07 -3.41 -3.16 -1.83 -1.72 -1.90 -1.62 -1.64 -1.80 -2.29 -2.27 -1.94 -1.55 -0.59 -1.70
-1.95 -2.48 -3.36 -3.01 -3.37 -2.49 -2.69 -2.60 -1.31 -1.15 -1.31 -1.05 -1.21 -1.29 -1.68 -1.80 -1.35 -0.59 -0.12 -0.97
-3.07 -3.45 -4.25 -3.76 -4.20 -3.32 -3.73 -3.19 -2.07 -1.87 -1.90 -1.57 -1.53 -1.73 -1.33 -1.26 -2.25 -1.70 -0.97 -1.75
"""
_MJ = {}
_mat = [list(map(float, r.split())) for r in _MJ_RAW.strip().splitlines()]
for a, ia in enumerate(_MJ_ORDER):
    for b, ib in enumerate(_MJ_ORDER):
        _MJ[(ia, ib)] = _mat[a][b]


def contact_energy(cx_path, pep_chain, contact_cut=6.5):
    """Σ MJ energy over peptide-receptor residue contacts (CB-CB or any-heavy < cutoff)."""
    cx = P.get_structure("c", cx_path)[0]
    pep = [r for r in cx[pep_chain] if r.id[0] == " "]
    rec = [r for ch in cx if ch.id != pep_chain for r in ch if r.id[0] == " "]
    if not pep or not rec:
        return None
    rec_atoms = [a for r in rec for a in r if a.element != "H"]
    ns = NeighborSearch(rec_atoms)
    seen = set()
    mj_sum = 0.0
    mj_vals = []
    for rp in pep:
        ap = rp.resname.upper()
        a1 = AA3to1.get(ap, "A")
        nbr = set()
        for atom in rp:
            if atom.element == "H":
                continue
            for b in ns.search(atom.coord, contact_cut):
                nbr.add(b.get_parent())
        for rr in nbr:
            key = (id(rp), id(rr))
            if key in seen:
                continue
            seen.add(key)
            a2 = AA3to1.get(rr.resname.upper(), "A")
            e = _MJ.get((a1, a2), -1.5)
            mj_sum += e
            mj_vals.append(e)
    if not mj_vals:
        return None
    return dict(mj_contact=mj_sum, mj_min=min(mj_vals), mj_mean=float(np.mean(mj_vals)),
                n_contacts=len(mj_vals))


def sasa_variants(pep_pdb, cx_path, pep_chain):
    """Buried-SASA fraction of total, and per-residue."""
    def per_res(struct):
        SR.compute(struct, level="A")
        return {(r.get_parent().id, r.id): sum(float(a.sasa) for a in r)
                for r in struct.get_residues() if r.id[0] == " "}
    free = per_res(P.get_structure("f", pep_pdb))
    cpx = per_res(P.get_structure("c", cx_path))
    cx = P.get_structure("cc", cx_path)[0]
    pep_res = [r for r in cx[pep_chain] if r.id[0] == " "]
    pf = [r for r in P.get_structure("pp", pep_pdb)[0].get_residues() if r.id[0] == " "]
    n = min(len(pep_res), len(pf))
    total_free = sum(free.values()) or 1.0
    buried = 0.0
    for i in range(n):
        rfree = free.get((pf[i].get_parent().id, pf[i].id), 0.0)
        rbound = cpx.get((pep_res[i].get_parent().id, pep_res[i].id), 0.0)
        buried += max(0.0, rfree - rbound)
    return dict(sasa_frac=buried / total_free, sasa_per_res=buried / max(1, n),
                sasa_total_free=total_free / 100.0)


def loo_pred(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return p


def rr(p, y):
    return pearsonr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean()))


def main():
    e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
    geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    out_path = Path("/tmp/e24_contact.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for pdb, g in geo.items():
        if pdb in out or pdb not in e0:
            continue
        r0 = e0[pdb]
        merged = Path(f"/tmp/e18v3_cx/{pdb}.pdb")
        if not merged.exists():
            continue
        try:
            ce = contact_energy(str(merged), "P")
            sv = sasa_variants(r0["pep_pdb"], str(merged), "P")
        except Exception as ex:  # noqa: BLE001
            print(f"  {pdb} FAIL {type(ex).__name__}", flush=True); continue
        if ce and sv:
            out[pdb] = {**ce, **sv}
            out_path.write_text(json.dumps(out))
    rows = []
    for pdb, g in geo.items():
        if pdb in out:
            rows.append(dict(g, **out[pdb], L=bench[pdb]["peptide_len"]))
    y = np.array([r["y"] for r in rows])
    Xg = np.array([[r.get(f, 0.0) for f in GEO] for r in rows])
    print(f"n={len(rows)}  (ΔG in kcal/mol; r unitless, RMSE kcal/mol)\n")
    base = loo_pred(Xg, y)
    r0, rmse0 = rr(base, y)
    print(f"{'model':<30}{'r':>8}{'RMSE(kcal/mol)':>16}")
    print(f"{'geometry (baseline)':<30}{r0:>8.3f}{rmse0:>16.2f}")
    for feat in ["mj_contact", "mj_min", "mj_mean", "sasa_frac", "sasa_per_res", "sasa_total_free"]:
        v = np.array([r[feat] for r in rows])
        print(f"  raw corr({feat:<16}) = {pearsonr(v,y).statistic:+.3f}", end="")
        X = np.column_stack([Xg, v])
        p = loo_pred(X, y); r, rmse = rr(p, y)
        print(f"   | +geometry: r={r:+.3f} RMSE={rmse:.2f}")
    # best combo
    for combo in [["mj_contact"], ["mj_min"], ["mj_contact", "mj_min"], ["sasa_per_res"],
                  ["mj_min", "sasa_per_res"]]:
        X = np.column_stack([Xg] + [[r[c] for r in rows] for c in combo])
        p = loo_pred(X, y); r, rmse = rr(p, y)
        print(f"  geometry + {'+'.join(combo):<22} r={r:+.3f} RMSE={rmse:.2f}")

    # hotspot failures: does mj help them specifically?
    print("\n=== hotspot failures: geometry vs +mj_min (pred error, kcal/mol) ===")
    Xm = np.column_stack([Xg, [r["mj_min"] for r in rows]])
    pm = loo_pred(Xm, y)
    pdbs = [r["pdb"] for r in rows]
    for f in ["1T7R", "1T76", "4LSJ", "2Q7K"]:
        if f in pdbs:
            i = pdbs.index(f)
            print(f"  {f}: exp={y[i]:+.1f} geo_pred={base[i]:+.1f}(err{base[i]-y[i]:+.1f}) "
                  f"+mj_pred={pm[i]:+.1f}(err{pm[i]-y[i]:+.1f})")
    print(f"\n>> if mj helps: run ESM per-contact next. if not: per-contact chemistry isn't the lever.")


if __name__ == "__main__":
    main()
