"""E11 — WHY does length flip sign across datasets? (hidden-variable diagnosis)

Hypothesis: total length L conflates two physically-OPPOSITE quantities:
  n_contact  (buried interface residues) -> FAVORABLE  (more = stronger, coeff<0)
  n_tail = L - n_contact (non-contacting) -> UNFAVORABLE (entropy cost, coeff>0)
A dataset's marginal corr(L, ΔG) then depends on its buried-vs-tail MIX, so it
flips between datasets (Simpson's paradox) while the underlying physics is fixed.

WIN CONDITION: n_contact and n_tail each have a CONSISTENT sign across crystal-65
AND PEPBI. That would prove the flip is a composition artifact and hand us a
sign-stable decomposition.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

P = PDBParser(QUIET=True)


def n_contact_pep(pep_pdb, poc_pdb, cut=5.5):
    pep = [r for r in P.get_structure("p", pep_pdb)[0].get_residues() if r.id[0] == " "]
    poc = [a for r in P.get_structure("q", poc_pdb)[0].get_residues()
           if r.id[0] == " " for a in r if a.element != "H"]
    if not pep or not poc:
        return None, None
    ns = NeighborSearch(poc)
    nc = sum(any(ns.search(a.coord, cut) for a in rp if a.element != "H") for rp in pep)
    return nc, len(pep)


def n_contact_chainB(pdb, cut=5.5):
    s = P.get_structure("x", pdb)[0]
    if "B" not in [c.id for c in s]:
        return None, None
    pep = [r for r in s["B"] if r.id[0] == " "]
    poc = [a for c in s if c.id != "B" for r in c if r.id[0] == " "
           for a in r if a.element != "H"]
    if not pep or not poc:
        return None, None
    ns = NeighborSearch(poc)
    nc = sum(any(ns.search(a.coord, cut) for a in rp if a.element != "H") for rp in pep)
    return nc, len(pep)


def load_crystal():
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    out = []
    for r in e0:
        if not r.get("pep_pdb"):
            continue
        nc, L = n_contact_pep(r["pep_pdb"], r["poc_pdb"])
        if nc is None:
            continue
        out.append(dict(y=r["y"], L=L, nc=nc, ntail=L - nc, aff=r["aff"], ds="crystal"))
    return out


def load_pepbi():
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    wb = openpyxl.load_workbook("/tmp/pepbi/PEPBI.xlsx", read_only=True)
    rows = list(wb["PEPBI Data"].iter_rows(values_only=True))
    hdr = rows[1]

    def ci(n):
        return hdr.index(n)

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    c_nm, c_dg, c_kd, c_bg = ci("PEPBI Complex Name"), ci("ΔG (kcal/mol)"), ci("KD (M)"), ci("Binding Group")
    out = []
    for r in rows[2:]:
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        nc, L = n_contact_chainB(files[nm])
        if nc is None or L == 0:
            continue
        out.append(dict(y=dg, L=L, nc=nc, ntail=L - nc, aff="Kd", ds="pepbi", bg=r[c_bg]))
    return out


def corr_row(rows, key):
    y = np.array([r["y"] for r in rows])
    v = np.array([r[key] for r in rows])
    if np.std(v) == 0:
        return float("nan")
    return pearsonr(v, y).statistic


def main():
    print("loading crystal-65...")
    cr = load_crystal()
    print(f"  {len(cr)} complexes")
    print("loading PEPBI...")
    pb = load_pepbi()
    print(f"  {len(pb)} complexes")

    print("\n" + "=" * 66)
    print("THE FLIP: corr(feature, ΔG) on each dataset  (ΔG<0=strong)")
    print("=" * 66)
    print(f"{'feature':<14}{'crystal-65':>14}{'PEPBI':>14}{'sign-stable?':>14}")
    for key in ["L", "nc", "ntail", "tail_frac"]:
        if key == "tail_frac":
            for r in cr + pb:
                r["tail_frac"] = r["ntail"] / max(1, r["L"])
        rc = corr_row(cr, key)
        rp = corr_row(pb, key)
        stable = "YES" if (np.isfinite(rc) and np.isfinite(rp) and rc * rp > 0) else "NO (FLIPS)"
        print(f"{key:<14}{rc:>14.3f}{rp:>14.3f}{stable:>14}")

    print("\n" + "=" * 66)
    print("buried-vs-tail MIX per dataset (explains the flip if mixes differ)")
    print("=" * 66)
    for name, rows in [("crystal-65", cr), ("PEPBI", pb)]:
        L = np.array([r["L"] for r in rows])
        nc = np.array([r["nc"] for r in rows])
        tf = np.array([r["ntail"] / max(1, r["L"]) for r in rows])
        print(f"  {name}: mean L={L.mean():.1f}  mean n_contact={nc.mean():.1f}  "
              f"mean tail_fraction={tf.mean():.2f}  (corr L~nc={pearsonr(L,nc).statistic:+.2f})")

    print("\n" + "=" * 66)
    print("POOLED across both datasets — do nc / ntail keep consistent signs?")
    print("=" * 66)
    allrows = cr + pb
    for key in ["L", "nc", "ntail"]:
        r = corr_row(allrows, key)
        print(f"  pooled corr({key}, ΔG) = {r:+.3f}  (n={len(allrows)})")

    Path("/tmp/e11_data.json").write_text(json.dumps(cr + pb))


if __name__ == "__main__":
    main()
