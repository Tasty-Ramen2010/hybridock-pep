"""E10 — is LENGTH a real signal or a per-dataset confound? + the conditional hypothesis.

User's hypothesis: length dominates scores EXCEPT when a genuinely strong binder
gives a genuinely good score ("breaks through"). Two tests:

  TEST A — sign consistency: corr(length, ΔG) in crystal-65 vs PEPBI. If length
    were real physics, the SIGN is the same in both. If it flips, length is a
    per-dataset accident (confound).

  TEST B — conditional/"break-through": split complexes into strong vs weak
    binders; does the score (⟨E_int⟩, NIS) predict ΔG better among STRONG binders?
    Also length-stratified correlation (within length bins).
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
from scipy.stats import pearsonr, spearmanr  # noqa: E402


def load_crystal():
    rows = json.loads(Path("/tmp/e3_features.json").read_text())
    e9 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e9_results.json").read_text())}
    out = []
    for r in rows:
        e = e9.get(r["pdb"].upper())
        out.append(dict(y=r["y"], L=r["L"], aff=r["aff"],
                        nis_p=r["nis_p_frac"],
                        e_int=e["e_int_mean"] if e and np.isfinite(e.get("e_int_mean", np.nan)) else None))
    return out


def load_pepbi():
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

    c_dg, c_kd, c_L = ci("ΔG (kcal/mol)"), ci("KD (M)"), ci("Peptide Length")
    out = []
    for r in rows[2:]:
        dg, kd, L = num(r[c_dg]), num(r[c_kd]), num(r[c_L])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is not None and L:
            out.append(dict(y=dg, L=L))
    return out


def main():
    cryst = load_crystal()
    pepbi = load_pepbi()

    print("=" * 60)
    print("TEST A — is length's DIRECTION consistent across datasets?")
    print("=" * 60)
    yc = np.array([r["y"] for r in cryst])
    Lc = np.array([r["L"] for r in cryst])
    yp = np.array([r["y"] for r in pepbi])
    Lp = np.array([r["L"] for r in pepbi])
    rc = pearsonr(Lc, yc).statistic
    rp = pearsonr(Lp, yp).statistic
    print(f"  crystal-65: corr(length, ΔG) = {rc:+.3f}  (n={len(yc)})")
    print(f"  PEPBI     : corr(length, ΔG) = {rp:+.3f}  (n={len(yp)})")
    print(f"  (ΔG negative=strong; +corr => longer binds WEAKER)")
    if rc * rp < 0:
        print("  >>> SIGN FLIPS across datasets => length is a per-dataset CONFOUND,")
        print("      not real physics. Cannot be used as signal; must residualize.")
    else:
        print("  >>> Same sign => length carries a CONSISTENT (real?) trend; investigate.")

    print("\n" + "=" * 60)
    print("TEST B — conditional 'break-through': does score predict ΔG")
    print("         better among STRONG binders than weak ones?")
    print("=" * 60)
    rec = [r for r in cryst if r["e_int"] is not None]
    y = np.array([r["y"] for r in rec])
    L = np.array([r["L"] for r in rec])
    E = np.array([r["e_int"] for r in rec])
    NP = np.array([r["nis_p"] for r in rec])
    med = np.median(y)
    strong = y <= med  # more negative = stronger
    weak = y > med
    for name, v in [("⟨E_int⟩", E), ("nis_p", NP), ("length", L)]:
        rs = pearsonr(v[strong], y[strong]).statistic
        rw = pearsonr(v[weak], y[weak]).statistic
        ra = pearsonr(v, y).statistic
        print(f"  {name:<10} all={ra:+.3f}  strong-half={rs:+.3f}  weak-half={rw:+.3f}")
    print("  (if strong-half >> weak-half, 'good binders break through' holds)")

    print("\n" + "=" * 60)
    print("TEST B2 — length-stratified: within a length bin, does ⟨E_int⟩ predict ΔG?")
    print("=" * 60)
    order = np.argsort(L)
    nbin = 3
    for b in range(nbin):
        idx = order[b * len(order) // nbin:(b + 1) * len(order) // nbin]
        if len(idx) < 5 or np.std(E[idx]) == 0:
            continue
        lo, hi = L[idx].min(), L[idx].max()
        r = pearsonr(E[idx], y[idx]).statistic
        rn = pearsonr(NP[idx], y[idx]).statistic
        print(f"  length {lo:.0f}-{hi:.0f} (n={len(idx)}): "
              f"corr(E_int,ΔG)={r:+.3f}  corr(nis_p,ΔG)={rn:+.3f}")


if __name__ == "__main__":
    main()
