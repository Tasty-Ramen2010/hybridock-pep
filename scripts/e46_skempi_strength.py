"""E46 — EXPERIMENTAL per-residue interface strength from SKEMPI 2.0 (Ram's 'strength dictionary').

The idea: build a real, data-derived dictionary of how strongly each residue type contributes to
binding — high for Trp/Phe/Leu, low for weak ones — but from EXPERIMENT, not hand-tuning. SKEMPI
2.0 has ~7000 interface point-mutation ΔKd measurements. For an alanine scan X->A:
    ΔΔG(X->A) = RT·ln(Kd_mut / Kd_wt)          (>0 = mutation weakens binding = X was favorable)
So strength[X] = mean ΔΔG(X->A) over all contexts = the experimental binding contribution of X.

This is the experimentally-grounded upgrade to MJ's *statistical* contact potential, and it's
protein-protein (transfers to peptides better than small-molecule ligands). Test:
  1. the dictionary itself (does it rank Trp/Phe/Leu high, as physics says?)
  2. burial-weighted peptide 'hotspot energy' vs ΔG on crystal-65 + the-98 (universal?)
  3. does it ADD to geometry + MJ (i.e. experimental signal beyond the statistical MJ)?
"""
from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from hybridock_pep.scoring.geometry_features import (GEOMETRY_FEATURE_KEYS,  # noqa: E402
                                                     compute_geometry_features)
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
R = 1.987e-3  # kcal/mol/K


def build_strength_dict(min_n=8):
    """strength[aa] = mean ΔΔG(aa->A) over SKEMPI single alanine mutations (kcal/mol, >0 = hotspot)."""
    rows = list(csv.reader((ROOT / "data/skempi_v2.csv").read_text().splitlines(), delimiter=";"))
    hdr = rows[0]
    ix = {c: i for i, c in enumerate(hdr)}
    acc = defaultdict(list)
    for row in rows[1:]:
        if len(row) < len(hdr):
            continue
        mut = row[ix["Mutation(s)_cleaned"]].strip()
        if not mut or "," in mut:          # single mutations only
            continue
        wt, chain, m = mut[0], mut[1], mut[-1]
        if m != "A" or wt not in "RNDCQEGHILKMFPSTWYV":  # alanine scan; skip X->A where X=A
            continue
        try:
            kd_m = float(row[ix["Affinity_mut_parsed"]]); kd_w = float(row[ix["Affinity_wt_parsed"]])
            T = float((row[ix["Temperature"]].split("(")[0].strip() or 298))
        except (ValueError, IndexError):
            continue
        if kd_m <= 0 or kd_w <= 0:
            continue
        ddg = R * T * math.log(kd_m / kd_w)   # >0: mut weakens binding -> wt residue was favorable
        acc[wt].append(ddg)
    strength = {aa: float(np.mean(v)) for aa, v in acc.items() if len(v) >= min_n}
    counts = {aa: len(v) for aa, v in acc.items()}
    return strength, counts


def buried_strength(pep_pdb, rec_pdb, strength):
    """Per-residue buried fraction x experimental strength; intensive (mean) + composition mean."""
    tmp = Path(f"/tmp/_e46_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (rec_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    try:
        SR.compute((pf := P.get_structure("f", str(pep_pdb))), level="R")
        free = [float(r.sasa) for r in pf.get_residues() if r.id[0] == " "]
        free_res = [r for r in pf.get_residues() if r.id[0] == " "]
        SR.compute((cx := P.get_structure("c", str(tmp))[0]), level="R")
        bound = [float(r.sasa) for r in cx["P"] if r.id[0] == " "]
        n = min(len(free), len(bound))
        if n == 0:
            return None
        comp, wsum, wnorm = [], 0.0, 0.0
        for i in range(n):
            aa = A3.get(free_res[i].resname.upper())
            if aa not in strength:
                continue
            s = strength[aa]
            comp.append(s)
            bur = max(0.0, free[i] - bound[i]) / (free[i] + 1e-6)   # buried fraction 0..1
            wsum += bur * s; wnorm += bur
        if not comp:
            return None
        return dict(
            strength_comp=float(np.mean(comp)),                 # mean dictionary value (composition)
            strength_bur=float(wsum / (wnorm + 1e-6)),          # burial-weighted mean (intensive)
            strength_burL=float(wsum / n),                      # burial-weighted per-residue
        )
    finally:
        tmp.unlink(missing_ok=True)


def main():
    strength, counts = build_strength_dict()
    print(f"=== SKEMPI experimental strength dict (mean ΔΔG of X->A, kcal/mol; >0 = hotspot, n>=8) ===")
    print(f"  {'aa':>3}{'strength':>10}{'n':>6}   (sorted: strongest binders first)")
    for aa, s in sorted(strength.items(), key=lambda kv: -kv[1]):
        tag = "  <- bulky/arom" if aa in "WFYLIM" else ("  <- charged" if aa in "DEKR" else "")
        print(f"  {aa:>3}{s:>10.2f}{counts[aa]:>6}{tag}")

    SK = ["strength_comp", "strength_bur", "strength_burL"]
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    rows = []
    for k, m in bench.items():
        pep, poc = ROOT / m["peptide_pdb"], ROOT / m["pocket_pdb"]
        if not pep.exists() or not poc.exists():
            continue
        g = compute_geometry_features(pep, poc)
        s = buried_strength(str(pep.resolve()), str(poc.resolve()), strength)
        if g and s:
            rows.append(dict(g, **s, pdb=k, set="cr", y=m["dg_exp"]))
    for k, v in e28.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if not pep.exists() or not rec.exists():
            continue
        g = compute_geometry_features(pep, rec)
        s = buried_strength(str(pep), str(rec), strength)
        if g and s:
            rows.append(dict(g, **s, pdb=k, set="b98", y=v["y"]))

    cr = [r for r in rows if r["set"] == "cr"]; b9 = [r for r in rows if r["set"] == "b98"]
    ycr = np.array([r["y"] for r in cr]); y9 = np.array([r["y"] for r in b9])
    print(f"\n=== strength feature vs ΔG — sign-consistency (cr n={len(cr)} / 98 n={len(b9)}) ===")
    print(f"  {'feature':<16}{'crystal-65':>12}{'the-98':>10}{'universal?':>12}")
    for f in SK + ["mj_contact"]:
        vc = np.array([r.get(f, 0.0) for r in cr]); v9 = np.array([r.get(f, 0.0) for r in b9])
        rc = pearsonr(vc, ycr).statistic if vc.std() > 0 else 0
        r9 = pearsonr(v9, y9).statistic if v9.std() > 0 else 0
        ok = "YES" if rc * r9 > 0 and min(abs(rc), abs(r9)) > 0.1 else "flip/weak"
        print(f"  {f:<16}{rc:>+12.3f}{r9:>+10.3f}{ok:>12}")

    y = np.array([r["y"] for r in rows])

    def loo(feats):
        X = np.array([[r.get(f, 0.0) for f in feats] for r in rows]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
            w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())

    G = list(GEOMETRY_FEATURE_KEYS)
    print(f"\n=== pooled LOO n={len(rows)}: does EXPERIMENTAL strength add to geometry+MJ? ===")
    for nm, fs in [("geometry (incl MJ) [baseline]", G),
                   ("+ strength_bur", G + ["strength_bur"]),
                   ("+ strength_comp", G + ["strength_comp"]),
                   ("strength alone (3 feats)", SK)]:
        r, e = loo(fs)
        print(f"  {nm:<32} r={r:+.3f} RMSE={e:.2f}")
    print("  >> if strength_bur LIFTS geometry+MJ, experiment adds signal beyond the statistical potential")
    print("  >> this is the cheap proof-of-concept for the full PDBbind atom-pair build")


if __name__ == "__main__":
    main()
