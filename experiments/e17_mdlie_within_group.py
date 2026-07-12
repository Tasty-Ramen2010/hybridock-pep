"""E17 — does PHYSICS-based MD-LIE escape the per-target coin flip that geometric
features hit (e16)? And does longer sampling (250ps) help within-target ΔΔG?

Test MD-LIE (single-traj MM-GBSA + interaction entropy) WITHIN binding groups:
  - pick groups where GEOMETRY worked (geometric within-group Spearman > 0) and
    where it FAILED (< 0), per e16.
  - for each variant: split PEPBI complex (chain B = peptide, rest = receptor),
    run MD-LIE at the requested ps, get <E_int> and dg_pred.
  - within-group Spearman(dg_pred, ΔG_exp) per group.

Decision:
  * MD-LIE positive on BOTH winner and loser groups -> physics is universal
    per-target (escapes the geometric coin flip) -> worth pursuing.
  * MD-LIE also flips -> endpoint physics can't resolve within-group ΔΔG cheaply;
    only alchemical FEP can. Honest stop.

Usage: python e17_mdlie_within_group.py [prod_ps] [max_per_group]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scipy.stats import spearmanr  # noqa: E402
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402

P = PDBParser(QUIET=True)
IO = PDBIO()

# groups to test: (name substring, expected geometry result from e16)
TARGET_GROUPS = {
    "TtSlyD - S2": "winner(+0.55)",
    "SH3 STAC": "winner(+0.41)",
    "PTPA - PP2A": "loser(-0.26)",
    "SGT2-TPR": "loser(-1.00)",
    "SOCS - VASA": "loser(-0.87)",
}


class _PepSel(Select):
    def accept_chain(self, ch):
        return ch.id == "B"

    def accept_residue(self, res):
        return res.id[0] == " "


class _PocketSel(Select):
    """Receptor residues with any heavy atom within `radius` Å of the peptide."""
    def __init__(self, keep_resids):
        self.keep = keep_resids  # set of (chain_id, resid_tuple)

    def accept_chain(self, ch):
        return ch.id != "B"

    def accept_residue(self, res):
        return res.id[0] == " " and (res.get_parent().id, res.id) in self.keep


def split_complex(cif_or_pdb, tag, radius=12.0):
    import numpy as np
    s = P.get_structure("x", cif_or_pdb)
    chains = [c.id for c in s[0]]
    if "B" not in chains:
        return None
    pep_atoms = [a.coord for r in s[0]["B"] if r.id[0] == " "
                 for a in r if a.element != "H"]
    if not pep_atoms:
        return None
    pep_xyz = np.array(pep_atoms)
    # pocket residues within radius of peptide
    keep = set()
    r2 = radius * radius
    for ch in s[0]:
        if ch.id == "B":
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            for a in res:
                if a.element == "H":
                    continue
                if np.min(((pep_xyz - a.coord) ** 2).sum(1)) <= r2:
                    keep.add((ch.id, res.id))
                    break
    if not keep:
        return None
    d = Path("/tmp/e17_split")
    d.mkdir(exist_ok=True)
    pep = d / f"{tag}_pep.pdb"
    rec = d / f"{tag}_rec.pdb"
    IO.set_structure(s)
    IO.save(str(pep), _PepSel())
    IO.save(str(rec), _PocketSel(keep))
    return str(pep), str(rec)


def main():
    prod_ps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    max_pg = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    from e9_md_ensemble_ie import run_complex

    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    import openpyxl
    wb = openpyxl.load_workbook("/tmp/pepbi/PEPBI.xlsx", read_only=True)
    rows = list(wb["PEPBI Data"].iter_rows(values_only=True))
    hdr = rows[1]
    ci = lambda n: hdr.index(n)
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    c_nm, c_dg, c_kd, c_bg = ci("PEPBI Complex Name"), ci("ΔG (kcal/mol)"), ci("KD (M)"), ci("Binding Group")

    # collect variants per target group
    bygroup = {}
    for r in rows[2:]:
        bg = str(r[c_bg]) if r[c_bg] else ""
        match = next((k for k in TARGET_GROUPS if k in bg), None)
        if not match:
            continue
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        bygroup.setdefault(match, []).append((nm, files[nm], dg))

    results = {}
    for gname, variants in bygroup.items():
        # spread variants across ΔG range, cap at max_pg
        variants = sorted(variants, key=lambda t: t[2])
        if len(variants) > max_pg:
            idx = np.linspace(0, len(variants) - 1, max_pg).astype(int)
            variants = [variants[i] for i in idx]
        print(f"\n=== {gname} [{TARGET_GROUPS[gname]}] : {len(variants)} variants, {prod_ps}ps ===", flush=True)
        recs = []
        for nm, fp, dg in variants:
            sp = split_complex(fp, nm)
            if not sp:
                continue
            t0 = time.time()
            try:
                res = run_complex(sp[0], sp[1], prod_ps=prod_ps, frame_every_ps=10)
                recs.append(dict(nm=nm, dg=dg, **res))
                print(f"  {nm:<22} dg_pred={res['dg_pred']:8.1f} <E>={res['e_int_mean']:8.1f} "
                      f"(exp {dg:6.2f}) {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {nm:<22} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
        if len(recs) >= 4:
            y = np.array([r["dg"] for r in recs])
            for key in ["e_int_mean", "dg_pred"]:
                v = np.array([r[key] for r in recs])
                if np.std(v) > 0:
                    rho = spearmanr(v, y).statistic
                    print(f"  >> within-group Spearman({key}, ΔG) = {rho:+.3f}  (n={len(recs)})", flush=True)
            results[gname] = recs
    Path("/tmp/e17_results.json").write_text(json.dumps(results))
    print("\n=== SUMMARY: MD-LIE within-group (does physics beat the geometric coin flip?) ===")
    for gname, recs in results.items():
        y = np.array([r["dg"] for r in recs])
        v = np.array([r["dg_pred"] for r in recs])
        rho = spearmanr(v, y).statistic if np.std(v) > 0 else float("nan")
        print(f"  {gname:<16} [{TARGET_GROUPS[gname]:<14}] MD-LIE Spearman={rho:+.3f}")


if __name__ == "__main__":
    main()
