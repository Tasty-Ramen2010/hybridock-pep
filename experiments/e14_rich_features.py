"""E14 — rich geometric interface features for both datasets, for the universal
within-protein ΔΔG model. Each feature is a candidate; sign-stability across
datasets (e15) decides inclusion.

Features (all from peptide heavy atoms vs receptor heavy atoms, work on both
crystal-65 (pep_pdb/poc_pdb) and PEPBI (chain B vs rest)):
  n_contact         contact residues (any heavy atom <=5.5Å of receptor)
  hb_count/density  peptide N/O within 3.5Å of receptor N/O  (count, per-contact)
  hb_sc             sidechain (non-backbone) H-bonds only
  salt_bridge       charged pep sidechain N/O within 4.0Å of opposite-charge rec
  sb_density        salt bridges per contact residue
  hydrophobic_cc    apolar-apolar residue contacts
  elec_compl        favorable(opp-charge) - unfavorable(same-charge) charged pairs
  aromatic_cc       aromatic(F/Y/W/H) pep residue near aromatic rec residue
  contact_pairs     # heavy-atom pairs <=4.5Å (packing tightness)
  pack_density      contact_pairs / n_contact
  min_gap_mean      mean over contact residues of min dist to receptor (closeness)
  bsa               buried SASA (Shrake-Rupley) — optional (slower)
  bsa_hphobic       buried apolar-atom SASA — optional
Writes /tmp/e14_cr.json and /tmp/e14_pb.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
POS = {"ARG", "LYS", "HIS"}
NEG = {"ASP", "GLU"}
CHARGED = POS | NEG
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "GLY", "TRP"}
AROM = {"PHE", "TYR", "TRP", "HIS"}
BB = {"N", "CA", "C", "O"}
DO_SASA = "--sasa" in sys.argv


def _features(pep_res, rec_res):
    rec_atoms = [a for r in rec_res for a in r if a.element != "H"]
    if not pep_res or not rec_atoms:
        return None
    ns = NeighborSearch(rec_atoms)
    nc = hb = hb_sc = sb = hyd = arom = pairs = 0
    elec_fav = elec_unfav = 0
    mingaps = []
    for rp in pep_res:
        rn = rp.resname.upper()
        contacted = False
        mind = 99.0
        for a in rp:
            if a.element == "H":
                continue
            near5 = ns.search(a.coord, 5.5)
            if near5:
                contacted = True
            for b in ns.search(a.coord, 4.5):
                pairs += 1
                d = float(np.linalg.norm(a.coord - b.coord))
                if d < mind:
                    mind = d
            # H-bond
            if a.element in ("N", "O"):
                for b in ns.search(a.coord, 3.5):
                    if b.element in ("N", "O"):
                        hb += 1
                        if a.name not in BB:
                            hb_sc += 1
                        break
            # salt bridge + elec complementarity (charged sidechain polar atoms)
            if rn in CHARGED and a.element in ("N", "O") and a.name not in BB:
                want = NEG if rn in POS else POS
                anti = POS if rn in POS else NEG
                for b in ns.search(a.coord, 4.0):
                    brn = b.get_parent().resname.upper()
                    if brn in want:
                        sb += 1
                        elec_fav += 1
                        break
                    if brn in anti:
                        elec_unfav += 1
                        break
        if contacted:
            nc += 1
            mingaps.append(mind)
            if rn in APOLAR:
                # apolar-apolar: does it contact an apolar receptor residue?
                for a in rp:
                    if a.element == "H":
                        continue
                    if any(bb.get_parent().resname.upper() in APOLAR
                           for bb in ns.search(a.coord, 5.0)):
                        hyd += 1
                        break
            if rn in AROM:
                for a in rp:
                    if a.element == "H":
                        continue
                    if any(bb.get_parent().resname.upper() in AROM
                           for bb in ns.search(a.coord, 5.5)):
                        arom += 1
                        break
    nc = max(1, nc)
    return dict(
        n_contact=nc, hb_count=hb, hb_density=hb / nc, hb_sc=hb_sc,
        salt_bridge=sb, sb_density=sb / nc, hydrophobic_cc=hyd,
        elec_compl=elec_fav - elec_unfav, aromatic_cc=arom,
        contact_pairs=pairs, pack_density=pairs / nc,
        min_gap_mean=float(np.mean(mingaps)) if mingaps else 0.0,
    )


def _sasa_feats(pep_pdb, full_structure_paths):
    """BSA + buried hydrophobic SASA. full_structure_paths = (pep_path, rec_path)."""
    pep_path, rec_path = full_structure_paths
    def sasa_by_apolar(struct):
        SR.compute(struct, level="A")
        tot = sum(float(a.sasa) for a in struct.get_atoms())
        apo = 0.0
        for r in struct.get_residues():
            if r.resname.upper() in APOLAR:
                apo += sum(float(a.sasa) for a in r)
        return tot, apo
    try:
        sp, sa_p = sasa_by_apolar(P.get_structure("p", pep_path))
        sr, sa_r = sasa_by_apolar(P.get_structure("r", rec_path))
        # merged
        from Bio.PDB import Structure, Model
        st = Structure.Structure("c"); m = Model.Model(0); st.add(m)
        used = set()
        for src in (pep_path, rec_path):
            for ch in P.get_structure("a", src)[0]:
                c = ch.copy(); cid = c.id
                while cid in used:
                    cid = chr(((ord(cid) - 64) % 26) + 65)
                c.id = cid; used.add(cid); m.add(c)
        sc, sa_c = sasa_by_apolar(st)
        return dict(bsa=sp + sr - sc, bsa_hphobic=sa_p + sa_r - sa_c)
    except Exception:
        return dict(bsa=float("nan"), bsa_hphobic=float("nan"))


def crystal():
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    out = []
    for i, r in enumerate(e0):
        if not r.get("pep_pdb"):
            continue
        pep_res = [x for x in P.get_structure("p", r["pep_pdb"])[0].get_residues() if x.id[0] == " "]
        rec_res = [x for x in P.get_structure("q", r["poc_pdb"])[0].get_residues() if x.id[0] == " "]
        f = _features(pep_res, rec_res)
        if not f:
            continue
        if DO_SASA:
            f.update(_sasa_feats(r["pep_pdb"], (r["pep_pdb"], r["poc_pdb"])))
        f.update(y=r["y"], seq=sm.get(r["pdb"].upper(), "X"), aff=r["aff"])
        out.append(f)
        if (i + 1) % 20 == 0:
            print(f"  crystal {i+1}", flush=True)
    return out


def pepbi():
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
    out = []
    n = 0
    for r in rows[2:]:
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        s = P.get_structure("x", files[nm])[0]
        if "B" not in [c.id for c in s]:
            continue
        pep_res = [x for x in s["B"] if x.id[0] == " "]
        rec_res = [x for c in s if c.id != "B" for x in c if x.id[0] == " "]
        f = _features(pep_res, rec_res)
        if not f:
            continue
        f.update(y=dg, grp=str(r[c_bg]), aff="Kd")
        out.append(f)
        n += 1
        if n % 50 == 0:
            print(f"  pepbi {n}", flush=True)
    return out


def main():
    print("crystal-65 features...")
    cr = crystal()
    # kmer family grouping for crystal
    sys.path.insert(0, str(ROOT / "scripts"))
    from e13_universal_scoring import FEATS  # noqa
    from e3_physical_entropy import kmer_groups
    g = kmer_groups([r["seq"] for r in cr], 0.3)
    for r, gi in zip(cr, g):
        r["grp"] = f"cr_{int(gi)}"
    Path("/tmp/e14_cr.json").write_text(json.dumps(cr))
    print(f"  -> {len(cr)} crystal")
    print("PEPBI features...")
    pb = pepbi()
    for r in pb:
        r["grp"] = f"pb_{r['grp']}"
    Path("/tmp/e14_pb.json").write_text(json.dumps(pb))
    print(f"  -> {len(pb)} pepbi")


if __name__ == "__main__":
    main()
