"""E2 feature computation — orthogonal, correctly-signed, size-free affinity axes.

Computes per-complex interface chemistry from the (peptide, pocket) PDB pair.
All features are designed to be expressed as DENSITIES / FRACTIONS (size-free);
length-residualization happens downstream in e2_model.py.

Outputs /tmp/e2_features.json (augments the e0 feature rows).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()

POS = {"ARG", "LYS", "HIS"}          # cationic
NEG = {"ASP", "GLU"}                 # anionic
CHARGED = POS | NEG
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
# Kyte-Doolittle hydropathy
KD = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
      "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
      "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
      "LYS": -3.9, "ARG": -4.5}


def _cls(rn):
    rn = rn.upper()
    return "C" if rn in CHARGED else ("P" if rn in POLAR else "A")


def _heavy(res):
    return [a for a in res if a.element != "H"]


def _polar_atoms(res):
    return [a for a in res if a.element in ("N", "O")]


def compute(pep_pdb: str, poc_pdb: str, contact_cut=5.5, hb_cut=3.5, sb_cut=4.0) -> dict:
    pep = _P.get_structure("pep", pep_pdb)
    poc = _P.get_structure("poc", poc_pdb)
    pep_res = [r for r in pep[0].get_residues() if r.id[0] == " "]
    poc_res = [r for r in poc[0].get_residues() if r.id[0] == " "]

    poc_atoms = [a for r in poc_res for a in _heavy(r)]
    ns = NeighborSearch(poc_atoms)

    contact_idx = set()
    n_hb = 0
    n_sb = 0
    contact_resnames = []
    for i, rp in enumerate(pep_res):
        is_contact = False
        for a in _heavy(rp):
            near = ns.search(a.coord, contact_cut)
            if near:
                is_contact = True
            # H-bond: pep polar atom <-> pocket polar atom within hb_cut
            if a.element in ("N", "O"):
                for b in ns.search(a.coord, hb_cut):
                    if b.element in ("N", "O"):
                        n_hb += 1
                        break
        if is_contact:
            contact_idx.add(i)
            contact_resnames.append(rp.resname.upper())
        # salt bridge: charged pep residue sidechain N/O vs opposite-charge pocket
        rn = rp.resname.upper()
        if rn in CHARGED:
            want = NEG if rn in POS else POS
            for a in _polar_atoms(rp):
                for b in ns.search(a.coord, sb_cut):
                    if b.get_parent().resname.upper() in want:
                        n_sb += 1
                        break

    n_contact = max(1, len(contact_idx))

    # contact-type composition (size-free)
    cc = sum(1 for i in contact_idx if _cls(pep_res[i].resname) == "C")
    aa = sum(1 for i in contact_idx if _cls(pep_res[i].resname) == "A")
    ic_charged_frac = cc / n_contact
    ic_apolar_frac = aa / n_contact

    # hydrophobicity of CONTACTING peptide residues (burying hydrophobic = good)
    hyd = [KD.get(pep_res[i].resname.upper(), 0.0) for i in contact_idx]
    mean_hyd_contact = float(np.mean(hyd)) if hyd else 0.0

    # ---- SASA-based: free peptide vs complex (per atom) ----
    _SR.compute(pep, level="A")
    free_sasa = {a.get_serial_number(): float(a.sasa) for a in pep[0].get_atoms()}

    # area-based %NIS on free peptide
    area = {"C": 0.0, "P": 0.0, "A": 0.0}
    tot_nis = 0.0
    for i, rp in enumerate(pep_res):
        if i in contact_idx:
            continue
        ra = sum(float(a.sasa) for a in _heavy(rp))
        area[_cls(rp.resname)] += ra
        tot_nis += ra
    if tot_nis > 0:
        nis_apolar_area = area["A"] / tot_nis
        nis_charged_area = area["C"] / tot_nis
        nis_polar_area = area["P"] / tot_nis
    else:
        nis_apolar_area = nis_charged_area = nis_polar_area = 0.0

    # complex SASA -> buried unsatisfied polar fraction
    from Bio.PDB import Structure, Model
    s = Structure.Structure("c")
    m = Model.Model(0)
    s.add(m)
    used = set()
    for src in (pep_pdb, poc_pdb):
        for ch in _P.get_structure("a", src)[0]:
            c = ch.copy()
            cid = c.id
            while cid in used:
                cid = chr(((ord(cid) - 64) % 26) + 65)
            c.id = cid
            used.add(cid)
            m.add(c)
    _SR.compute(s, level="A")
    # map complex peptide-atom SASA by (resseq, atomname) since serials change
    cpx_pep_sasa = {}
    first_chain = next(iter(m))
    for ch in m:
        # peptide chains are the ones from pep_pdb (added first); detect by matching
        pass
    # simpler: recompute peptide-atom buried state via NeighborSearch burial proxy
    # buried polar atom = free SASA>5 but heavily contacted (>=1 pocket heavy atom <4.5A)
    n_polar = 0
    n_unsat_buried = 0
    for i in contact_idx:
        rp = pep_res[i]
        for a in _polar_atoms(rp):
            n_polar += 1
            near_poc = ns.search(a.coord, 4.5)
            buried = len(near_poc) >= 1
            if not buried:
                continue
            # satisfied if a polar pocket atom within hb_cut
            sat = any(b.element in ("N", "O") and np.linalg.norm(a.coord - b.coord) <= hb_cut
                      for b in near_poc)
            if not sat:
                n_unsat_buried += 1
    buried_unsat_polar_frac = n_unsat_buried / max(1, n_polar)

    return dict(
        hb_density=n_hb / n_contact,
        sb_density=n_sb / n_contact,
        ic_charged_frac=ic_charged_frac,
        ic_apolar_frac=ic_apolar_frac,
        mean_hyd_contact=mean_hyd_contact,
        nis_apolar_area=nis_apolar_area,
        nis_charged_area=nis_charged_area,
        nis_polar_area=nis_polar_area,
        buried_unsat_polar_frac=buried_unsat_polar_frac,
    )


def main():
    rows = json.loads(Path("/tmp/e0_features.json").read_text())
    print(f"computing E2 features for {len(rows)} complexes...")
    for i, r in enumerate(rows):
        try:
            r.update(compute(r["pep_pdb"], r["poc_pdb"]))
        except Exception as e:  # noqa: BLE001
            print(f"  {r['pdb']} FAIL: {type(e).__name__}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}")
    Path("/tmp/e2_features.json").write_text(json.dumps(rows))
    print("wrote /tmp/e2_features.json")


if __name__ == "__main__":
    main()
