"""Pose geometry + per-contact-energy features for the ensemble scorer (SCORE-ENS).

Extracts, from a peptide pose PDB + receptor PDB, the descriptors the geometry+Vina
ensemble consumes (src/hybridock_pep/scoring/ensemble.py):

  pocket descriptors  : composition of receptor residues lining the bound peptide
                        (size, hydrophobic/aromatic/charged fractions, mean hydropathy)
  interface features  : per-residue buried SASA split by favourability + interface
                        H-bond / salt-bridge / aromatic counts
  mj_contact          : Σ Miyazawa-Jernigan contact energy over peptide-receptor residue
                        contacts — the per-contact ENERGY term that captures hotspot
                        residues (Trp/Phe) a contact COUNT misses (docs E24: +0.04 r)

Free-state peptide SASA is computed from the peptide chain in isolation (deployment has
the pose, not a separate apo peptide). Pure Biopython + Shrake-Rupley; no GPU, ~150 ms/pose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.SASA import ShrakeRupley

from hybridock_pep.scoring.mj_potential import MJ_ENERGY

_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()

_AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
           "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
           "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
           "TYR": "Y", "VAL": "V"}
_EISENBERG = {"I": 0.73, "F": 0.61, "V": 0.54, "L": 0.53, "W": 0.37, "M": 0.26, "A": 0.25,
              "G": 0.16, "C": 0.04, "Y": 0.02, "P": -0.07, "T": -0.18, "S": -0.26,
              "H": -0.40, "E": -0.62, "N": -0.64, "Q": -0.69, "D": -0.72, "K": -1.10,
              "R": -1.76}
_POS, _NEG = {"ARG", "LYS", "HIS"}, {"ASP", "GLU"}
_CHARGED = _POS | _NEG
_POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
_APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "GLY"}
_AROM = {"PHE", "TYR", "TRP", "HIS"}
_HPHOBIC_AA = set("AVLIMFWC")

# Experimental per-residue interface binding strength: mean ΔΔG of an X->Ala mutation over
# ~7000 SKEMPI 2.0 interface mutations (kcal/mol, >0 = X is a hotspot). Recovers the Bogan-Thorn
# ranking blind (W>F>Y>L>I top, Ser bottom). Used as a burial-weighted intensive term that, unlike
# the size-confounded statistical mj_contact, is sign-consistent across datasets (docs E46:
# strength_bur −0.283 crystal-65 / −0.124 the-98). Residues absent here (A, C) are skipped, matching
# the validated feature. data/skempi_v2.csv (gitignored); scripts/e46_skempi_strength.py.
_SKEMPI_STRENGTH = {
    "W": 2.1636, "F": 1.5713, "Y": 1.5614, "L": 1.2342, "I": 1.1869, "K": 1.1405,
    "D": 1.1343, "R": 1.1273, "H": 1.0038, "G": 0.8263, "M": 0.8222, "E": 0.7864,
    "T": 0.7855, "N": 0.6634, "P": 0.6418, "V": 0.6291, "Q": 0.5074, "S": 0.2600,
}

GEOMETRY_FEATURE_KEYS = [
    "poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis",
    "bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count", "mj_contact", "strength_bur",
]


def _merge_complex(peptide_pdb: Path, receptor_pdb: Path) -> Path:
    """Write a merged complex (peptide chain P, receptor chain R), waters stripped."""
    out = Path("/tmp") / f"_ens_{peptide_pdb.stem}_{receptor_pdb.stem}.pdb"
    lines = []
    for src, ch in ((peptide_pdb, "P"), (receptor_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    out.write_text("\n".join(lines) + "\nEND\n")
    return out


def _per_res_sasa(struct) -> dict:
    _SR.compute(struct, level="A")
    return {(r.get_parent().id, r.id): sum(float(a.sasa) for a in r)
            for r in struct.get_residues() if r.id[0] == " "}


def _pocket_descriptors(cx_model, pep_chain: str, radius: float = 8.0) -> dict | None:
    pep_xyz = np.array([a.coord for r in cx_model[pep_chain] if r.id[0] == " "
                        for a in r if a.element != "H"])
    if len(pep_xyz) == 0:
        return None
    r2 = radius * radius
    poc = []
    for ch in cx_model:
        if ch.id == pep_chain:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            for a in res:
                if a.element != "H" and np.min(((pep_xyz - a.coord) ** 2).sum(1)) <= r2:
                    poc.append(res.resname.upper())
                    break
    n = len(poc)
    if n == 0:
        return None
    names = [_AA3TO1.get(x, "A") for x in poc]
    return dict(
        poc_n=float(n),
        poc_f_hyd=sum(a in _HPHOBIC_AA for a in names) / n,
        poc_f_arom=sum(x in _AROM for x in poc) / n,
        poc_net=(sum(x in _POS for x in poc) - sum(x in _NEG for x in poc)) / n,
        poc_eis=float(np.mean([_EISENBERG.get(a, 0.0) for a in names])),
    )


def _interface_features(peptide_pdb: Path, cx_path: Path, pep_chain: str) -> dict | None:
    free = _per_res_sasa(_P.get_structure("f", str(peptide_pdb)))
    cpx = _per_res_sasa(_P.get_structure("c", str(cx_path)))
    cx = _P.get_structure("cc", str(cx_path))[0]
    pep_res = [r for r in cx[pep_chain] if r.id[0] == " "]
    rec_atoms = [a for ch in cx if ch.id != pep_chain for r in ch if r.id[0] == " "
                 for a in r if a.element != "H"]
    if not pep_res or not rec_atoms:
        return None
    ns = NeighborSearch(rec_atoms)
    pf = [r for r in _P.get_structure("pp", str(peptide_pdb))[0].get_residues()
          if r.id[0] == " "]
    n = min(len(pep_res), len(pf))
    bsa_hyd = sasa_hb = sasa_sb = 0.0
    hb_count = arom_cc = 0
    s_wsum = s_wnorm = 0.0  # burial-weighted experimental strength accumulators
    for i in range(n):
        rc = pep_res[i]
        rn = rc.resname.upper()
        aa = _AA3TO1.get(rn, "A")
        rfree = free.get((pf[i].get_parent().id, pf[i].id), 0.0)
        rbound = cpx.get((rc.get_parent().id, rc.id), 0.0)
        dsasa = max(0.0, rfree - rbound)
        if aa in _SKEMPI_STRENGTH:  # accumulate over ALL residues, weighted by buried fraction
            bur = dsasa / (rfree + 1e-6)
            s_wsum += bur * _SKEMPI_STRENGTH[aa]
            s_wnorm += bur
        if dsasa < 1.0:
            continue
        if aa in _HPHOBIC_AA:
            bsa_hyd += dsasa
        has_hb = has_sb = has_arom = False
        for a in rc:
            if a.element == "H":
                continue
            for b in ns.search(a.coord, 5.5):
                d = float(np.linalg.norm(a.coord - b.coord))
                brn = b.get_parent().resname.upper()
                if a.element in ("N", "O") and b.element in ("N", "O") and d <= 3.5:
                    has_hb = True
                if rn in _CHARGED and a.element in ("N", "O") and d <= 4.0:
                    want = _NEG if rn in _POS else _POS
                    if brn in want:
                        has_sb = True
                if rn in _AROM and brn in _AROM and d <= 5.5:
                    has_arom = True
        if has_hb:
            sasa_hb += dsasa
            hb_count += 1
        if has_sb:
            sasa_sb += dsasa
        if has_arom:
            arom_cc += 1
    return dict(bsa_hyd=bsa_hyd / 100, sasa_hb=sasa_hb / 100, sasa_sb=sasa_sb / 100,
                arom_cc=float(arom_cc), hb_count=float(hb_count),
                strength_bur=float(s_wsum / (s_wnorm + 1e-6)))


def _mj_contact(cx_path: Path, pep_chain: str, contact_cut: float = 6.5) -> float:
    cx = _P.get_structure("c", str(cx_path))[0]
    pep = [r for r in cx[pep_chain] if r.id[0] == " "]
    rec = [r for ch in cx if ch.id != pep_chain for r in ch if r.id[0] == " "]
    if not pep or not rec:
        return 0.0
    rec_atoms = [a for r in rec for a in r if a.element != "H"]
    ns = NeighborSearch(rec_atoms)
    seen: set = set()
    total = 0.0
    for rp in pep:
        a1 = _AA3TO1.get(rp.resname.upper(), "A")
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
            a2 = _AA3TO1.get(rr.resname.upper(), "A")
            total += MJ_ENERGY.get((a1, a2), -1.5)
    return total


def compute_geometry_features(peptide_pose_pdb: Path, receptor_pdb: Path) -> dict | None:
    """Extract the ensemble's geometry + per-contact-energy features for one pose.

    Args:
        peptide_pose_pdb: PDB of the peptide pose (the docked/generated conformation).
        receptor_pdb: PDB of the receptor (full or pocket).

    Returns:
        Dict with GEOMETRY_FEATURE_KEYS, or None if the pose has no usable interface.
    """
    peptide_pose_pdb = Path(peptide_pose_pdb)
    receptor_pdb = Path(receptor_pdb)
    cx = _merge_complex(peptide_pose_pdb, receptor_pdb)
    try:
        pk = _pocket_descriptors(_P.get_structure("m", str(cx))[0], "P")
        fi = _interface_features(peptide_pose_pdb, cx, "P")
        if pk is None or fi is None:
            return None
        return {**pk, **fi, "mj_contact": _mj_contact(cx, "P")}
    finally:
        cx.unlink(missing_ok=True)
