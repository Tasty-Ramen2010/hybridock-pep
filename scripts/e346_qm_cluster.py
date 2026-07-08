"""E346 — cheap-QM leg for --ultra-charged: GFN2-xTB cluster charge-morph (the buried-H-bond fix).

ECC + continuum fix the salt-bridge OVERstabilization but CAN'T touch the buried-polar/aromatic-H-bond class
(1IAR: Glu9 H-bonded to Tyr13/127/183, wrong-sign everywhere) — that needs quantum electronic structure
(directional H-bonds + polarization). This is the cheapest honest QM: semi-empirical GFN2-xTB on a small interface
CLUSTER, with ALPB(water) implicit solvent.

Method (a QM charge-morph double-difference — capping errors cancel because only the acidic proton toggles):
  cluster_bound = mutated residue X + partner-chain residues within R_cut of X's sidechain
  cluster_free  = X + its OWN-chain neighbours within R_cut (partner removed)
  ΔΔG_QM = [E(bound, X charged COO⁻) − E(bound, X neutral COOH)]
         − [E(free,  X charged COO⁻) − E(free,  X neutral COOH)]
Neutral state = protonate the carboxylate (add 1 H, charge +1 relative to charged): the isosteric
charge-neutralising mutation's dominant effect is loss of the −1, and QM captures the resulting H-bond/polarisation
change that MM misses. >0 ⇒ WT charge helps binding (matches SKEMPI sign).

Standalone test: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e346_qm_cluster.py
"""
from __future__ import annotations
import sys, os, subprocess, tempfile
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch

XTB = "/home/igem/miniconda3/envs/qm-env/bin/xtb"
HARTREE = 627.509                                   # kcal/mol per hartree
ACID = {"D": ("ASP", "OD1", "OD2", "CG"), "E": ("GLU", "OE1", "OE2", "CD")}
FORMAL = {"ASP": -1, "GLU": -1, "LYS": +1, "ARG": +1}   # HIS treated neutral
R_CUT = 5.0
ELEM = {"C": "C", "N": "N", "O": "O", "S": "S", "H": "H", "P": "P"}


def _elem(atom):
    e = atom.element.strip().capitalize() if atom.element.strip() else atom.name[0]
    return ELEM.get(e, e if e in ("F", "Cl", "Br", "I", "Se") else "C")


def _cluster(res_x, neigh_residues):
    """Return (atoms list, net formal charge of NEIGHBOURS excluding X)."""
    atoms, q_neigh = [], 0
    for a in res_x:                                 # whole mutated residue
        atoms.append(a)
    for r in neigh_residues:
        q_neigh += FORMAL.get(r.get_resname().strip(), 0)
        for a in r:
            atoms.append(a)
    return atoms, q_neigh


def _write_xyz(path, atoms, extra=None):
    lines = []
    for a in atoms:
        x, y, z = a.coord
        lines.append(f"{_elem(a):2s} {x:12.5f} {y:12.5f} {z:12.5f}")
    if extra:
        for e, (x, y, z) in extra:
            lines.append(f"{e:2s} {x:12.5f} {y:12.5f} {z:12.5f}")
    open(path, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")


def _cooh_proton(res_x, wt):
    """Position of an added carboxylate proton (on the O further from any neighbour) to neutralise COO⁻→COOH."""
    _, o1, o2, cg = ACID[wt]
    O2 = res_x[o2].coord; C = res_x[cg].coord
    # place H ~0.98 Å from O2 along the C→O2 direction (anti), a reasonable COOH proton
    d = O2 - C; d = d / (np.linalg.norm(d) + 1e-9)
    return ("H", tuple(O2 + 0.98 * d))


def _xtb_energy(atoms, charge, tag, extra=None):
    d = tempfile.mkdtemp(prefix="xtb_")
    xyz = os.path.join(d, "c.xyz"); _write_xyz(xyz, atoms, extra)
    try:
        r = subprocess.run([XTB, xyz, "--gfn", "2", "--alpb", "water", "--chrg", str(charge), "--sp"],
                           cwd=d, capture_output=True, text=True, timeout=300)
        for ln in r.stdout.splitlines():
            if "TOTAL ENERGY" in ln:
                return float(ln.split()[3]) * HARTREE
    except Exception:
        return None
    return None


def qm_ddg(tag, mut):
    pdb = tag.split("_")[0]; groups = tag.split("_")[1:]
    wt, ch, resid = mut[0], mut[1], int(mut[2:-1])
    st = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))[0]
    res_x = next((r for r in st[ch] if r.id[1] == resid and r.get_resname().strip() == ACID[wt][0]), None)
    if res_x is None:
        raise RuntimeError("X residue not found")
    tip = [res_x[n] for n in ACID[wt][1:] if n in res_x]
    partner_chains = set("".join(groups)) - {ch}
    all_atoms = [a for c in st for a in c.get_atoms() if a.element.strip() != ""]
    ns = NeighborSearch(all_atoms)

    def neighbours(from_chains):
        near = set()
        for a in tip:
            for nb in ns.search(a.coord, R_CUT):
                r = nb.get_parent(); pc = r.get_parent().id
                if pc in from_chains and not (pc == ch and r.id[1] == resid) and r.id[0] == " ":
                    near.add(r)
        return list(near)

    hpos = _cooh_proton(res_x, wt)
    out = {}
    for name, chset in (("bound", partner_chains), ("free", {ch})):
        neigh = neighbours(chset if name == "bound" else {ch})
        atoms, qn = _cluster(res_x, neigh)
        e_chg = _xtb_energy(atoms, qn - 1, f"{name}_chg")           # X carries −1
        e_neu = _xtb_energy(atoms, qn, f"{name}_neu", extra=[hpos])  # X protonated → neutral
        if e_chg is None or e_neu is None:
            raise RuntimeError(f"xtb failed ({name}): natoms={len(atoms)} qn={qn}")
        out[name] = (e_chg - e_neu, len(atoms), len(neigh))
    # SKEMPI convention: ΔΔG = ΔG_bind(neutral) − ΔG_bind(charged) > 0 when WT charge helps binding.
    # out[name][0] = E(chg)−E(neu), so ΔΔG = free_diff − bound_diff (matches the classical charged→neutral morph).
    ddg = out["free"][0] - out["bound"][0]
    return ddg, out


def main():
    tests = [("1IAR_A_B", "EA9Q", 3.11), ("2PCB_A_B", "DA34N", 0.82), ("3HFM_HL_Y", "DY101N", 1.49),
             ("1K8R_A_B", "DA38N", 1.97)]
    print("=== E346 GFN2-xTB cluster charge-morph ΔΔG ===", flush=True)
    for tag, mut, exp in tests:
        try:
            ddg, out = qm_ddg(tag, mut)
            print(f"{tag:14s} {mut:8s} QM={ddg:+.2f}  exp={exp:+.2f}  |Δ|={abs(ddg-exp):.2f}  "
                  f"[bound {out['bound'][1]}at/{out['bound'][2]}res, free {out['free'][1]}at/{out['free'][2]}res]",
                  flush=True)
        except Exception as e:
            print(f"{tag:14s} {mut:8s} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)


if __name__ == "__main__":
    main()
