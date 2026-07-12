"""E331 — does Rosetta relaxation rescue ref2015 interface-dG on PDBbind peptides?

Takes a spread of complexes already scored unrelaxed in e329, runs FastRelax (bound complex,
coordinate-constrained so it refines rather than drifts), then InterfaceAnalyzer dG_separated.
Compares relaxed vs unrelaxed ifdG and their correlation with experimental ΔG on the SAME set.

Usage: python experiments/e331_relax_pdbbind.py [--n 25]
Run in score-env.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PL_ROOT = ROOT / "data/drive_pull/pl/P-L"
UNREL = ROOT / "data/e329_ref2015_pdbbind.json"
OUT = ROOT / "data/e331_relax_pdbbind.json"

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def build_dirmap():
    m = {}
    for yd in PL_ROOT.iterdir():
        if yd.is_dir():
            for cd in yd.iterdir():
                if cd.is_dir():
                    m[cd.name] = cd
    return m


def mol2_to_pep_pdb(mol2, out):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            continue
        name, x, y, z, sname = f[1], f[2], f[3], f[4], f[7]
        rn = "".join(c for c in sname if c.isalpha()).upper()[:3]
        if rn not in AA3:
            return None
        try:
            atoms.append((name, rn, float(x), float(y), float(z)))
        except ValueError:
            return None
    rec, seq, resnum = [], [], 0
    for aid, (name, rn, x, y, z) in enumerate(atoms, start=1):
        if name == "N":
            resnum += 1
            seq.append(rn)
        assigned = max(resnum, 1)
        rec.append(f"ATOM  {aid:>5} {name:<4} {rn:>3} P{assigned:>4}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2}")
    if len(seq) < 3:
        return None
    out.write_text("\n".join(rec) + "\nTER\nEND\n")
    return "".join(AA3[r] for r in seq)


def prot_chain_R(prot, out):
    keep = [ln[:21] + "R" + ln[22:] for ln in prot.read_text().splitlines() if ln.startswith("ATOM")]
    out.write_text("\n".join(keep) + "\nTER\nEND\n")


def main():
    n = 25
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    unrel = json.loads(UNREL.read_text())
    unrel.sort(key=lambda r: r["y"])
    # even spread across the affinity range
    pick_idx = np.linspace(0, len(unrel) - 1, n).astype(int)
    picks = [unrel[i] for i in pick_idx]
    unrel_map = {r["pdb"]: r for r in unrel}

    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false", silent=True)
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector, NeighborhoodResidueSelector)
    from pyrosetta.rosetta.core.kinematics import MoveMap

    dirmap = build_dirmap()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["pdb"] for r in out}
    tmp = Path("/tmp/e331"); tmp.mkdir(exist_ok=True)
    sfxn = pyrosetta.get_fa_scorefxn()

    for k, r in enumerate(picks):
        pid = r["pdb"]
        if pid in done:
            continue
        d = dirmap.get(pid)
        if d is None:
            continue
        pep_pdb = tmp / f"{pid}_pep.pdb"; prot_pdb = tmp / f"{pid}_prot.pdb"; merged = tmp / f"{pid}_m.pdb"
        try:
            seq = mol2_to_pep_pdb(d / f"{pid}_ligand.mol2", pep_pdb)
            if not seq:
                continue
            prot_chain_R(d / f"{pid}_protein.pdb", prot_pdb)
            pl = [l for l in pep_pdb.read_text().splitlines() if l.startswith("ATOM")]
            rl = [l for l in prot_pdb.read_text().splitlines() if l.startswith("ATOM")]
            merged.write_text("\n".join(pl) + "\nTER\n" + "\n".join(rl) + "\nTER\nEND\n")
            pose = pyrosetta.pose_from_pdb(str(merged))

            # interface-restricted FastRelax: only the peptide (chain P) + its <8A neighbors move.
            # Local refinement in the spirit of FlexPepDock; ~10-20s vs minutes for whole-complex.
            pep_sel = ChainSelector("P")
            iface = NeighborhoodResidueSelector(pep_sel, 8.0, True)
            subset = iface.apply(pose)
            mm = MoveMap()
            mm.set_bb(False); mm.set_chi(False)
            for i in range(1, pose.total_residue() + 1):
                if subset[i]:
                    mm.set_bb(i, True); mm.set_chi(i, True)
            fr = FastRelax(sfxn, 1)
            fr.constrain_relax_to_start_coords(True)
            fr.set_movemap(mm)
            fr.apply(pose)

            iam = InterfaceAnalyzerMover("P_R")
            iam.set_pack_separated(True)
            iam.apply(pose)
            rlx = float(iam.get_interface_dG())
        except Exception as e:  # noqa: BLE001
            print(f"  {pid} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        finally:
            for p in (pep_pdb, prot_pdb, merged):
                if p.exists():
                    p.unlink()
        row = dict(pdb=pid, y=r["y"], ifdG_unrelaxed=unrel_map[pid]["ros_ifdG"], ifdG_relaxed=rlx)
        out.append(row)
        OUT.write_text(json.dumps(out))
        print(f"  [{k+1}/{len(picks)}] {pid} y={r['y']:+.2f}  unrel={row['ifdG_unrelaxed']:+7.1f}"
              f"  -> relaxed={rlx:+7.1f}", flush=True)

    if len(out) >= 8:
        y = np.array([r["y"] for r in out])
        xu = np.array([r["ifdG_unrelaxed"] for r in out])
        xr = np.array([r["ifdG_relaxed"] for r in out])
        print(f"\n=== relax vs unrelaxed on same n={len(out)} ===")
        print(f"  UNRELAXED ifdG vs ΔG: r={pearsonr(xu,y)[0]:+.3f}  rho={spearmanr(xu,y).statistic:+.3f}")
        print(f"  RELAXED   ifdG vs ΔG: r={pearsonr(xr,y)[0]:+.3f}  rho={spearmanr(xr,y).statistic:+.3f}")
        print(f"  mean ifdG:  unrelaxed={xu.mean():+.1f}  relaxed={xr.mean():+.1f}  "
              f"(relax should pull clash-inflated scores down)")


if __name__ == "__main__":
    main()
