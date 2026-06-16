"""E242 — GIST (Grid Inhomogeneous Solvation Theory) = the OPEN, already-installed WaterMap equivalent.
Unlike E230 3D-RISM (which gave water DENSITY only), GIST decomposes pocket hydration into per-voxel FREE
ENERGY: solute-water (Esw) + water-water (Eww) enthalpy and translational/orientational entropy
(dTStrans/dTSorient). Integrated over the binding pocket this is the displaceable hydration free energy —
the actual WaterMap signal that density (max_g) could not see. cpptraj has `gist` built in (ambertools).

Per receptor:  apo PDB (reuse runs/e230_rism/{pdb}/) -> tleap solvate TIP3P -> OpenMM restrained MD
(min/NPT-equil/NVT-prod, DCD) -> cpptraj rms-fit + gist on a grid centered on the pocket -> integrate
pocket descriptors:  dG_pocket, Esw, Eww, -TdS, n_unhappy_sites, max_site_dG.

Smoke (validate pipeline + descriptor sanity BEFORE any campaign):
  python3 scripts/e242_gist_pipeline.py --pdb 4e34 --equil-ps 50 --prod-ps 500
Then (campaign over multi-binders): --manifest ... --prod-ps 4000   (slow: explicit-solvent MD)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e229_pocket_md_pilot as e229  # site_and_apo

AMBER = Path("/home/igem/miniconda3/envs/ambertools")
TLEAP = AMBER / "bin" / "tleap"
CPPTRAJ = AMBER / "bin" / "cpptraj"
WORK = ROOT / "runs" / "e242_gist"; WORK.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data" / "e228_pilot_manifest.json"
CACHE = ROOT / "data" / "e242_gist.jsonl"
ENV = {**os.environ, "AMBERHOME": str(AMBER), "PATH": f"{AMBER/'bin'}:{os.environ.get('PATH','')}"}
POCKET_R = 6.0   # A, sphere over which pocket descriptors are integrated
GRID_HALF = 12.0  # A, half-extent of the GIST grid box around the pocket center


def solvate(apo_pdb: Path, wd: Path):
    """tleap: ff14SB + TIP3P box (12 A pad) + neutralize -> solv.prmtop / solv.inpcrd."""
    leapin = wd / "solv.in"
    leapin.write_text(
        "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
        f"mol = loadpdb {apo_pdb.name}\n"
        "solvateBox mol TIP3PBOX 12.0\naddIons mol Na+ 0\naddIons mol Cl- 0\n"
        "saveamberparm mol solv.prmtop solv.inpcrd\nquit\n")
    subprocess.run([str(TLEAP), "-f", "solv.in"], cwd=wd, env=ENV, capture_output=True, timeout=600)
    prm, crd = wd / "solv.prmtop", wd / "solv.inpcrd"
    if not (prm.exists() and crd.exists()):
        raise RuntimeError("tleap solvate failed")
    return prm, crd


NONPROT = ("HOH", "WAT", "NA", "CL", "Na+", "Cl-")


def solvate_translation(apo_pdb: Path, prm: Path, crd: Path):
    """tleap rigid-translates the solute when it centers the box. Recover that translation from the
    heavy-atom centroid shift (no rotation), so we can map the pocket site into the inpcrd frame."""
    from openmm import unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, PDBFile
    apo = PDBFile(str(apo_pdb))
    apo_heavy = np.array([apo.positions[i].value_in_unit(unit.angstrom)
                          for i, a in enumerate(apo.topology.atoms()) if a.element.symbol != "H"])
    top = AmberPrmtopFile(str(prm)); inp = AmberInpcrdFile(str(crd))
    pos = np.array(inp.positions.value_in_unit(unit.angstrom))
    inp_heavy = np.array([pos[a.index] for a in top.topology.atoms()
                          if a.residue.name not in NONPROT and a.element is not None
                          and a.element.symbol != "H"])
    return inp_heavy.mean(0) - apo_heavy.mean(0)   # translation vector (A)


def write_min_eq_prod(prm, crd, wd, equil_ps, prod_ps):
    """OpenMM: restrain protein heavy atoms, minimize, NPT equil, NVT production -> prod.dcd."""
    import openmm as mm
    from openmm import app, unit
    prmtop = app.AmberPrmtopFile(str(prm)); inpcrd = app.AmberInpcrdFile(str(crd))
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.HBonds, rigidWater=True)
    # positional restraint on protein heavy atoms
    rest = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", 10.0 * unit.kilocalories_per_mole / unit.angstrom**2)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    prot_heavy = []
    for atom in prmtop.topology.atoms():
        if atom.residue.name not in ("HOH", "WAT", "NA", "CL", "Na+", "Cl-") and atom.element is not None \
           and atom.element.symbol != "H":
            rest.addParticle(atom.index, inpcrd.positions[atom.index].value_in_unit(unit.nanometer))
            prot_heavy.append(atom.index)
    system.addForce(rest)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    plat = mm.Platform.getPlatformByName(
        next((n for n in ("CUDA", "OpenCL", "CPU")
              if _has_platform(n)), "CPU"))
    sim = app.Simulation(prmtop.topology, system, integ, plat)
    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    sim.minimizeEnergy(maxIterations=5000)
    # NVT warmup BEFORE the barostat (a bad early volume move on an unsettled box is the NaN cause)
    sim.context.setVelocitiesToTemperature(50 * unit.kelvin)
    sim.step(5000)                                        # 10 ps NVT warmup
    # NPT equilibration: add barostat now that the box has relaxed
    barostat = mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin)
    bidx = system.addForce(barostat)
    sim.context.reinitialize(preserveState=True)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(int(equil_ps / 0.002))                       # NPT equilibration
    # switch to NVT for production (GIST needs fixed-volume frame)
    system.removeForce(bidx)
    sim.context.reinitialize(preserveState=True)
    sim.reporters.append(app.DCDReporter(str(wd / "prod.dcd"), 500))   # 1 ps stride
    sim.step(int(prod_ps / 0.002))
    return wd / "prod.dcd"


def _has_platform(name):
    import openmm as mm
    try:
        mm.Platform.getPlatformByName(name); return True
    except Exception:
        return False


def run_gist(prm, crd, dcd, gridcntr, wd):
    nx = int(2 * GRID_HALF / 0.5)
    cx, cy, cz = gridcntr
    cin = wd / "gist.in"
    cin.write_text(
        f"parm {prm.name}\nreference {crd.name}\ntrajin {dcd.name}\n"
        "autoimage\nrms reference @CA,C,N&!:WAT,HOH\n"
        f"gist gridcntr {cx:.2f} {cy:.2f} {cz:.2f} griddim {nx} {nx} {nx} gridspacn 0.5 "
        "prefix gist out gist-out.dat\nrun\nquit\n")
    p = subprocess.run([str(CPPTRAJ), "-i", "gist.in"], cwd=wd, env=ENV, capture_output=True,
                       timeout=3600, text=True)
    out = wd / "gist-out.dat"
    if not out.exists():
        raise RuntimeError(f"gist failed: {p.stderr[-400:]}")
    return out


def parse_gist(out: Path, gridcntr):
    """Integrate BULK-REFERENCED per-voxel free-energy density over the pocket sphere. Entropy and Esw
    columns are bulk-referenced by GIST construction; Eww-dens is UNREF, so subtract a self-calibrated
    bulk water-water baseline (median over far, normal-density voxels)."""
    cols = out.read_text().splitlines()[1].split()
    data = np.genfromtxt(out, skip_header=2)
    if data.ndim == 1 or data.size == 0:
        raise RuntimeError("empty gist grid")
    idx = {c: i for i, c in enumerate(cols)}
    C = lambda key: data[:, idx[key]]
    xyz = data[:, 1:4]
    d = np.linalg.norm(xyz - np.array(gridcntr), axis=1)
    g_O = C("g_O")
    esw = C("Esw-dens(kcal/mol/A^3)")
    eww_u = C("Eww-dens(kcal/mol/A^3)")
    dtt = C("dTStrans-dens(kcal/mol/A^3)"); dto = C("dTSorient-dens(kcal/mol/A^3)")
    # self-calibrated bulk water-water energy density at g_O=1
    bulk = (d > 10.0) & (g_O > 0.8) & (g_O < 1.2)
    bulk_eww = float(np.median((eww_u[bulk] / g_O[bulk]))) if bulk.sum() > 50 else -0.30
    deww = eww_u - bulk_eww * g_O                  # water-water referenced to bulk
    vox = 0.5 ** 3
    dG = (esw + deww - dtt - dto) * vox            # referenced per-voxel free energy (kcal/mol)
    pk = d <= POCKET_R
    dGp = dG[pk]
    return {
        "gist_dG_pocket": float(dGp.sum()),
        "gist_Esw": float((esw[pk] * vox).sum()),
        "gist_dEww": float((deww[pk] * vox).sum()),
        "gist_mTdS": float((-(dtt + dto)[pk] * vox).sum()),
        "gist_unhappy_dG": float(dGp[dGp > 0].sum()),   # displaceable (WaterMap) reward
        "gist_happy_dG": float(dGp[dGp < 0].sum()),     # costly-to-displace water
        "gist_max_vox_dG": float(dGp.max()) if pk.any() else 0.0,
        "gist_n_pocket_vox": int(pk.sum()),
        "gist_bulk_eww": bulk_eww,
    }


def run_one(pdb, seq, pep_ch, equil_ps, prod_ps):
    site, apo = e229.site_and_apo(pdb, seq, pep_ch)
    if apo is None:
        raise RuntimeError("no apo/site")
    # prefer the RISM-validated apo (already passed tleap for the RISM run) over a fresh clean
    rism_apo = ROOT / "runs" / "e230_rism" / pdb / "apo_amber.pdb"
    src = rism_apo if rism_apo.exists() else Path(apo)
    wd = WORK / pdb; wd.mkdir(exist_ok=True)
    apo2 = wd / "apo.pdb"; apo2.write_bytes(src.read_bytes())
    prm, crd = solvate(apo2, wd)
    gridcntr = np.asarray(site, float) + solvate_translation(apo2, prm, crd)   # pocket in inpcrd frame
    dcd = write_min_eq_prod(prm, crd, wd, equil_ps, prod_ps)
    out = run_gist(prm, crd, dcd, gridcntr, wd)
    desc = parse_gist(out, gridcntr)
    # cleanup big files
    for f in (dcd, wd / "gist-gO.dx", wd / "gist-gH.dx"):
        Path(f).unlink(missing_ok=True)
    return desc | {"gridcntr": [float(x) for x in gridcntr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=2000.0)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    man = json.load(open(a.manifest))["receptors"]
    if a.pdb:
        man = [r for r in man if r["peptides"][0]["pdb"] == a.pdb]
    done = {json.loads(l)["rep_pdb"] for l in CACHE.read_text().splitlines()} if CACHE.exists() else set()
    man = [r for r in man if r["peptides"][0]["pdb"] not in done]
    man.sort(key=lambda r: r.get("receptor_len", 9999))   # cheapest MD first
    if a.limit:
        man = man[: a.limit]
    print(f"=== E242 GIST: {len(man)} receptor(s) TODO ({len(done)} done), "
          f"equil={a.equil_ps}ps prod={a.prod_ps}ps ===", flush=True)
    for rc in man:
        rep = rc["peptides"][0]; pdb = rep["pdb"]; t0 = time.time()
        try:
            d = run_one(pdb, rep["seq"], rep["pep_ch"], a.equil_ps, a.prod_ps)
            row = {"rep_pdb": pdb, "n_pep": rc["n_pep"], "y_mean": rc["y_mean"], "y_std": rc["y_std"], **d}
            with open(CACHE, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {pdb} dG_pocket={d['gist_dG_pocket']:+.1f} unhappy={d['gist_unhappy_dG']:+.1f} "
                  f"Esw={d['gist_Esw']:+.1f} dEww={d['gist_dEww']:+.1f} -TdS={d['gist_mTdS']:+.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} FAILED: {e}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
