"""E9 — MM-GBSA + interaction entropy WITH short-MD ensemble averaging (GPU).

The thing FEP/LIE have that all our static features lacked: a conformational
ensemble. With CUDA now working on the RTX 5070 (~11s/50ps), we can finally add
it. Per complex:

  1. Build complex (peptide+receptor) in amber14 + GBn2 implicit solvent.
  2. Minimize, equilibrate briefly, run short production MD collecting frames.
  3. Single-trajectory decomposition per frame:
       E_int(frame) = E_complex - E_receptor - E_peptide   (GB implicit)
  4. ⟨E_int⟩ = MM-GBSA ΔG (ensemble-averaged, NOT single static pose).
     -TΔS_IE = kT·ln⟨exp(β(E_int-⟨E_int⟩))⟩  (interaction entropy from fluctuations)
     ΔG_pred = ⟨E_int⟩ - TΔS_IE

Test whether the ENSEMBLE-averaged ΔG_pred correlates with experimental ΔG
better than the static single-pose MM-GBSA did (which was a size meter, r~-0.03
per-residue). Cross-family AND within-family.

Usage: python e9_md_ensemble_ie.py [N_complexes] [prod_ps]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

KT_300 = 0.5961612775922  # kcal/mol
KJ2KCAL = 0.2390057361


def _build_ff():
    import openmm.app as app
    return app.ForceField("amber14-all.xml", "implicit/gbn2.xml")


def _prep(pep_pdb, poc_pdb):
    """Merge peptide+pocket, add H, return (topology, positions, pep_atom_idx, rec_atom_idx)."""
    import openmm.app as app
    from pdbfixer import PDBFixer
    import openmm.unit as unit

    # write a merged PDB then fix
    merged = Path("/tmp/e9_merged.pdb")
    lines = []
    for src, chain in ((pep_pdb, "P"), (poc_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + chain + ln[22:])
    merged.write_text("\n".join(lines) + "\nEND\n")

    fixer = PDBFixer(filename=str(merged))
    # Do NOT rebuild missing residues: the pocket is an intentional crop, and
    # building giant loops across the chain breaks blows up the MD (NaN).
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    # peptide = chain P residues
    pep_idx, rec_idx = [], []
    for atom in fixer.topology.atoms():
        if atom.residue.chain.id == "P":
            pep_idx.append(atom.index)
        else:
            rec_idx.append(atom.index)
    return fixer.topology, fixer.positions, pep_idx, rec_idx


def _subsystem_energy(ff, topology, positions, keep_idx, platform):
    """Energy (kcal/mol) of only the atoms in keep_idx, via a deleted-atom Modeller."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    modeller = app.Modeller(topology, positions)
    keep = set(keep_idx)
    to_delete = [a for a in topology.atoms() if a.index not in keep]
    modeller.delete(to_delete)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                             constraints=None)
    integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = mm.Context(system, integ, platform)
    ctx.setPositions(modeller.positions)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx, integ
    return e * KJ2KCAL


def run_complex(pep_pdb, poc_pdb, prod_ps=100, frame_every_ps=5, platform_name="CUDA"):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    ff = _build_ff()
    topo, pos, pep_idx, rec_idx = _prep(pep_pdb, poc_pdb)
    plat = mm.Platform.getPlatformByName(platform_name)

    system = ff.createSystem(topo, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                        0.002 * unit.picoseconds)
    sim = app.Simulation(topo, system, integ, plat)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=1000)
    # gentle equilibration; bail early if it blows up
    sim.step(2500)  # 5 ps equilibration
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(e0):
        raise ValueError("equilibration diverged (NaN energy)")

    n_frames = max(1, prod_ps // frame_every_ps)
    steps_per_frame = int(frame_every_ps / 0.002)
    e_ints = []
    for _ in range(n_frames):
        sim.step(steps_per_frame)
        state = sim.context.getState(getPositions=True, getEnergy=True)
        if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
            break  # trajectory diverged; use frames collected so far
        p = state.getPositions()
        e_cplx = _subsystem_energy(ff, topo, p, pep_idx + rec_idx, plat)
        e_rec = _subsystem_energy(ff, topo, p, rec_idx, plat)
        e_pep = _subsystem_energy(ff, topo, p, pep_idx, plat)
        e_ints.append(e_cplx - e_rec - e_pep)
    e_ints = np.array(e_ints)
    e_mean = float(e_ints.mean())
    # interaction entropy from fluctuations (log-sum-exp stable)
    dE = e_ints - e_mean
    beta = 1.0 / KT_300
    minus_tds = KT_300 * (np.log(np.mean(np.exp(np.clip(beta * dE, -50, 50)))))
    return dict(e_int_mean=e_mean, e_int_std=float(e_ints.std()),
                minus_tds_ie=float(minus_tds), dg_pred=e_mean + float(minus_tds),
                n_frames=len(e_ints))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    prod_ps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rows = json.loads(Path("/tmp/e0_rows.json").read_text())
    rows = [r for r in rows if r.get("pep_pdb")][:n]
    print(f"E9: MD-ensemble MM-GBSA+IE on {len(rows)} complexes, {prod_ps}ps each")
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        ts = time.time()
        try:
            res = run_complex(r["pep_pdb"], r["poc_pdb"], prod_ps=prod_ps)
            res.update(pdb=r["pdb"], y=r["y"], L=r["L"], aff=r["aff"])
            out.append(res)
            print(f"  [{i+1}/{len(rows)}] {r['pdb']}: "
                  f"<E_int>={res['e_int_mean']:.1f} -TdS_IE={res['minus_tds_ie']:.1f} "
                  f"dG_pred={res['dg_pred']:.1f}  (exp {r['y']:.1f})  "
                  f"{time.time()-ts:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(rows)}] {r['pdb']}: FAILED {type(e).__name__}: {str(e)[:100]}")
    Path("/tmp/e9_results.json").write_text(json.dumps(out))
    print(f"total {time.time()-t0:.1f}s  ({(time.time()-t0)/max(1,len(out)):.1f}s/complex)")
    if len(out) >= 3:
        from scipy.stats import pearsonr
        y = np.array([o["y"] for o in out])
        for k in ("e_int_mean", "dg_pred"):
            v = np.array([o[k] for o in out])
            if np.std(v) > 0:
                print(f"  pearson({k}, exp) = {pearsonr(v, y).statistic:+.3f} (n={len(out)})")


if __name__ == "__main__":
    main()
