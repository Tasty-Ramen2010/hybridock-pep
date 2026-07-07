"""E338 — decisive test: does the POLARIZABLE FF (AMOEBA) give Asp75 a bigger charged binding contribution than
fixed-charge amber14? (Is polarization actually our fix, before committing to a full polarizable FEP?)

The E337 failure is under-estimating a buried salt bridge; JACS 2022 says fixed-charge FFs under-stabilise buried
ion pairs and polarizable FFs fix it. This checks it directly on OUR case (2O3B D75N, Asp75-Lys101) with a clean
single-point Hamiltonian comparison (implicit solvent, no alchemy, no sampling):
  charged-contribution(FF) = E(full) - E(Asp75 electrostatics zeroed)     [in bound and free]
  Asp75 binding contribution(FF) = contribution(bound) - contribution(free)
If AMOEBA's binding contribution >> amber14's, polarization is the lever → a polarizable/hybrid FEP is worth
building. If they're similar, polarization is NOT our problem and we save weeks.

Single-structure (minimised), so it is a HAMILTONIAN comparison, not the thermal ΔΔG — but it directly answers
"does AMOEBA make this salt bridge stronger?".

Run: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e338_amoeba_vs_amber_saltbridge.py
"""
from __future__ import annotations
import sys, tempfile
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch, ChainSel
from Bio.PDB import PDBParser, PDBIO
import openmm as mm
from openmm import app, unit

MUT_CHAIN, RESID = "B", 75            # Asp75
SIDE = {"CB", "CG", "OD1", "OD2", "HB2", "HB3"}   # Asp side chain
PLAT = mm.Platform.getPlatformByName("CUDA")


def prep(chains):
    from pdbfixer import PDBFixer
    st = PDBParser(QUIET=True).get_structure("2o3b", fetch("2O3B"))
    tmp = tempfile.mktemp(suffix=".pdb"); io = PDBIO(); io.set_structure(st); io.save(tmp, ChainSel(chains))
    fx = PDBFixer(filename=tmp)
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    return fx.topology, fx.positions


def asp_atoms(topology):
    return [a.index for a in topology.atoms()
            if a.residue.chain.id == MUT_CHAIN and int(a.residue.id) == RESID and a.name in SIDE]


def amber_contrib(topology, positions):
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond), PLAT)
    ctx.setPositions(positions)
    e_full = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    saved = {}
    for i in asp_atoms(topology):
        q, sig, eps = nb.getParticleParameters(i); saved[i] = q
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sig, eps)
    nb.updateParametersInContext(ctx)
    e_off = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return e_full - e_off   # stabilisation from Asp75 charges (negative = favorable)


def amoeba_contrib(topology, positions):
    ff = app.ForceField("amoeba2018.xml", "amoeba2018_gk.xml")
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, polarization="mutual",
                             mutualInducedTargetEpsilon=1e-5)
    mp = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "AmoebaMultipoleForce")
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond), PLAT)
    ctx.setPositions(positions)
    e_full = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    for i in asp_atoms(topology):
        p = list(mp.getMultipoleParameters(i))
        p[0] = 0.0 * unit.elementary_charge                      # charge
        p[1] = [0, 0, 0] * unit.elementary_charge * unit.nanometer   # dipole
        p[2] = [0] * 9 * unit.elementary_charge * unit.nanometer ** 2  # quadrupole
        p[-1] = 0.0 * unit.nanometer ** 3                        # polarizability
        mp.setMultipoleParameters(i, *p)
    mp.updateParametersInContext(ctx)
    e_off = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return e_full - e_off


def minimize(topology, positions):
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond), PLAT)
    ctx.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    return ctx.getState(getPositions=True).getPositions()


def main():
    out = {}
    for kind, chains in (("bound", "AB"), ("free", "B")):
        top, pos = prep(chains)
        pos = minimize(top, pos)
        amb = amber_contrib(top, pos)
        try:
            amo = amoeba_contrib(top, pos)
        except Exception as e:
            print(f"[{kind}] AMOEBA FAILED: {type(e).__name__}: {str(e)[:160]}"); return
        out[kind] = (amb, amo)
        print(f"[{kind}] Asp75 charged contribution:  amber14={amb:+.1f}   AMOEBA={amo:+.1f} kcal/mol", flush=True)
    amb_bind = out["bound"][0] - out["free"][0]
    amo_bind = out["bound"][1] - out["free"][1]
    print(f"\nAsp75 BINDING contribution (bound − free):  amber14={amb_bind:+.1f}   AMOEBA={amo_bind:+.1f} kcal/mol")
    print(f"exp ΔΔG(D75N) = +5.90 (removing Asp75 charge weakens binding by 5.90)")
    print("VERDICT: " + ("AMOEBA gives a MUCH larger charged binding contribution than amber14 → polarization IS "
                         "the lever; a polarizable/hybrid FEP is worth building."
                         if abs(amo_bind) > abs(amb_bind) + 2 else
                         "AMOEBA is NOT much larger than amber14 → polarization is NOT the main gap here; the "
                         "problem is elsewhere (setup/sampling/charge-only). Reconsider before a polarizable FEP."))


if __name__ == "__main__":
    main()
