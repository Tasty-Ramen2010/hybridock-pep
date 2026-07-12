"""E329 — T1-charged G1 SPIKE + the --ultra Tier-3 test on a hard charged complex.

Ram: kill the (dead) N2 cloud campaign, run the real charged FEP spike, and test whether the expensive
`--ultra` charged-verification tier actually MAKES A DIFFERENCE on a hard charged complex.

This runs the electrostatic-decoupling (decharging) leg in EXPLICIT TIP3P water, BOTH legs of the thermodynamic
cycle, on a complex our fast scorer gets wrong:
  target = 2jqk (peptide DEEIERQLKALGVD, charge-rich; receptor 586 atoms; fast-scorer residual +2.76 kcal).
  ΔΔG_elec(binding) = ΔG_decharge(bound) − ΔG_decharge(free)
    = the free-energy contribution of the peptide's formal charges to binding — exactly the term the static
      scorer cannot see (poly-ALA moved ΔG 0.07 kcal; E327 Born proxy was null). Sign: >0 ⇒ the charges help
      binding (net favorable electrostatic/salt-bridge contribution).

Alchemical atoms = the peptide's charged side chains (Lys/Arg/Asp/Glu); annihilate electrostatics only (sterics
stay — our scorer already handles shape). Standard residues ⇒ amber14 parametrises directly (no antechamber).

HONEST SCOPE: short/unconverged demo of the mechanism + the bound−free difference on a real complex. NOT a
production ΔΔG (that needs long sampling + the Rocklin/Hünenberger net-charge finite-size correction, flagged).

Usage:
  build-only sanity:  /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e329_g1_charged_spike.py --build-only
  full short spike:   /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e329_g1_charged_spike.py
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

ROOT = "/home/igem/unknown_software"
PID = "2jqk"
CHARGED_RES = {"LYS", "ARG", "ASP", "GLU"}
PEP_CHAIN = "X"


def _find(pid, key):
    return next((f for f in glob.glob(f"{ROOT}/datasets/**/*.pdb", recursive=True)
                 if os.path.basename(f).lower().startswith(pid) and key in f), None)


def build_system(kind):
    """kind='bound' (receptor+peptide) or 'free' (peptide only). Returns (system, modeller, alch_atom_idx)."""
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    recf, pepf = _find(PID, "_rec"), _find(PID, "_pep")
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")

    def fixed(path, chain_id):
        fx = PDBFixer(filename=path)
        fx.findMissingResidues(); fx.missingResidues = {}
        fx.findMissingAtoms(); fx.addMissingAtoms()
        fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
        fx.removeHeterogens(keepWater=False)
        for ch in fx.topology.chains():
            ch.id = chain_id
        return fx.topology, fx.positions

    pep_top, pep_pos = fixed(pepf, PEP_CHAIN)
    if kind == "bound":
        rec_top, rec_pos = fixed(recf, "A")
        model = app.Modeller(rec_top, rec_pos)
        model.add(pep_top, pep_pos)
    else:
        model = app.Modeller(pep_top, pep_pos)
    model.addHydrogens(ff)
    model.addSolvent(ff, model="tip3p", padding=1.0 * unit.nanometer, neutralize=True)

    # alchemical atoms = peptide charged side-chain atoms (exclude backbone N,CA,C,O,H on backbone)
    bb = {"N", "CA", "C", "O", "HA", "H", "HN"}
    alch = [a.index for a in model.topology.atoms()
            if a.residue.chain.id == PEP_CHAIN and a.residue.name in CHARGED_RES and a.name not in bb]
    system = ff.createSystem(model.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds,
                             rigidWater=True)
    return system, model, alch


def decharge_dG(kind, n_iter, n_steps):
    import openmm as mm
    from openmm import unit
    from openmmtools import alchemy, states, mcmc, multistate
    from openmmtools.multistate import MultiStateSamplerAnalyzer
    import copy, tempfile

    system, model, alch = build_system(kind)
    print(f"  [{kind}] {system.getNumParticles()} atoms, {len(alch)} alchemical (charged side-chain) atoms",
          flush=True)
    factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False,
                                                alchemical_pme_treatment="direct-space")
    region = alchemy.AlchemicalRegion(alchemical_atoms=alch, annihilate_electrostatics=True,
                                      annihilate_sterics=False)
    alch_sys = factory.create_alchemical_system(system, region)
    astate = alchemy.AlchemicalState.from_system(alch_sys)

    # minimize at full charge first
    ctx = mm.Context(alch_sys, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                           2 * unit.femtosecond),
                     mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=300)
    minpos = ctx.getState(getPositions=True).getPositions()
    del ctx

    lambdas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    ref = states.ThermodynamicState(alch_sys, temperature=300 * unit.kelvin)
    comp = []
    for lam in lambdas:
        cs = states.CompoundThermodynamicState(copy.deepcopy(ref), composable_states=[copy.deepcopy(astate)])
        cs.lambda_electrostatics = lam
        cs.lambda_sterics = 1.0
        comp.append(cs)
    move = mcmc.LangevinDynamicsMove(timestep=2 * unit.femtosecond, collision_rate=1 / unit.picosecond,
                                     n_steps=n_steps)
    sampler = multistate.ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=n_iter)
    rep = multistate.MultiStateReporter(tempfile.mktemp(suffix=".nc"), checkpoint_interval=n_iter)
    sampler.create(thermodynamic_states=comp,
                   sampler_states=states.SamplerState(minpos, box_vectors=alch_sys.getDefaultPeriodicBoxVectors()),
                   storage=rep)
    sampler.run()
    az = MultiStateSamplerAnalyzer(sampler._reporter)
    Df, dDf = az.get_free_energy()
    kT = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(unit.kilocalories_per_mole)
    return Df[0, -1] * kT, dDf[0, -1] * kT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    if args.build_only:
        for kind in ("free", "bound"):
            t = time.time()
            system, model, alch = build_system(kind)
            print(f"[{kind}] built OK: {system.getNumParticles()} atoms, {len(alch)} alchemical atoms "
                  f"({time.time()-t:.0f}s)", flush=True)
        print("BUILD OK — parametrisation + solvation + alchemical selection all succeed.")
        return

    print(f"=== E329 G1 charged spike on {PID} (iters={args.iters}, steps={args.steps}) ===", flush=True)
    t0 = time.time()
    dg_bound, e_bound = decharge_dG("bound", args.iters, args.steps)
    print(f"ΔG_decharge(bound) = {dg_bound:+.2f} ± {e_bound:.2f} kcal/mol", flush=True)
    dg_free, e_free = decharge_dG("free", args.iters, args.steps)
    print(f"ΔG_decharge(free)  = {dg_free:+.2f} ± {e_free:.2f} kcal/mol", flush=True)
    ddg = dg_bound - dg_free
    err = (e_bound ** 2 + e_free ** 2) ** 0.5
    print(f"\nΔΔG_elec(binding) = decharge(bound) − decharge(free) = {ddg:+.2f} ± {err:.2f} kcal/mol")
    print(f"  (>0 ⇒ the peptide's charges contribute FAVORABLY to binding — the term the static scorer misses)")
    print(f"  fast-scorer residual on {PID} was +2.76 kcal (scorer wrong); does this move it right?")
    print(f"  wall={time.time()-t0:.0f}s. SHORT/unconverged demo; production needs longer sampling + "
          "Rocklin/Hünenberger net-charge finite-size correction.")


if __name__ == "__main__":
    main()
