"""E332 — CORRECTED T1-charged FEP spike (fixes the three failures E329 surfaced).

E329 (naive) gave ΔΔG_elec = −12.4 ± 39.2 kcal — noise-dominated. Diagnosis: (1) it ANNIHILATED electrostatics
(removes the peptide's intramolecular self-energy too → two ~+330 kcal legs whose noise swamps the ~−12 signal);
(2) no net-charge finite-size correction; (3) too little sampling, free leg worst. Corrections here:

  1. DECOUPLE, not annihilate (annihilate_electrostatics=False): keep the peptide's intramolecular charges ON,
     perturb only peptide↔environment. This deletes the huge, noisy, identically-cancelling self-energy term
     from BOTH legs — the dominant fix for the catastrophic cancellation.
  2. Rocklin/Hünenberger leading net-charge finite-size correction (Wigner self-energy term), per leg (bound and
     free boxes differ in size → it contributes to ΔΔG).
  3. More sampling, 8 λ-windows, and the FREE leg runs longest (it was the ±37 bottleneck).

Same complex as E329 (2jqk) for a head-to-head. HONEST: still a spike — better-converged, not necessarily
production. GPU-hours; run in background.

Run:  nohup /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e332_g1_charged_corrected.py \
        > logs/e332_corrected.log 2>&1 &
"""
from __future__ import annotations
import sys, time
ROOT = "/home/igem/unknown_software"
sys.path.insert(0, ROOT + "/scripts")
from e329_g1_charged_spike import build_system, PID  # reuse the proven PDBFixer+amber14+solvate builder

XI_EW = -2.837297           # cubic-lattice Wigner constant
COULOMB = 332.0637          # e^2/(4πε0) in kcal·Å/mol
EPS_S = 78.4                # water dielectric


def net_charge_of_alch(system, alch):
    """Net charge (e) carried by the alchemical (peptide charged side-chain) atoms."""
    nb = next(f for f in (system.getForce(i) for i in range(system.getNumForces()))
              if f.__class__.__name__ == "NonbondedForce")
    from openmm import unit
    return sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in alch)


def rocklin_correction(Q, L_angstrom):
    """Leading net-charge finite-size (Wigner self-energy) correction, kcal/mol.
    ΔG ≈ -ξ_EW · Q² · (e²/8πε0) / (ε_s · L).  Dominant Rocklin term; sign per Hünenberger convention."""
    return -XI_EW * (Q ** 2) * (COULOMB / 2.0) / (EPS_S * L_angstrom)


def decouple_dG(kind, n_iter, n_steps):
    import openmm as mm
    from openmm import unit
    from openmmtools import alchemy, states, mcmc, multistate
    from openmmtools.multistate import MultiStateSamplerAnalyzer
    import copy, tempfile

    system, model, alch = build_system(kind)
    Q = net_charge_of_alch(system, alch)
    box = system.getDefaultPeriodicBoxVectors()
    L = box[0][0].value_in_unit(unit.angstrom)
    corr = rocklin_correction(Q, L)
    print(f"  [{kind}] {system.getNumParticles()} atoms, {len(alch)} alch, net Q={Q:+.1f} e, "
          f"L={L:.1f} Å, Rocklin corr={corr:+.2f} kcal", flush=True)

    factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False,
                                                alchemical_pme_treatment="direct-space")
    region = alchemy.AlchemicalRegion(alchemical_atoms=alch, annihilate_electrostatics=False,  # DECOUPLE
                                      annihilate_sterics=False)
    alch_sys = factory.create_alchemical_system(system, region)
    astate = alchemy.AlchemicalState.from_system(alch_sys)

    ctx = mm.Context(alch_sys, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                           2 * unit.femtosecond),
                     mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    minpos = ctx.getState(getPositions=True).getPositions()
    del ctx

    lambdas = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0]  # 8 windows
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
    raw = Df[0, -1] * kT
    return raw, dDf[0, -1] * kT, corr


def main():
    print(f"=== E332 CORRECTED charged FEP on {PID} (decouple + Rocklin + more sampling) ===", flush=True)
    t0 = time.time()
    # free leg is the convergence bottleneck → give it the most iterations
    fb_raw, fb_err, fb_corr = decouple_dG("bound", n_iter=400, n_steps=500)
    print(f"ΔG_decouple(bound) = {fb_raw:+.2f} ± {fb_err:.2f} (+ Rocklin {fb_corr:+.2f}) kcal", flush=True)
    ff_raw, ff_err, ff_corr = decouple_dG("free", n_iter=800, n_steps=500)
    print(f"ΔG_decouple(free)  = {ff_raw:+.2f} ± {ff_err:.2f} (+ Rocklin {ff_corr:+.2f}) kcal", flush=True)

    raw_ddg = fb_raw - ff_raw
    corr_ddg = (fb_raw + fb_corr) - (ff_raw + ff_corr)
    err = (fb_err ** 2 + ff_err ** 2) ** 0.5
    print(f"\nΔΔG_elec RAW        = {raw_ddg:+.2f} ± {err:.2f} kcal")
    print(f"ΔΔG_elec CORRECTED  = {corr_ddg:+.2f} ± {err:.2f} kcal   (Rocklin net-charge applied)")
    print(f"  (>0 ⇒ peptide charges favor binding — the term the static scorer misses)")
    print(f"  compare E329 naive (annihilate): −12.4 ± 39.2 kcal.  Error now {err:.1f} vs 39.2 "
          f"(decouple removes the self-energy noise).")
    print(f"  fast-scorer residual on {PID} = +2.76 kcal. wall={ (time.time()-t0)/60:.0f} min.")
    print("  HONEST: still a spike; a production ΔΔG needs the full Rocklin scheme (all terms) + convergence "
          "checks (overlap, dHdλ) + ideally charge-balanced co-alchemical morphing.")


if __name__ == "__main__":
    main()
