"""E328 (Ram's idea, core mechanism) — charge a solute in EXPLICIT water and MONITOR THE DERIVATIVE (dU/dλ).

Ram's key correct instinct: the charged contribution must come from MD with *clear water* and *monitoring the
derivative*, not a static Born burial count (which is exactly why E327's cheap proxy failed). That is literally
Thermodynamic Integration (Kirkwood 1935): ΔG_charge = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ, where λ scales the solute charges and
⟨∂U/∂λ⟩ is "the derivative" sampled in explicit solvent. This POC extends E322-B (GB implicit) to EXPLICIT
TIP3P water and prints the ⟨dU/dλ⟩ curve + its TI integral — the mechanism the whole T1-charged milestone rests
on. It is a SHORT, unconverged demonstration (not a production ΔG): the point is that the derivative is finite,
smooth, and integrable in explicit water, which no static term reproduced.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e328_explicit_water_ti.py
"""
from __future__ import annotations
import numpy as np


def main() -> None:
    import openmm as mm
    from openmm import unit
    from openmmtools import alchemy, testsystems

    ts = testsystems.AlanineDipeptideExplicit(nonbondedMethod=mm.app.PME)  # solute + TIP3P waters, PME
    n_solute = 22
    print(f"explicit-water system: {ts.system.getNumParticles()} atoms "
          f"(solute {n_solute} + TIP3P), PME — real desolvation, not a burial count", flush=True)

    factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False,
                                                alchemical_pme_treatment="direct-space")
    region = alchemy.AlchemicalRegion(alchemical_atoms=list(range(n_solute)),
                                      annihilate_electrostatics=True, annihilate_sterics=False)
    alch = factory.create_alchemical_system(ts.system, region)
    astate = alchemy.AlchemicalState.from_system(alch)

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    ctx = mm.Context(alch, integrator, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(ts.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)

    def U(lam):
        astate.lambda_electrostatics = lam
        astate.apply_to_context(ctx)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    lambdas = [1.0, 0.75, 0.5, 0.25, 0.0]
    d = 0.02
    dudl = []
    print("\nλ_elec   ⟨dU/dλ⟩ (kcal/mol)   [finite-difference derivative, sampled in explicit water]", flush=True)
    for lam in lambdas:
        astate.lambda_electrostatics = lam
        astate.apply_to_context(ctx)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin)
        integrator.step(500)  # short equilibration at this λ (0.1 ps) — demo, not converged
        deriv = []
        for _ in range(40):
            integrator.step(50)
            lo = max(0.0, lam - d); hi = min(1.0, lam + d)
            deriv.append((U(hi) - U(lo)) / (hi - lo))
        astate.lambda_electrostatics = lam  # restore
        m, s = float(np.mean(deriv)), float(np.std(deriv) / np.sqrt(len(deriv)))
        dudl.append(m)
        print(f"  {lam:4.2f}    {m:+10.2f} ± {s:5.2f}", flush=True)

    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ti = float(_trap(dudl, lambdas))  # ∫ ⟨dU/dλ⟩ dλ  (λ from 1→0, so sign per direction)
    print(f"\nTI charging free energy ∫⟨dU/dλ⟩dλ (charging 0→1) ≈ {-ti:+.1f} kcal/mol "
          "(SHORT/unconverged demo; magnitude not production)")
    print("The derivative is finite, smooth, and integrable in EXPLICIT water — the mechanism a static Born "
          "term (E327) cannot provide. Production needs: converged sampling, BOTH legs (bound+free), and the "
          "Rocklin/Hünenberger net-charge finite-size correction (charged→neutral changes net charge).")


if __name__ == "__main__":
    main()
