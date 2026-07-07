"""E339 — decisive (sampled) test: does AMOEBA give Asp75 a bigger THERMALLY-AVERAGED charged binding
contribution than amber14? (E338 single-point was artifact-dominated; this averages over an ensemble.)

Reference-potential idea (cheap sampling + expensive endpoint eval): sample the CHARGED bound and free endpoints
with fast amber14+GBSA MD; on each snapshot compute Asp75's electrostatic interaction with its environment
  ⟨ΔE⟩ = ⟨E_full − E(Asp75 electrostatics zeroed)⟩
in BOTH amber14 (fixed charge) AND AMOEBA (polarizable, incl. induced dipoles), on the SAME coordinates. Then
  Asp75 binding contribution(FF) = ⟨ΔE⟩_bound − ⟨ΔE⟩_free.
If AMOEBA ≫ amber14, the missing electronic polarization of the buried Asp75-Lys101 salt bridge IS our gap →
a polarizable/hybrid FEP is the right build. If similar, polarization is NOT the main problem → save the weeks.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e339_amoeba_reweight_sampled.py
"""
from __future__ import annotations
import sys, tempfile
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch, ChainSel
from e338_amoeba_vs_amber_saltbridge import prep, asp_atoms, SIDE
import openmm as mm
from openmm import app, unit

PLAT = mm.Platform.getPlatformByName("CUDA")


def build_forces(topology):
    amber = app.ForceField("amber14-all.xml", "implicit/gbn2.xml").createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    amoeba = app.ForceField("amoeba2018.xml", "amoeba2018_gk.xml").createSystem(
        topology, nonbondedMethod=app.NoCutoff, polarization="mutual", mutualInducedTargetEpsilon=1e-4)
    return amber, amoeba


def amber_dE(system, ctx, alch):
    """ΔE with Asp75 charge zeroed CONSISTENTLY in Coulomb (NonbondedForce) AND GB solvation (CustomGBForce)."""
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    gb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "CustomGBForce")
    e_full = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    snb, sgb = {}, {}
    for i in alch:
        q, s, e = nb.getParticleParameters(i); snb[i] = (q, s, e)
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, s, e)
        gp = gb.getParticleParameters(i); sgb[i] = list(gp)
        gp = list(gp); gp[0] = 0.0; gb.setParticleParameters(i, gp)
    nb.updateParametersInContext(ctx); gb.updateParametersInContext(ctx)
    e_off = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    for i in alch:
        nb.setParticleParameters(i, *snb[i]); gb.setParticleParameters(i, sgb[i])
    nb.updateParametersInContext(ctx); gb.updateParametersInContext(ctx)
    return e_full - e_off


def amoeba_dE(system, ctx, alch):
    """ΔE with Asp75 zeroed CONSISTENTLY in multipoles+polarizability AND GK solvation charge."""
    mp = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "AmoebaMultipoleForce")
    gk = next((system.getForce(i) for i in range(system.getNumForces())
               if system.getForce(i).__class__.__name__ == "AmoebaGeneralizedKirkwoodForce"), None)
    e_full = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    smp, sgk = {}, {}
    for i in alch:
        p = list(mp.getMultipoleParameters(i)); smp[i] = list(p)
        p[0] = 0.0 * unit.elementary_charge
        p[1] = [0, 0, 0] * unit.elementary_charge * unit.nanometer
        p[2] = [0] * 9 * unit.elementary_charge * unit.nanometer ** 2
        p[-1] = 0.0 * unit.nanometer ** 3
        mp.setMultipoleParameters(i, *p)
        if gk is not None:
            g = list(gk.getParticleParameters(i)); sgk[i] = list(g)
            g[0] = 0.0 * unit.elementary_charge; gk.setParticleParameters(i, *g)
    mp.updateParametersInContext(ctx)
    if gk is not None:
        gk.updateParametersInContext(ctx)
    e_off = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    for i in alch:
        mp.setMultipoleParameters(i, *smp[i])
        if gk is not None:
            gk.setParticleParameters(i, *sgk[i])
    mp.updateParametersInContext(ctx)
    if gk is not None:
        gk.updateParametersInContext(ctx)
    return e_full - e_off


def endpoint(chains, n_snap=40):
    top, pos = prep(chains)
    alch = asp_atoms(top)
    amber, amoeba = build_forces(top)
    # sample with amber
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx_a = mm.Context(amber, integ, PLAT); ctx_a.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx_a, maxIterations=500)
    ctx_a.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(5000)
    # AMOEBA evaluation context (reused, positions set per snapshot)
    ctx_m = mm.Context(amoeba, mm.VerletIntegrator(1 * unit.femtosecond), PLAT)
    dE_a, dE_m = [], []
    for k in range(n_snap):
        integ.step(500)
        xyz = ctx_a.getState(getPositions=True).getPositions()
        dE_a.append(amber_dE(amber, ctx_a, alch))
        ctx_m.setPositions(xyz)
        dE_m.append(amoeba_dE(amoeba, ctx_m, alch))
    return np.array(dE_a), np.array(dE_m)


def main():
    print("=== E339 sampled AMOEBA-vs-amber Asp75 charged contribution (2O3B D75N) ===", flush=True)
    res = {}
    for kind, ch in (("free", "B"), ("bound", "AB")):
        a, m = endpoint(ch)
        res[kind] = (a, m)
        print(f"[{kind}] ⟨ΔE_Asp75⟩  amber={a.mean():+.1f}±{a.std()/len(a)**.5:.1f}   "
              f"AMOEBA={m.mean():+.1f}±{m.std()/len(m)**.5:.1f} kcal ({len(a)} snaps)", flush=True)
    amb = res["bound"][0].mean() - res["free"][0].mean()
    amo = res["bound"][1].mean() - res["free"][1].mean()
    print(f"\nAsp75 BINDING contribution ⟨bound⟩−⟨free⟩:  amber={amb:+.1f}   AMOEBA={amo:+.1f} kcal")
    print(f"(exp: removing Asp75 charge weakens binding by +5.90 ⇒ its charge contributes ~−5.9 to binding)")
    print("VERDICT: " + ("AMOEBA gives a MUCH stronger Asp75 binding contribution than amber → electronic "
                         "polarization IS the missing physics; build the polarizable/hybrid FEP."
                         if amo < amb - 2 else
                         "AMOEBA is NOT much stronger than amber → polarization is not the dominant gap here; "
                         "reconsider (setup / charge-only morph / sampling) before committing to a polarizable FEP."))


if __name__ == "__main__":
    main()
