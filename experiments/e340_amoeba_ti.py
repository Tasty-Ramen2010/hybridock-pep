"""E340 — turn the E339 diagnostic into a real free energy: AMOEBA charge-morph TI ΔΔG on 2O3B D75N.

E339 (⟨ΔE⟩ diagnostic) said AMOEBA recovers Asp75's salt-bridge binding contribution (−7.2 ~ exp −5.9) where
amber gives ~0. This runs the actual polarizable free energy: the SAME charge-morph TI as our amber FEP (e334),
but in AMOEBA (amoeba2018 + generalized-Kirkwood implicit solvent). Morph Asp75 monopoles charged→neutral, sample
AMOEBA MD, ⟨∂U/∂morph⟩ by finite difference, integrate the bound−free difference.
  Expect: if polarization is the fix, ΔΔG_AMOEBA ≈ +5.9 (exp) vs amber's +1.5.

AMOEBA MD is ~100× slower than fixed-charge, so this uses implicit solvent (protein only) + short sampling; a
`--speed-test` mode times a few steps first. HONEST: short/unconverged; the point is whether AMOEBA moves the
number toward +5.9.

Run:  python experiments/e340_amoeba_ti.py --speed-test
      python experiments/e340_amoeba_ti.py
"""
from __future__ import annotations
import sys, time, argparse
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e338_amoeba_vs_amber_saltbridge import prep, asp_atoms
import openmm as mm
from openmm import app, unit
PLAT = mm.Platform.getPlatformByName("CUDA")


def build(chains):
    top, pos = prep(chains)
    # MUTUAL polarization is essential: the buried-ion-pair stabilisation is the self-consistent induced-dipole
    # response (JACS 2022). 'direct' (one-shot) misses it — E340-direct gave +1.04, an artifact.
    system = app.ForceField("amoeba2018.xml", "amoeba2018_gk.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, polarization="mutual", mutualInducedTargetEpsilon=1e-5)
    alch = asp_atoms(top)
    mp = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "AmoebaMultipoleForce")
    gk = next((system.getForce(i) for i in range(system.getNumForces())
               if system.getForce(i).__class__.__name__ == "AmoebaGeneralizedKirkwoodForce"), None)
    q0 = {i: mp.getMultipoleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in alch}
    shift = sum(q0.values()) / len(q0)
    qn = {i: q0[i] - shift for i in alch}   # neutralise net → 0
    return top, pos, system, mp, gk, alch, q0, qn


def set_morph(mp, gk, alch, q0, qn, m):
    for i in alch:
        q = q0[i] + m * (qn[i] - q0[i])
        p = list(mp.getMultipoleParameters(i)); p[0] = q * unit.elementary_charge
        mp.setMultipoleParameters(i, *p)
        if gk is not None:
            g = list(gk.getParticleParameters(i)); g[0] = q * unit.elementary_charge
            gk.setParticleParameters(i, *g)


def deriv_curve(chains, morphs, n_equil, n_samp, n_stride):
    top, pos, system, mp, gk, alch, q0, qn = build(chains)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    ctx = mm.Context(system, integ, PLAT); ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    print(f"  [{chains}] {system.getNumParticles()} atoms, morph Asp75 ({len(alch)} atoms)", flush=True)

    def U(m):
        set_morph(mp, gk, alch, q0, qn, m); mp.updateParametersInContext(ctx)
        if gk is not None:
            gk.updateParametersInContext(ctx)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    d, out = 0.02, []
    for m in morphs:
        set_morph(mp, gk, alch, q0, qn, m); mp.updateParametersInContext(ctx)
        if gk is not None:
            gk.updateParametersInContext(ctx)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(n_equil)
        der = []
        for _ in range(n_samp):
            integ.step(n_stride)
            der.append((U(min(1., m + d)) - U(max(0., m - d))) / (min(1., m + d) - max(0., m - d)))
        out.append((float(np.mean(der)), float(np.std(der) / np.sqrt(len(der)))))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1][0]:+8.2f} ± {out[-1][1]:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--speed-test", action="store_true"); a = ap.parse_args()
    if a.speed_test:
        top, pos, system, mp, gk, alch, q0, qn = build("B")
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
        ctx = mm.Context(system, integ, PLAT); ctx.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)
        t = time.time(); integ.step(1000)
        rate = 1000 / (time.time() - t)
        print(f"AMOEBA implicit ({system.getNumParticles()} atoms): {rate:.0f} steps/s "
              f"→ {rate*86400*1e-6:.2f} ns/day (1fs).  free-leg only; bound is ~3× bigger.", flush=True)
        return
    print("=== E340 AMOEBA charge-morph TI on 2O3B D75N ===", flush=True)
    morphs = [0.0, 0.25, 0.5, 0.75, 1.0]
    t0 = time.time()
    db = deriv_curve("AB", morphs, 500, 30, 100)
    df = deriv_curve("B", morphs, 500, 40, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    be = np.array([v[1] for v in db]); fe = np.array([v[1] for v in df])
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ddg = float(_trap(bnd - fre, morphs))
    w = np.gradient(np.array(morphs)); err = float(np.sqrt(np.sum((w * np.sqrt(be**2 + fe**2))**2)))
    print(f"\nΔΔG_AMOEBA(D75N) = {ddg:+.2f} ± {err:.2f} kcal    (amber FEP +1.5, exp +5.90)")
    print(f"wall={(time.time()-t0)/60:.0f} min.  " + ("AMOEBA MOVES it toward exp → polarizable FEP validated"
          if ddg > 3.0 else "AMOEBA short-TI did not reach exp — needs convergence or the reweighting route"))


if __name__ == "__main__":
    main()
