"""E335 — CONVERGED SKEMPI validation: fix E334's under-equilibration (not its precision).

E334 gave D75N +1.07 ± 0.54 vs exp +5.90 — a tight error on an UNCONVERGED estimate. Diagnosis: Asp75–Lys101 is
a real 3.1 Å salt bridge, but a short (~16 ps/window) unrestrained sim doesn't equilibrate the bound interface,
so the salt-bridge desolvation balance is mis-sampled (bound ⟨dU/dm⟩ ≈ free even at full charge). More samples
would only tighten a biased number. The fix is EQUILIBRATION + longer sampling + more windows:
  • NPT equilibration at full charge (MonteCarloBarostat) so the interface/salt bridge settles in explicit water;
  • 11 λ-windows (was 5); ~10× longer production per window;
  • same relative charge-morph difference-of-derivatives estimator (that part was fine).
Target ~1–1.5 h/complex. Honest: if this closes the +1→+6 gap, E334 was under-equilibrated; if it plateaus, the
charge-only morph / fixed-charge FF can't reach this buried-salt-bridge ΔΔG and a full mutation FEP is needed.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e335_skempi_converged.py 2O3B_A_B DB75N 5.90
"""
from __future__ import annotations
import sys, time
import numpy as np
ROOT = "/home/igem/unknown_software"
sys.path.insert(0, ROOT + "/scripts")
from e334_skempi_validation import build          # verified chain-extraction + morph-offset build
from e332_g1_charged_corrected import rocklin_correction


def deriv_curve(system, model, morphs, equil_steps, prod_equil, n_samp, n_stride):
    import openmm as mm
    from openmm import unit
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25))  # NPT
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions); ctx.setParameter("morph", 0.0)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=1000)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin)
    integ.step(equil_steps)                        # NPT equilibration at FULL charge (settle the salt bridge)
    L = ctx.getState().getPeriodicBoxVectors()[0][0].value_in_unit(unit.angstrom)
    U = lambda m: (ctx.setParameter("morph", m),
                   ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))[1]
    d, out = 0.02, []
    for m in morphs:
        ctx.setParameter("morph", m); integ.step(prod_equil)
        der = []
        for _ in range(n_samp):
            integ.step(n_stride)
            lo, hi = max(0., m - d), min(1., m + d)
            der.append((U(hi) - U(lo)) / (hi - lo)); ctx.setParameter("morph", m)
        out.append((float(np.mean(der)), float(np.std(der) / np.sqrt(len(der)))))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1][0]:+8.2f} ± {out[-1][1]:.2f}", flush=True)
    return out, L


def main():
    tag, mut, exp = sys.argv[1], sys.argv[2], float(sys.argv[3])
    print(f"=== E335 CONVERGED SKEMPI: {tag} {mut}  exp={exp:+.2f} kcal (NPT equil + 11 windows + long) ===",
          flush=True)
    t0 = time.time()
    morphs = list(np.round(np.linspace(0, 1, 11), 3))
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    db, Lb = deriv_curve(sysb, modb, morphs, equil_steps=25000, prod_equil=5000, n_samp=150, n_stride=100)
    sysf, modf, af, _ = build(tag, mut, "free")
    df, Lf = deriv_curve(sysf, modf, morphs, equil_steps=25000, prod_equil=5000, n_samp=200, n_stride=100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    be = np.array([v[1] for v in db]); fe = np.array([v[1] for v in df])
    diff = bnd - fre
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ddg = float(_trap(diff, morphs))
    w = np.gradient(np.array(morphs)); err = float(np.sqrt(np.sum((w * np.sqrt(be**2 + fe**2))**2)))
    corr = rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf)
    print(f"\nΔΔG_bind CALC = {ddg + corr:+.2f} ± {err:.2f} kcal   (raw {ddg:+.2f}, Rocklin {corr:+.2f})")
    print(f"ΔΔG_bind EXP  = {exp:+.2f} kcal")
    print(f"|calc − exp|  = {abs(ddg + corr - exp):.2f} kcal   ({'✓ within 1.5' if abs(ddg+corr-exp)<1.5 else '✗ still off'})")
    print(f"vs E334 (short): +1.07.  wall={(time.time()-t0)/60:.0f} min.")


if __name__ == "__main__":
    main()
