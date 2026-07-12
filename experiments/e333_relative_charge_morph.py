"""E333 — RELATIVE charge-morph FEP by DIFFERENCE OF DERIVATIVES (Ram's "derivative it" idea).

Progression: E329 annihilate (two ~+330 kcal legs → ±39, useless) → E332 decouple (keep intramolecular →
smaller) → E333 (this): don't subtract two big absolute numbers at all. Instead:
  • single-topology CHARGE MORPH — same atoms, partial charges interpolated real→neutralised via an OpenMM
    NonbondedForce ParticleParameterOffset on a global `morph` param (no atom mapping, no softcore). This is the
    charged contribution specifically (the term the static scorer misses), as a SMALL perturbation (remove net
    charge, keep dipoles) — not a full annihilation.
  • compute ⟨∂U/∂morph⟩ in BOTH legs and integrate the DIFFERENCE of the derivative curves:
        ΔΔG = ∫₀¹ [ ⟨∂U/∂morph⟩_bound − ⟨∂U/∂morph⟩_free ] d(morph)
    The peptide↔solvent part of ∂U/∂morph is nearly common to both legs and cancels POINTWISE at each morph,
    so we integrate a small difference — never forming two huge numbers (Ram's point). TI (Kirkwood), the same
    "monitor the derivative" quantity as E328, now on a relative coordinate.
  • Rocklin/Hünenberger leading net-charge finite-size correction (neutralisation changes net charge).

Same 2jqk target for a clean head-to-head vs E329/E332. HONEST: charge-only morph = the ELECTROSTATIC part of a
Lys→Gln-type mutation (not the vdW/atom change); that is exactly the charged term we want, and it is what a full
pmx/perses single-topology mutation would add the shape part to. Still a spike (short sampling).

Run: nohup /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e333_relative_charge_morph.py \
       > logs/e333_relative.log 2>&1 &
"""
from __future__ import annotations
import sys, time
from collections import defaultdict
import numpy as np

ROOT = "/home/igem/unknown_software"
sys.path.insert(0, ROOT + "/scripts")
from e329_g1_charged_spike import build_system, PID
from e332_g1_charged_corrected import rocklin_correction


def build_morph(kind):
    """Return (system, model, alch, ΔQ_net) with a global `morph` param interpolating charged→neutralised."""
    import openmm as mm
    from openmm import unit
    system, model, alch = build_system(kind)
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    atoms = list(model.topology.atoms())
    by_res = defaultdict(list)
    for i in alch:
        a = atoms[i]
        by_res[(a.residue.chain.id, a.residue.id)].append(i)
    q0 = {i: nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in alch}
    # neutralise each charged residue's side chain: subtract its mean excess so side-chain net → 0
    qn = {}
    for idxs in by_res.values():
        shift = sum(q0[i] for i in idxs) / len(idxs)
        for i in idxs:
            qn[i] = q0[i] - shift
    nb.addGlobalParameter("morph", 0.0)
    for i in alch:
        nb.addParticleParameterOffset("morph", i, qn[i] - q0[i], 0.0, 0.0)  # q(morph)=q0 + morph·(qn−q0)
    dQnet = sum(qn[i] - q0[i] for i in alch)  # net charge change over the whole peptide (for Rocklin)
    return system, model, alch, dQnet


def deriv_curve(kind, morphs, n_equil, n_samp, n_stride):
    """⟨∂U/∂morph⟩ at each morph value, sampled in explicit water on CUDA (finite-difference derivative)."""
    import openmm as mm
    from openmm import unit
    system, model, alch, dQnet = build_morph(kind)
    box = system.getDefaultPeriodicBoxVectors()
    L = box[0][0].value_in_unit(unit.angstrom)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions)
    ctx.setParameter("morph", 0.0)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    print(f"  [{kind}] {system.getNumParticles()} atoms, ΔQnet={dQnet:+.2f} e, L={L:.1f} Å", flush=True)

    def U(m):
        ctx.setParameter("morph", m)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    d = 0.02
    out = []
    for m in morphs:
        ctx.setParameter("morph", m)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin)
        integ.step(n_equil)
        deriv = []
        for _ in range(n_samp):
            integ.step(n_stride)
            lo, hi = max(0.0, m - d), min(1.0, m + d)
            deriv.append((U(hi) - U(lo)) / (hi - lo))
            ctx.setParameter("morph", m)
        out.append((float(np.mean(deriv)), float(np.std(deriv) / np.sqrt(len(deriv)))))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1][0]:+8.2f} ± {out[-1][1]:5.2f}", flush=True)
    return morphs, out, L, dQnet


def main():
    print(f"=== E333 relative charge-morph FEP on {PID} (difference of derivatives) ===", flush=True)
    t0 = time.time()
    morphs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mb, db, Lb, dQ = deriv_curve("bound", morphs, n_equil=1000, n_samp=60, n_stride=100)
    mf, df, Lf, _ = deriv_curve("free", morphs, n_equil=1000, n_samp=120, n_stride=100)

    dudm_b = np.array([v[0] for v in db]); eb = np.array([v[1] for v in db])
    dudm_f = np.array([v[0] for v in df]); ef = np.array([v[1] for v in df])
    diff = dudm_b - dudm_f
    diff_err = np.sqrt(eb ** 2 + ef ** 2)
    _trap = getattr(np, "trapezoid", None) or np.trapz  # np>=2.0 renamed trapz→trapezoid
    ddg_raw = float(_trap(diff, morphs))
    # error via trapezoid weights on the per-point errors
    w = np.gradient(np.array(morphs))
    ddg_err = float(np.sqrt(np.sum((w * diff_err) ** 2)))
    corr = rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf)
    ddg_corr = ddg_raw + corr

    print("\n  morph   <dU/dm>_bound   <dU/dm>_free   difference (integrand)")
    for i, m in enumerate(morphs):
        print(f"   {m:.2f}    {dudm_b[i]:+8.2f}      {dudm_f[i]:+8.2f}      {diff[i]:+8.2f}")
    print(f"\nΔΔG_elec (∫ of the DIFFERENCE of derivatives) = {ddg_raw:+.2f} ± {ddg_err:.2f} kcal")
    print(f"ΔΔG_elec + Rocklin net-charge corr           = {ddg_corr:+.2f} ± {ddg_err:.2f} kcal")
    print(f"\n  head-to-head:  E329 annihilate −12.4 ± 39.2 | E332 decouple (see log) | E333 relative {ddg_corr:+.1f} ± {ddg_err:.1f}")
    print(f"  fast-scorer residual on {PID} = +2.76 kcal.  wall={(time.time()-t0)/60:.0f} min.")
    print("  HONEST: charge-only morph = the ELECTROSTATIC part of a Lys→Gln mutation (not vdW/atoms); a full "
          "pmx/perses single-topology mutation adds the shape part. Still short sampling; the win is the SMALL, "
          "common-term-cancelling perturbation vs the ±39 of subtracting two huge annihilation legs.")


if __name__ == "__main__":
    main()
