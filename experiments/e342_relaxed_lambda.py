"""E342 — does PER-λ RELAXATION close the frozen-pose overshoot? (tests hypothesis 6 from the decomposition)

E341 (frozen-ish morph) systematically OVERSHOOTS on clean cases: 2PCB DA34N calc +9.50 vs exp +0.82 (10×),
3HFM DY101N +4.28 vs +1.34 (3×) — and the overshoot is reproducible (both reps agree), so it is a SYSTEMATIC
bias, not sampling noise. Diagnosis: e334.deriv_curve minimizes ONCE at morph=0 then gives each window only ~3 ps
equilibration. When we discharge the side chain in the BOUND complex, the salt-bridge partner + waters don't get
to relax inward, so ⟨∂U/∂m⟩_bound stays artificially high → the integral overshoots.

This variant does what the reorganization hypothesis demands: at EACH morph window, set the charge, then
LocalEnergyMinimizer.minimize (let the environment find the new basin) + a LONG equilibration (default 20 ps)
before sampling ⟨∂U/∂m⟩. If per-λ relaxation is the missing physics, the bound curve drops at high morph and
2PCB/3HFM fall toward experiment. If the overshoot survives full relaxation, it is NOT reorganization — it's the
charge-only morph incompleteness (hypothesis 5), and the honest verdict stands.

Same 3 clean cases that parsed in E341; 2 replicates; prints relaxed calc next to the frozen E341 number.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e342_relaxed_lambda.py
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build
from e332_g1_charged_corrected import rocklin_correction
import openmm as mm
from openmm import unit

MORPHS = [0.0, 0.25, 0.5, 0.75, 1.0]
# (tag, mut, exp, frozen-E341 calc) — the frozen number is the thing we're trying to beat toward exp.
CASES = [("1IAR_A_B", "EA9Q", 3.11, -4.37),
         ("3HFM_HL_Y", "DY101N", 1.34, +4.28),
         ("2PCB_A_B", "DA34N", 0.82, +9.50)]


def deriv_curve_relaxed(system, model, morphs, n_equil, n_samp, n_stride, min_iter=500):
    """TI with PROPER per-λ relaxation: minimize + long-equilibrate the structure at each morph before sampling."""
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions); ctx.setParameter("morph", 0.0)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    L = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(unit.angstrom)

    def U(m):
        ctx.setParameter("morph", m)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    d, out = 0.02, []
    for m in morphs:
        ctx.setParameter("morph", m)
        # KEY DIFFERENCE vs E341: relax the structure INTO the new charge state at this window.
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=min_iter)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin)
        integ.step(n_equil)                        # long equilibration (default 20 ps) — reorganization happens here
        der = []
        for _ in range(n_samp):
            integ.step(n_stride)
            lo, hi = max(0., m - d), min(1., m + d)
            der.append((U(hi) - U(lo)) / (hi - lo)); ctx.setParameter("morph", m)
        out.append((float(np.mean(der)), float(np.std(der) / np.sqrt(len(der)))))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1][0]:+8.2f} ± {out[-1][1]:.2f}", flush=True)
    return out, L


def one(tag, mut):
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    db, Lb = deriv_curve_relaxed(sysb, modb, MORPHS, 10000, 80, 100)   # 20 ps equil/window
    sysf, modf, af, _ = build(tag, mut, "free")
    df, Lf = deriv_curve_relaxed(sysf, modf, MORPHS, 10000, 120, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(bnd - fre, MORPHS)) + (rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf))


def main():
    print("=== E342 per-λ RELAXED TI (minimize + 20 ps equil per window) vs E341 frozen ===", flush=True)
    res = []
    for tag, mut, exp, frozen in CASES:
        vals = []
        for rep in range(2):
            t = time.time()
            try:
                vals.append(one(tag, mut))
                print(f"  {tag} {mut} rep{rep}: calc={vals[-1]:+.2f}  ({(time.time()-t)/60:.0f}min)", flush=True)
            except Exception as e:
                print(f"  {tag} {mut} rep{rep}: FAIL {str(e)[:90]}", flush=True)
        if vals:
            calc = float(np.mean(vals)); sp = float(np.std(vals)) if len(vals) > 1 else 0.0
            res.append((tag, mut, exp, frozen, calc, sp))
            print(f"  => {tag} {mut}: RELAXED={calc:+.2f}±{sp:.2f}  frozen={frozen:+.2f}  exp={exp:+.2f}  "
                  f"|Δ|={abs(calc-exp):.2f}", flush=True)
    print("\n=== SUMMARY: did relaxation close the overshoot? ===")
    print(f"{'case':16s} {'relaxed':>12s} {'frozen':>8s} {'exp':>6s} {'|err|':>6s}")
    for tag, mut, exp, frozen, calc, sp in res:
        print(f"{tag+' '+mut:16s} {calc:+.2f}±{sp:.2f}  {frozen:+.2f}  {exp:+.2f}  {abs(calc-exp):.2f}")
    if len(res) >= 3:
        c = np.array([r[4] for r in res]); e = np.array([r[2] for r in res]); fz = np.array([r[3] for r in res])
        mae_r, mae_f = np.mean(np.abs(c - e)), np.mean(np.abs(fz - e))
        print(f"\nMAE relaxed={mae_r:.2f}  vs  frozen={mae_f:.2f}   (Δ={mae_f-mae_r:+.2f} kcal)")
        print("VERDICT: " + ("per-λ RELAXATION closes the overshoot → the gap WAS conformational reorganization "
                             "(hyp 6); a relaxed-TI charged tier is worth building."
                             if mae_r < mae_f - 1.5 else
                             "overshoot SURVIVES full relaxation → it is NOT reorganization; it's the charge-only "
                             "morph incompleteness (hyp 5). Honest verdict stands: fast scorer + N5 flag ships."))


if __name__ == "__main__":
    main()
