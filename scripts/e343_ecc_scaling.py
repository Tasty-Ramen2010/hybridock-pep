"""E343 — ECC test: does charge scaling (q → 0.75q) collapse the frozen-pose overshoot?

The literature diagnosis (Ram's "abuse the physics" instinct): fixed-charge amber14 OVER-stabilizes salt bridges
because it has no electronic screening — the standard fix is Electronic Continuum Correction, scaling every charge
by 1/√ε_el = 1/√1.78 ≈ 0.75 (JCP 153:050901; insulin salt-bridge overbinding resolved this way, PMC12302216).
Our E341 overshoot (2PCB +9.5 vs exp +0.82, 3HFM +4.3 vs +1.3) is exactly fixed-charge salt-bridge
overstabilization. This re-runs the SAME frozen charge-morph TI as E341, but with all NonbondedForce charges (and
the morph offsets, and the Rocklin ΔQ) scaled ×0.75.
  If the overshoot collapses toward exp → our error WAS fixed-charge overstabilization; ECC is the cheap fix.
  If it barely moves → overstabilization isn't it, and the honest verdict holds.

Same 3 clean cases as E341; 2 replicates; prints ECC calc next to the unscaled-frozen E341 number.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e343_ecc_scaling.py
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build, deriv_curve
from e332_g1_charged_corrected import rocklin_correction
from openmm import unit

SCALE = 0.75                      # ECC: 1/sqrt(1.78) ≈ 0.75
MORPHS = [0.0, 0.25, 0.5, 0.75, 1.0]
CASES = [("1IAR_A_B", "EA9Q", 3.11, -4.37),
         ("3HFM_HL_Y", "DY101N", 1.34, +4.28),
         ("2PCB_A_B", "DA34N", 0.82, +9.50)]


def apply_ecc(system, scale):
    """Scale every particle charge and every morph parameter-offset in the NonbondedForce by `scale` (ECC)."""
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q * scale, sig, eps)
    for k in range(nb.getNumParticleParameterOffsets()):
        p, idx, dq, dsig, deps = nb.getParticleParameterOffset(k)
        nb.setParticleParameterOffset(k, p, idx, dq * scale, dsig, deps)
    # exceptions (1-4 scaled pairs) carry their own chargeProd — scale by scale^2 (product of two charges)
    for k in range(nb.getNumExceptions()):
        a, b, qq, sig, eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, a, b, qq * scale * scale, sig, eps)


def one(tag, mut):
    sysb, modb, ab, dQ = build(tag, mut, "bound"); apply_ecc(sysb, SCALE)
    db, Lb = deriv_curve(sysb, modb, MORPHS, 1500, 80, 100)
    sysf, modf, af, _ = build(tag, mut, "free"); apply_ecc(sysf, SCALE)
    df, Lf = deriv_curve(sysf, modf, MORPHS, 1500, 120, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    _trap = getattr(np, "trapezoid", None) or np.trapz
    dQs = dQ * SCALE                                   # scaled net-charge change for Rocklin
    return float(_trap(bnd - fre, MORPHS)) + (rocklin_correction(dQs, Lb) - rocklin_correction(dQs, Lf))


def main():
    print(f"=== E343 ECC charge-scaling (×{SCALE}) vs E341 unscaled-frozen ===", flush=True)
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
            print(f"  => {tag} {mut}: ECC={calc:+.2f}±{sp:.2f}  unscaled={frozen:+.2f}  exp={exp:+.2f}  "
                  f"|Δ|={abs(calc-exp):.2f}", flush=True)
    print("\n=== SUMMARY: did ECC scaling fix the overstabilization? ===")
    print(f"{'case':16s} {'ECC':>12s} {'unscaled':>9s} {'exp':>6s} {'|err|':>6s}")
    for tag, mut, exp, frozen, calc, sp in res:
        print(f"{tag+' '+mut:16s} {calc:+.2f}±{sp:.2f}  {frozen:+.2f}  {exp:+.2f}  {abs(calc-exp):.2f}")
    if len(res) >= 3:
        c = np.array([r[4] for r in res]); e = np.array([r[2] for r in res]); fz = np.array([r[3] for r in res])
        mae_e, mae_u = np.mean(np.abs(c - e)), np.mean(np.abs(fz - e))
        print(f"\nMAE ECC={mae_e:.2f}  vs  unscaled={mae_u:.2f}   (Δ={mae_u-mae_e:+.2f} kcal)")
        print("VERDICT: " + ("ECC scaling COLLAPSES the overshoot → the error WAS fixed-charge salt-bridge "
                             "overstabilization; scale charges 0.75× and bolt the charged term onto the neutral "
                             "scorer (Ram's decomposition)."
                             if mae_e < mae_u - 1.5 else
                             "ECC barely moves it → overstabilization is NOT the main error. The residual is the "
                             "desolvation self-energy subtraction; try the continuum-PB route for the charged term."))


if __name__ == "__main__":
    main()
