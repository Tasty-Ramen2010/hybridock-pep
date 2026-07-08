"""E348 — Layer 3 decisive test: does dynamic water penetration + reorganization flip the buried-charge WRONG SIGN?

The unifying cause of the collective failure on buried-no-salt-bridge cases (1BRS, 1E96): buried ionizable groups
have apparent dielectric 10–20 (not 2–4) because water PENETRATES and the structure REORGANIZES around the charge
— a dynamic effect our static/3-ps-equilibrated FEP misses, so it over-charges the desolvation penalty and gets
the sign wrong (predicts the charge HURTS binding; exp says it helps).

Controlled test on the two buried wrong-sign cases: run the SAME ECC charge-morph FEP twice —
  SHORT equil (1500 steps = 3 ps)  → reproduces the campaign (expected wrong-sign),
  LONG  equil (50000 steps = 100 ps) → lets water re-equilibrate / penetrate and the local structure relax.
If LONG moves the ΔΔG toward exp (flips sign / shrinks error), Axis B is a SAMPLING problem HybriCharge Layer 3
can fix (thin explicit-water shell + short restrained MD). If it doesn't move, water penetration into the buried
cavity is slower than 100 ps and we need explicit cavity hydration.

Runs alongside the E345 campaign (small systems). Report SHORT vs LONG vs exp per case.

Run: OMP_NUM_THREADS=1 /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e348_reorganization_test.py
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build, deriv_curve
from e343_ecc_scaling import apply_ecc
from e332_g1_charged_corrected import rocklin_correction

MORPHS = [0.0, 0.25, 0.5, 0.75, 1.0]
_trap = getattr(np, "trapezoid", None) or np.trapz
# buried, wrong-sign in the campaign: 1BRS ecc −8.2 (exp +1.45), 1E96 ecc −0.8 (exp +2.16)
CASES = [("1BRS_A_D", "EA73Q", 1.45), ("1E96_A_B", "DA38N", 2.16)]


def run(tag, mut, n_equil):
    sysb, modb, ab, dQ = build(tag, mut, "bound"); apply_ecc(sysb, 0.75)
    db, Lb = deriv_curve(sysb, modb, MORPHS, n_equil, 60, 100)
    sysf, modf, af, _ = build(tag, mut, "free"); apply_ecc(sysf, 0.75)
    df, Lf = deriv_curve(sysf, modf, MORPHS, n_equil, 90, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    dQs = dQ * 0.75
    return float(_trap(bnd - fre, MORPHS)) + (rocklin_correction(dQs, Lb) - rocklin_correction(dQs, Lf))


def main():
    print("=== E348 reorganization/water-penetration test (short 3ps vs long 100ps equil, ECC) ===", flush=True)
    for tag, mut, exp in CASES:
        row = {}
        for label, eq in (("SHORT_3ps", 1500), ("LONG_100ps", 50000)):
            t = time.time()
            try:
                row[label] = run(tag, mut, eq)
                print(f"  {tag} {mut} {label:10s}: ΔΔG={row[label]:+.2f}  exp={exp:+.2f}  "
                      f"({(time.time()-t)/60:.0f}min)", flush=True)
            except Exception as e:
                print(f"  {tag} {mut} {label:10s}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        if "SHORT_3ps" in row and "LONG_100ps" in row:
            s, l = row["SHORT_3ps"], row["LONG_100ps"]
            moved = l - s
            better = abs(l - exp) < abs(s - exp)
            flip = (s < 0) and (l > 0)
            print(f"  => {tag}: short={s:+.2f} → long={l:+.2f} (Δ={moved:+.2f})  "
                  f"{'SIGN FLIP toward exp ✓' if flip else ('closer to exp ✓' if better else 'no improvement')}",
                  flush=True)
    print("\nVERDICT: if long-equil moved the buried cases toward exp, Axis B is a sampling problem → build "
          "HybriCharge Layer 3 (explicit water shell + restrained MD). If not, need explicit cavity hydration.")


if __name__ == "__main__":
    main()
