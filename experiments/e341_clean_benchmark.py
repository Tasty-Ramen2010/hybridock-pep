"""E341 — the CORRECTED clean benchmark: are we accurate on charge mutations when the setup is right?

The full decomposition (docs/charged_failure_full_decomposition) showed our "FEP fails on charged interfaces"
was driven by confounders, not physics: a stripped Mg2+ (2O3B), atypically-large surface ΔΔG cherry-picks, and
~2 kcal irreproducibility. This re-tests on CLEAN cases: metal-free interfaces, moderate ΔΔG (0.8-3.1 kcal),
isosteric single charge mutations (D→N/E→Q, so the charge-only morph is a good approximation), explicit TIP3P,
and REPLICATES (2×) to expose/average the irreproducibility.
  If calc≈exp (~within 1.5) on these, the earlier "failure" was case selection + setup, not the method.
  If we still miss, the residual is the full-mutation (vdW/atoms) + conformational-sampling terms.

Cases: 1IAR EA9Q (exp +3.11; was the wrong-sign −6.12 blowup — does clean setup fix it?), 3HFM DY101N (+1.34),
1MAH DA71N (+1.88), 2PCB DA34N (+0.82).

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e341_clean_benchmark.py
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build, deriv_curve
from e332_g1_charged_corrected import rocklin_correction

CASES = [("1IAR_A_B", "EA9Q", 3.11), ("3HFM_HL_Y", "DY101N", 1.34),
         ("1MAH_A_F", "DA71N", 1.88), ("2PCB_A_B", "DA34N", 0.82)]
MORPHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def one(tag, mut):
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    db, Lb = deriv_curve(sysb, modb, MORPHS, 1500, 80, 100)
    sysf, modf, af, _ = build(tag, mut, "free")
    df, Lf = deriv_curve(sysf, modf, MORPHS, 1500, 120, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(bnd - fre, MORPHS)) + (rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf))


def main():
    print("=== E341 CLEAN benchmark (metal-free, moderate ΔΔG, isosteric, 2 replicates) ===", flush=True)
    res = []
    for tag, mut, exp in CASES:
        vals = []
        for rep in range(2):
            t = time.time()
            try:
                vals.append(one(tag, mut))
                print(f"  {tag} {mut} rep{rep}: calc={vals[-1]:+.2f}  ({(time.time()-t)/60:.0f}min)", flush=True)
            except Exception as e:
                print(f"  {tag} {mut} rep{rep}: FAIL {str(e)[:90]}", flush=True)
        if vals:
            calc = float(np.mean(vals)); spread = float(np.std(vals)) if len(vals) > 1 else 0.0
            res.append((tag, mut, exp, calc, spread))
            print(f"  => {tag} {mut}: calc={calc:+.2f}±{spread:.2f}  exp={exp:+.2f}  |Δ|={abs(calc-exp):.2f}", flush=True)
    print("\n=== SUMMARY (clean cases) ===")
    print(f"{'case':16s} {'calc':>10s} {'exp':>6s} {'|err|':>6s}")
    for tag, mut, exp, calc, sp in res:
        print(f"{tag+' '+mut:16s} {calc:+.2f}±{sp:.2f}  {exp:+.2f}  {abs(calc-exp):.2f}")
    if len(res) >= 3:
        c = np.array([r[3] for r in res]); e = np.array([r[2] for r in res])
        from scipy.stats import pearsonr
        print(f"\nn={len(res)}  MAE={np.mean(np.abs(c-e)):.2f}  mean signed={np.mean(c-e):+.2f}  "
              f"Pearson={pearsonr(c,e)[0]:+.2f}")
        print("VERDICT: " + ("ACCURATE on clean cases (MAE<1.5) → the earlier 'failure' was case selection + "
                             "setup, NOT the method." if np.mean(np.abs(c-e)) < 1.5 else
                             "still off on clean cases → the residual is full-mutation (vdW/atoms) + "
                             "conformational sampling, the last untested levers."))


if __name__ == "__main__":
    main()
