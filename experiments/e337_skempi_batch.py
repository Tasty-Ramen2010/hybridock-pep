"""E337 — map the charged-FEP accuracy across ΔΔG magnitudes (does it get small/exposed right, miss big buried?).

E335 showed converging doesn't help D75N; the salt bridge stays formed (diagnostic) → the +1.5-vs-+5.9 gap is the
missing electronic polarization of a buried ion pair (JACS 2022: fixed-charge FFs underestimate buried ion pairs
by tens of kcal). Prediction: our charge-morph should get SMALL/solvent-exposed charge mutations ~right and
progressively UNDER-estimate the large buried-salt-bridge ones. This runs several SKEMPI isosteric charge
mutations spanning ΔΔG_exp and tabulates calc vs exp (E334 fast protocol — extra sampling doesn't change it).

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e337_skempi_batch.py
"""
from __future__ import annotations
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build, deriv_curve
from e332_g1_charged_corrected import rocklin_correction

# (tag, mutation, exp ΔΔG) spanning ~0 → large; smaller receptors first for speed
CASES = [
    ("1VFB_AB_C", "DC58N", -0.13),   # near zero
    ("1K8R_A_B", "DA38N", 1.97),     # small
    ("1E96_A_B", "DA38N", 2.16),     # small-med
    ("1IAR_A_B", "EA9Q", 3.11),      # medium
    ("2O3B_A_B", "EB24Q", 5.40),     # large buried salt bridge (Glu24-Arg93)
    ("2O3B_A_B", "DB75N", 5.90),     # large buried salt bridge (Asp75-Lys101)
]
OUT = "/home/igem/unknown_software/data/e337_skempi_map.json"


def one(tag, mut):
    morphs = [0.0, 0.25, 0.5, 0.75, 1.0]
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    db, Lb = deriv_curve(sysb, modb, morphs, 1000, 80, 100)
    sysf, modf, af, _ = build(tag, mut, "free")
    df, Lf = deriv_curve(sysf, modf, morphs, 1000, 150, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    be = np.array([v[1] for v in db]); fe = np.array([v[1] for v in df])
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ddg = float(_trap(bnd - fre, morphs)) + (rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf))
    w = np.gradient(np.array(morphs)); err = float(np.sqrt(np.sum((w * np.sqrt(be**2 + fe**2))**2)))
    return ddg, err


def main():
    res = []
    for tag, mut, exp in CASES:
        t = time.time()
        try:
            calc, err = one(tag, mut)
            res.append(dict(tag=tag, mut=mut, exp=exp, calc=round(calc, 2), err=round(err, 2)))
            print(f"{tag} {mut}: calc={calc:+.2f}±{err:.2f}  exp={exp:+.2f}  |Δ|={abs(calc-exp):.2f}  "
                  f"({(time.time()-t)/60:.0f}min)", flush=True)
        except Exception as e:
            print(f"{tag} {mut}: FAIL {str(e)[:80]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    if len(res) >= 3:
        c = np.array([r["calc"] for r in res]); e = np.array([r["exp"] for r in res])
        from scipy.stats import pearsonr
        print(f"\nn={len(res)}  Pearson(calc,exp)={pearsonr(c, e)[0]:+.2f}  "
              f"mean signed err (calc-exp)={np.mean(c - e):+.2f} (negative ⇒ we UNDER-estimate)  "
              f"MAE={np.mean(np.abs(c - e)):.2f}")
        print("PATTERN: if small ~ok and large under-estimated ⇒ missing polarization of buried ion pairs.")


if __name__ == "__main__":
    main()
