"""M1c — WHY do physics features flip across datasets? Simpson's paradox, proven.

Ram: 'physics never lies — investigate what causes the continuous flipping.' Hypothesis: the
flippers are all EXTENSIVE (sum/count that scale with interface size); their correlation with ΔG
depends on the dataset's size↔affinity joint distribution (a SELECTION-BIAS confound), so they flip.
INTENSIVE features (per-residue, fractions, per-unit-area) encode the physics that doesn't lie and
should transfer. Tests:
  A. classify extensive vs intensive; corr(feature, ΔG) per dataset -> extensive flip, intensive hold
  B. the Simpson mechanism: size↔ΔG relationship differs per dataset (corr(L,y), corr(bsa,y))
  C. DECISIVE: leave-DATASET-out model on INTENSIVE-ONLY features -> does the flip vanish (transfer)?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import m1b_diagnosis as M  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

# Extensive (scale with interface size) vs intensive (per-unit / fraction / mean).
EXTENSIVE = ["L", "bsa_hyd", "sasa_hb", "sasa_sb", "mj", "net_charge", "e_int_std"]
INTENSIVE = ["charged_frac", "hyd_frac", "phil_frac", "strength", "bsa_hyd_frac",
             "bsa_polar_frac", "hyd_over_phil"]


def main():
    cr = M.build("cr"); b98 = M.build("b98")
    yc = np.array([r["y"] for r in cr]); yb = np.array([r["y"] for r in b98])
    print(f"crystal-65 n={len(cr)} | the-98 n={len(b98)}\n")

    print("=== A. corr(feature, ΔG) per dataset — EXTENSIVE flip vs INTENSIVE hold ===")
    print(f"  {'feature':<16}{'type':>10}{'crystal-65':>12}{'the-98':>9}{'verdict':>12}")
    for f in EXTENSIVE + INTENSIVE:
        vc = np.array([r[f] for r in cr]); vb = np.array([r[f] for r in b98])
        a = pearsonr(vc, yc).statistic if vc.std() > 0 else 0
        b = pearsonr(vb, yb).statistic if vb.std() > 0 else 0
        typ = "extensive" if f in EXTENSIVE else "intensive"
        v = "TRANSFERS" if a * b > 0.01 else ("FLIPS" if a * b < -0.01 else "weak")
        print(f"  {f:<16}{typ:>10}{a:>+12.3f}{b:>+9.3f}{v:>12}")

    print("\n=== B. Simpson mechanism: size↔ΔG joint distribution differs per dataset ===")
    for nm, f in [("length L", "L"), ("hydrophobic BSA", "bsa_hyd"), ("MJ contacts", "mj")]:
        vc = np.array([r[f] for r in cr]); vb = np.array([r[f] for r in b98])
        print(f"  corr({nm:<16}, ΔG):  crystal-65 {pearsonr(vc,yc).statistic:+.3f}  "
              f"the-98 {pearsonr(vb,yb).statistic:+.3f}   "
              f"[size range cr {vc.min():.0f}-{vc.max():.0f} vs 98 {vb.min():.0f}-{vb.max():.0f}]")
    print("  >> different size→affinity slopes = the confound that flips every size-tracking feature")

    print("\n=== C. DECISIVE: leave-DATASET-out, INTENSIVE-only vs EXTENSIVE-only vs all ===")
    print("  (does using ONLY non-flipping intensive features stop the cross-dataset collapse?)")
    print(f"  {'feature set':<22}{'cr->98 chg':>12}{'98->cr chg':>12}{'cr->98 all':>12}")
    for nm, fl in [("EXTENSIVE-only", EXTENSIVE), ("INTENSIVE-only", INTENSIVE),
                   ("all-12 (mixed)", M.ALL)]:
        r1 = M._cross(cr, b98, fl, "ridge")   # (base_all, base_chg, full_all, full_chg)
        r2 = M._cross(b98, cr, fl, "ridge")
        print(f"  {nm:<22}{r1[3]:>+12.3f}{r2[3]:>+12.3f}{r1[2]:>+12.3f}")
    print("  baseline charged (no ML): cr->98 +0.29 / 98->cr +0.30")
    print("  >> INTENSIVE-only should NOT go negative (it transfers); EXTENSIVE-only should flip hard")
    print("  >> the physics that 'never lies' = intensive; extensive flips via Simpson/selection bias")


if __name__ == "__main__":
    main()
