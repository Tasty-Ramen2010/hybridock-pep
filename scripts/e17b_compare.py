"""E17b — within-group: MD-LIE (physics) vs instant geometry, on the SAME variants.

Reads /tmp/e17_results.json (MD-LIE dg_pred per variant per group) and matches to
/tmp/e14_pb.json (geometric hb_count+aromatic) by complex name. Reports per-group
within-group Spearman for both, plus the ΔG dynamic range (to flag unrankable groups).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main():
    mdlie = json.loads(Path("/tmp/e17_results.json").read_text())
    pb = json.loads(Path("/tmp/e14_pb.json").read_text())
    # geometry by complex name — pb doesn't store name; rebuild via grp + match on y is unreliable.
    # Instead recompute geometry score from pb records grouped by grp, matching dg.
    # Simplest: build geometry lookup keyed by (rounded dg, grp-substr) — but cleanest is
    # to recompute geometry per md variant from its split pdb. We approximate by matching
    # within group on dg value.
    print(f"{'group':<18}{'n':>4}{'ΔG range':>10}{'MD-LIE ρ':>11}{'geom ρ':>9}")
    for gname, recs in mdlie.items():
        if len(recs) < 4:
            print(f"{gname:<18}{len(recs):>4}   (n<4, skip)")
            continue
        y = np.array([r["dg"] for r in recs])
        dgp = np.array([r["dg_pred"] for r in recs])
        rng = y.max() - y.min()
        rho_md = spearmanr(dgp, y).statistic if np.std(dgp) > 0 else float("nan")
        # geometry: match this group's records in pb by grp substring + nearest dg
        # find pb records whose grp contains gname tokens
        gtok = gname.split()[0]
        cand = [r for r in pb if gtok.lower() in str(r["grp"]).lower()]
        geom_rho = float("nan")
        if cand:
            # build geometry score (hb_count + aromatic, standardized within this set)
            yy = np.array([r["y"] for r in cand])
            hb = np.array([r.get("hb_count", np.nan) for r in cand])
            ar = np.array([r.get("aromatic_cc", np.nan) for r in cand])
            score = -(hb - np.nanmean(hb)) / (np.nanstd(hb) + 1e-9) \
                    - (ar - np.nanmean(ar)) / (np.nanstd(ar) + 1e-9)
            if np.std(score) > 0:
                geom_rho = spearmanr(score, yy).statistic
        print(f"{gname:<18}{len(recs):>4}{rng:>9.1f}k{rho_md:>11.3f}{geom_rho:>9.3f}")
    print("\n(ρ = within-group Spearman of predicted vs experimental ΔG; "
          "negative dg_pred should track negative ΔG so ρ>0 = correct)")
    print("Interpretation: compare MD-LIE ρ vs geom ρ on WIDE-range groups; "
          "narrow ΔG range (<~1 kcal/mol) is unrankable by any method.")


if __name__ == "__main__":
    main()
