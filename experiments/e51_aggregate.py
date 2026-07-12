"""E51-aggregate — combine all SKEMPI ΔΔG caches into the total cross-complex validation.

Reports per-complex and POOLED ΔΔG correlation (our MM-GBSA vs experimental), with the key
robust metrics, positioned vs FoldX/flex-ddG/FEP. The pooled cross-complex number is the
'LIE-level across datasets' claim the campaign is built to test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CXS = ["1PPF_E_I", "1CHO_EFG_I", "1R0R_E_I", "3SGB_E_I", "1AO7_ABC_DE"]


def main():
    allp, alle = [], []
    print(f"=== SKEMPI ΔΔG validation — OUR MM-GBSA selectivity vs experimental ===")
    print(f"  {'complex':<14}{'n':>5}{'Pearson':>9}{'Spearman':>10}{'RMSE':>7}")
    for cx in CXS:
        f = Path(f"/tmp/e51_{cx}.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        pairs = [(v["ddg_pred"], v["ddg_exp"]) for k, v in d.items()
                 if k != "WT" and abs(v.get("ddg_pred", 99)) < 50]
        if len(pairs) < 5:
            print(f"  {cx:<14}{len(pairs):>5}  (too few)"); continue
        p = np.array([x[0] for x in pairs]); e = np.array([x[1] for x in pairs])
        allp += list(p); alle += list(e)
        print(f"  {cx:<14}{len(pairs):>5}{pearsonr(p,e).statistic:>+9.3f}"
              f"{spearmanr(p,e).statistic:>+10.3f}{np.sqrt(((p-e)**2).mean()):>7.2f}")
    if len(allp) >= 10:
        p = np.array(allp); e = np.array(alle)
        print(f"  {'-'*44}")
        print(f"  {'POOLED':<14}{len(p):>5}{pearsonr(p,e).statistic:>+9.3f}"
              f"{spearmanr(p,e).statistic:>+10.3f}{np.sqrt(((p-e)**2).mean()):>7.2f}")
        # stabilizers: correlation after a linear rescale (MM-GBSA over-predicts magnitude)
        a, b = np.polyfit(p, e, 1)
        print(f"  rescaled (MM-GBSA over-predicts ~{1/a:.1f}x): RMSE {np.sqrt(((a*p+b-e)**2).mean()):.2f}")
        print(f"\n  LITERATURE: FoldX r≈0.5 | flex-ddG r≈0.55 | FEP r≈0.8 | single-pt MM-GBSA 0.3-0.5")
        print(f"  >> POOLED n={len(p)} is the 'is selectivity LIE-level across complexes' verdict")


if __name__ == "__main__":
    main()
