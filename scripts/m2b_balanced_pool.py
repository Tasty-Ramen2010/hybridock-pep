"""M2b — Ram's balancing idea: does an EVEN distribution of weak/strong x short/long break the
Simpson confound and improve transfer?

The sign-flip arises because each dataset's size↔affinity joint distribution is selection-biased
(crystal-65 corr(L,ΔG)=+0.46 strong-only; the-98 −0.40 includes weak). Hypothesis: combine both and
STRATIFY to even coverage across (ΔG tertile)x(L tertile) so length and affinity become independent
-> the spurious size correlation -> 0 -> features can't flip -> model learns real per-unit physics.

Tests: (1) confound corr(L,ΔG) biased-pool vs balanced-pool, (2) leave-one-complex-out on the
balanced pool (intensive features) vs unbalanced, charge-stratified, Spearman.
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

INTENSIVE = ["charged_frac", "hyd_frac", "phil_frac", "strength", "bsa_hyd_frac", "bsa_polar_frac"]
rng = np.random.default_rng(0)


def balance(rows, n_per_cell=None):
    """Stratify by ΔG tertile x L tertile (9 cells); sample even n per cell to break the confound."""
    y = np.array([r["y"] for r in rows]); L = np.array([r["L"] for r in rows])
    yb = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    lb = np.digitize(L, np.quantile(L, [1 / 3, 2 / 3]))
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((yb[i], lb[i]), []).append(i)
    k = n_per_cell or min(len(v) for v in cells.values())
    keep = []
    for v in cells.values():
        keep += list(rng.choice(v, size=min(k, len(v)), replace=False))
    return [rows[i] for i in keep]


def loo(rows, feats):
    y = np.array([r["y"] for r in rows]); cf = np.array([r["cf"] for r in rows])
    X = np.array([[r[f] for f in feats] for r in rows]); xe = np.array([r["e_int_perL"] for r in rows])
    full = np.zeros(len(rows))
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        lo, hi = np.percentile(xe[tr], 5), np.percentile(xe[tr], 95)
        a, b = np.polyfit(np.clip(xe[tr], lo, hi), y[tr], 1)
        base = a * np.clip(xe[i], lo, hi) + b
        rtr = y[tr] - (a * np.clip(xe[tr], lo, hi) + b)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); R = 10 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ rtr)
        full[i] = base + np.r_[1, (X[i] - mu) / sd] @ w
    h = cf >= 0.3
    return spearmanr(full, y).statistic, spearmanr(full[h], y[h]).statistic


def main():
    cr = M.build("cr"); b98 = M.build("b98"); pool = cr + b98
    print(f"crystal-65 n={len(cr)} | the-98 n={len(b98)} | pool n={len(pool)}\n")

    print("=== 1. confound: corr(L, ΔG) and corr(bsa_hyd, ΔG) — biased vs BALANCED ===")
    for nm, rows in [("crystal-65", cr), ("the-98", b98), ("pool (biased)", pool)]:
        y = np.array([r["y"] for r in rows])
        print(f"  {nm:<16} corr(L,ΔG)={pearsonr([r['L'] for r in rows],y).statistic:+.3f}  "
              f"corr(bsa_hyd,ΔG)={pearsonr([r['bsa_hyd'] for r in rows],y).statistic:+.3f}  n={len(rows)}")
    # balanced pool (averaged over resamples for stability)
    cLs, cBs, ns = [], [], []
    for s in range(20):
        global rng; rng = np.random.default_rng(s)
        bp = balance(pool)
        y = np.array([r["y"] for r in bp])
        cLs.append(pearsonr([r["L"] for r in bp], y).statistic)
        cBs.append(pearsonr([r["bsa_hyd"] for r in bp], y).statistic); ns.append(len(bp))
    print(f"  {'pool BALANCED':<16} corr(L,ΔG)={np.mean(cLs):+.3f}  corr(bsa_hyd,ΔG)={np.mean(cBs):+.3f}  "
          f"n={int(np.mean(ns))} (avg 20 resamples)")
    print("  >> if balanced corr(L,ΔG)->0, the selection-bias confound that flips features is BROKEN")

    print("\n=== 2. leave-one-complex-out: unbalanced pool vs BALANCED pool (intensive feats) ===")
    ua, uc = loo(pool, INTENSIVE)
    print(f"  unbalanced pool (n={len(pool)})   all={ua:+.3f}  charged={uc:+.3f}")
    bas, bcs = [], []
    for s in range(15):
        rng = np.random.default_rng(s); bp = balance(pool)
        a, c = loo(bp, INTENSIVE); bas.append(a); bcs.append(c)
    print(f"  BALANCED pool (n~{int(np.mean(ns))})    all={np.mean(bas):+.3f}  charged={np.mean(bcs):+.3f}  (avg 15)")
    print("  >> does balancing improve transfer (esp charged) by removing the confound?")


if __name__ == "__main__":
    main()
