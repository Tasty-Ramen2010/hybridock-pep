"""E16 — the make-or-break test: does the universal ΔΔG ranker work PER PROTEIN,
or is pooled r=0.45 averaging winners against losers?

PEPBI has real within-group power (binding groups of mutants, TtSlyD n=125 etc).
Test the H-bond+aromatic universal model with LEAVE-ONE-BINDING-GROUP-OUT CV:
  - fit universal (standardized, within-group-demeaned) slopes on ALL OTHER groups
  - apply to the held-out group's demeaned features; rank vs held-out demeaned ΔG
  - report pooled honest r AND the per-group Spearman DISTRIBUTION

Decision:
  * if most held-out groups rank in the correct direction (fraction positive >> 0.5,
    median Spearman clearly >0) -> the within-target ranker is REAL -> build it.
  * if it is a coin flip (fraction ~0.5, median ~0) -> pooled r is a mirage -> do NOT
    ship a per-target ranker; say so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

MODELS = {
    "hb_count+aromatic": ["hb_count", "aromatic_cc"],
    "hb_count+aromatic+bsa": ["hb_count", "aromatic_cc", "bsa"],
    "hb_count only": ["hb_count"],
}


def groups_of(rows):
    g = {}
    for i, r in enumerate(rows):
        g.setdefault(r["grp"], []).append(i)
    return g


def demean_fit(rows, idxs_by_group, keys, exclude=None):
    """Fit standardized within-group-demeaned slopes on all groups except `exclude`."""
    Yd, Xd = [], []
    for gid, idxs in idxs_by_group.items():
        if gid == exclude or len(idxs) < 2:
            continue
        y = np.array([rows[i]["y"] for i in idxs], float)
        X = np.array([[rows[i].get(k, np.nan) for k in keys] for i in idxs], float)
        if not np.all(np.isfinite(X)):
            continue
        Yd.append(y - y.mean())
        Xd.append(X - X.mean(0))
    Yd = np.concatenate(Yd)
    Xd = np.concatenate(Xd, 0)
    sd = Xd.std(0); sd[sd == 0] = 1.0
    beta, *_ = np.linalg.lstsq(Xd / sd, Yd, rcond=None)
    return beta, sd


def main():
    rows = json.loads(Path("/tmp/e14_pb.json").read_text())
    G = groups_of(rows)
    multi = {g: idx for g, idx in G.items() if len(idx) >= 2}
    print(f"PEPBI n={len(rows)}; binding groups with >=2 members: {len(multi)}; "
          f">=4 members: {sum(len(v) >= 4 for v in multi.values())}")

    for name, keys in MODELS.items():
        all_pred, all_y = [], []
        per_group = []
        for gid, idxs in multi.items():
            if not all(all(np.isfinite(rows[i].get(k, np.nan)) for k in keys) for i in idxs):
                continue
            beta, sd = demean_fit(rows, multi, keys, exclude=gid)
            y = np.array([rows[i]["y"] for i in idxs], float)
            X = np.array([[rows[i][k] for k in keys] for i in idxs], float)
            yd = y - y.mean()
            xd = X - X.mean(0)
            pred = (xd / sd) @ beta
            all_pred.append(pred); all_y.append(yd)
            if len(idxs) >= 4 and np.std(pred) > 0:
                per_group.append((gid, len(idxs), spearmanr(pred, y).statistic))
        ap = np.concatenate(all_pred); ay = np.concatenate(all_y)
        pooled_r = pearsonr(ap, ay).statistic
        rhos = np.array([r for *_, r in per_group])
        print(f"\n=== {name} (LEAVE-GROUP-OUT) ===")
        print(f"  pooled held-out within-group r = {pooled_r:+.3f}  (n_pairs={len(ay)})")
        print(f"  per-group Spearman (groups n>=4, k={len(rhos)}): "
              f"median={np.median(rhos):+.2f}  mean={rhos.mean():+.2f}")
        print(f"  fraction of groups in CORRECT direction (rho>0): "
              f"{np.mean(rhos > 0):.0%}")
        print(f"  distribution: " + "  ".join(
            f"{lab}={np.percentile(rhos, p):+.2f}" for lab, p in
            [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90)]))
        # show the biggest groups individually (most trustworthy)
        big = sorted(per_group, key=lambda t: -t[1])[:8]
        print("  largest groups:  " + " | ".join(
            f"{str(g)[:18]}(n{n}):{r:+.2f}" for g, n, r in big))


if __name__ == "__main__":
    main()
