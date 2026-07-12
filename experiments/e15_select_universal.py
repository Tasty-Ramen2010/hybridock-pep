"""E15 — sign-stability gate + forward-selection on CROSS-DATASET TRANSFER.

Uses e14 rich features. For the universal within-protein ΔΔG model:
  1. sign-stability: within-group standardized slope on crystal vs PEPBI; keep
     only features with the SAME sign on both (and physically correct: stronger
     binding = more negative ΔG).
  2. forward selection maximizing AVERAGE cross-dataset transfer r (crystal-fit ->
     PEPBI within-group, and PEPBI-fit -> crystal within-group). This is the honest
     anti-overfit metric: a feature earns inclusion only if it helps the OTHER
     dataset, not in-sample.
  3. report final universal formula + pooled within-group r vs the
     hb_density+n_contact baseline (pooled 0.345 / transfer cr->pb 0.375).
"""
from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

CAND = ["n_contact", "hb_count", "hb_density", "hb_sc", "salt_bridge", "sb_density",
        "hydrophobic_cc", "elec_compl", "aromatic_cc", "contact_pairs",
        "pack_density", "min_gap_mean", "bsa", "bsa_hphobic"]


def demean(rows, keys):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["grp"], []).append(i)
    Yd, Xd = [], []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        y = np.array([rows[i]["y"] for i in idxs], float)
        X = np.array([[rows[i].get(k, np.nan) for k in keys] for i in idxs], float)
        if not np.all(np.isfinite(X)):
            continue
        Yd.append(y - y.mean())
        Xd.append(X - X.mean(0))
    if not Yd:
        return np.array([]), np.zeros((0, len(keys)))
    return np.concatenate(Yd), np.concatenate(Xd, 0)


def zfit(rows, keys):
    Yd, Xd = demean(rows, keys)
    if len(Yd) < 3:
        return None, None
    sd = Xd.std(0); sd[sd == 0] = 1.0
    beta, *_ = np.linalg.lstsq(Xd / sd, Yd, rcond=None)
    return beta, sd


def transfer_r(train, test, keys):
    """fit z-slopes on train within-group, predict test within-group ΔΔG."""
    beta, sd = zfit(train, keys)
    if beta is None:
        return float("nan")
    Yd, Xd = demean(test, keys)
    if len(Yd) < 3:
        return float("nan")
    sd_t = Xd.std(0); sd_t[sd_t == 0] = 1.0
    pred = (Xd / sd_t) @ beta
    if np.std(pred) == 0:
        return float("nan")
    return pearsonr(pred, Yd).statistic


def avg_transfer(cr, pb, keys):
    a = transfer_r(cr, pb, keys)
    b = transfer_r(pb, cr, keys)
    return np.nanmean([a, b]), a, b


def main():
    cr = json.loads(Path("/tmp/e14_cr.json").read_text())
    pb = json.loads(Path("/tmp/e14_pb.json").read_text())
    avail = [k for k in CAND if all(np.isfinite(r.get(k, np.nan)) for r in (cr[:1] + pb[:1]))
             and k in cr[0] and k in pb[0]]
    print(f"crystal n={len(cr)} pepbi n={len(pb)}; features available: {avail}")

    print("\n=== sign-stability (within-group standardized slope) ===")
    print(f"{'feature':<16}{'β crystal':>11}{'β pepbi':>11}{'stable?':>9}")
    stable = []
    for k in avail:
        bc, _ = zfit(cr, [k])
        bp, _ = zfit(pb, [k])
        if bc is None or bp is None:
            continue
        ss = bc[0] * bp[0] > 0
        if ss:
            stable.append(k)
        print(f"{k:<16}{bc[0]:>11.3f}{bp[0]:>11.3f}{'YES' if ss else 'no':>9}")

    print(f"\nsign-stable features: {stable}")

    print("\n=== forward selection on AVG cross-dataset transfer r ===")
    chosen, best = [], -1.0
    pool = list(stable)
    for _ in range(len(pool)):
        scored = []
        for k in pool:
            avg, a, b = avg_transfer(cr, pb, chosen + [k])
            scored.append((avg, k, a, b))
        scored.sort(reverse=True)
        if not scored or scored[0][0] <= best + 0.005:
            break
        best, pick, a, b = scored[0]
        chosen.append(pick)
        pool.remove(pick)
        print(f"  + {pick:<16} avg_transfer={best:+.3f}  (cr->pb={a:+.3f}, pb->cr={b:+.3f})")

    print(f"\n>>> SELECTED: {chosen}")
    # baseline comparison
    base_avg, ba, bb = avg_transfer(cr, pb, ["hb_density", "n_contact"])
    sel_avg, sa, sb = avg_transfer(cr, pb, chosen)
    print(f"  baseline [hb_density,n_contact]: avg={base_avg:+.3f} (cr->pb {ba:+.3f}, pb->cr {bb:+.3f})")
    print(f"  selected {chosen}: avg={sel_avg:+.3f} (cr->pb {sa:+.3f}, pb->cr {sb:+.3f})")

    # pooled within-group fit + r (the headline universal-formula number)
    pooled = cr + pb
    Yd, Xd = demean(pooled, chosen)
    sd = Xd.std(0); sd[sd == 0] = 1
    beta, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
    pred = Xd @ beta
    print(f"\n=== UNIVERSAL FORMULA (pooled within-protein, {len(Yd)} pairs) ===")
    print("  ΔΔG ≈ " + "  ".join(f"{beta[i]:+.3f}·Δ{chosen[i]}" for i in range(len(chosen))))
    print(f"  pooled within-group r = {pearsonr(pred, Yd).statistic:+.3f}")
    # also pooled r of baseline for reference
    Yb, Xb = demean(pooled, ["hb_density", "n_contact"])
    bb2, *_ = np.linalg.lstsq(Xb, Yb, rcond=None)
    print(f"  (baseline hb_density+n_contact pooled r = {pearsonr(Xb @ bb2, Yb).statistic:+.3f})")


if __name__ == "__main__":
    main()
