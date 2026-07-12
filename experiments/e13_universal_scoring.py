"""E13 — build a sign-stable (universal) scoring function via within-protein fixed effects.

Mechanism proven (e12): the length/contact sign-flip is Simpson's paradox with
per-protein BASELINE affinity as the confounder. The universal physics lives in
the WITHIN-protein (demeaned) relationship. So the correct model is mixed-effects:

    ΔG(pep, prot) = b_prot  +  Σ_k β_k · feature_k(pep, prot)

where b_prot is a per-protein baseline (random intercept, absorbs the confounder)
and β_k are UNIVERSAL fixed-effect slopes. We estimate β_k by within-group
demeaning (the fixed-effects estimator) and test the central claim:

    Do the β_k fit on PEPBI keep their SIGN and predict within-group ΔΔG on
    crystal-65 (and vice versa)?  If yes, we have universal, transferable physics.

For a NEW protein, b_prot comes from one known reference binder (=> relative ΔΔG),
which is the mathematically honest form of "calibrate to a reference".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["nc", "hb_density", "nis_p", "L"]


def demean(rows, keys):
    """Within-group demean y and features. Returns (Yd, Xd[n,k]) over multi-member groups."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["grp"], []).append(i)
    Yd, Xd = [], []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        y = np.array([rows[i]["y"] for i in idxs])
        X = np.array([[rows[i][k] for k in keys] for i in idxs], float)
        Yd.append(y - y.mean())
        Xd.append(X - X.mean(0))
    return np.concatenate(Yd), np.concatenate(Xd, axis=0)


def fit_slopes(rows, keys):
    Yd, Xd = demean(rows, keys)
    # standardize columns for comparable coefficients
    sd = Xd.std(0)
    sd[sd == 0] = 1.0
    Xs = Xd / sd
    beta, *_ = np.linalg.lstsq(Xs, Yd, rcond=None)
    return beta, sd


def within_pred_r(rows, keys, beta, sd):
    """Apply standardized slopes to within-group-demeaned features; corr with demeaned ΔG."""
    Yd, Xd = demean(rows, keys)
    pred = (Xd / sd) @ beta
    if np.std(pred) == 0:
        return float("nan")
    return pearsonr(pred, Yd).statistic


def main():
    cr = json.loads(Path("/tmp/e12_cr.json").read_text())
    pb = json.loads(Path("/tmp/e12_pb.json").read_text())
    print(f"crystal-65 n={len(cr)}  PEPBI n={len(pb)}")

    # ---- single-feature sign-stability table (standardized within-group slopes) ----
    print("\n" + "=" * 64)
    print("UNIVERSAL within-protein slopes (standardized), per dataset")
    print("sign-stable feature = candidate universal physics")
    print("=" * 64)
    print(f"{'feature':<12}{'β (crystal)':>14}{'β (PEPBI)':>14}{'sign-stable':>13}")
    stable = []
    for f in FEATS:
        bc, _ = fit_slopes(cr, [f])
        bp, _ = fit_slopes(pb, [f])
        ss = "YES" if bc[0] * bp[0] > 0 else "no"
        if ss == "YES":
            stable.append(f)
        print(f"{f:<12}{bc[0]:>14.3f}{bp[0]:>14.3f}{ss:>13}")

    # ---- candidate multi-feature scoring functions ----
    candidates = {
        "SF1: hb_density": ["hb_density"],
        "SF2: hb_density+nc": ["hb_density", "nc"],
        "SF3: stable-set": stable or ["hb_density"],
        "SF4: all4": FEATS,
    }
    print("\n" + "=" * 64)
    print("CROSS-DATASET TRANSFER (the real test of universality):")
    print("fit within-protein slopes on TRAIN, predict within-protein ΔΔG on TEST")
    print("=" * 64)
    print(f"{'scoring fn':<24}{'fit→PEPBI':>12}{'fit→cryst':>12}{'self-cr':>9}{'self-pb':>9}")
    for name, keys in candidates.items():
        bc, sdc = fit_slopes(cr, keys)   # slopes from crystal
        bp, sdp = fit_slopes(pb, keys)   # slopes from PEPBI
        # transfer: PEPBI-fit slopes applied to crystal within-group, and vice versa
        r_pb_on_cr = within_pred_r(cr, keys, bp, sdp)   # PEPBI-trained -> crystal
        r_cr_on_pb = within_pred_r(pb, keys, bc, sdc)   # crystal-trained -> PEPBI
        r_self_cr = within_pred_r(cr, keys, bc, sdc)
        r_self_pb = within_pred_r(pb, keys, bp, sdp)
        print(f"{name:<24}{r_cr_on_pb:>12.3f}{r_pb_on_cr:>12.3f}{r_self_cr:>9.3f}{r_self_pb:>9.3f}")

    # ---- the universal formula (pooled within-group fit) ----
    print("\n" + "=" * 64)
    print("UNIVERSAL FORMULA — pooled within-group fit (both datasets), key feats")
    print("=" * 64)
    pooled = cr + pb
    # ensure unique group ids across datasets
    for i, r in enumerate(cr):
        r["grp"] = f"cr_{r['grp']}"
    for i, r in enumerate(pb):
        r["grp"] = f"pb_{r['grp']}"
    pooled = cr + pb
    keys = ["hb_density", "nc"]
    Yd, Xd = demean(pooled, keys)
    sd = Xd.std(0); sd[sd == 0] = 1
    beta, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)  # raw (unstandardized) coeffs
    pred = Xd @ beta
    print(f"  ΔΔG ≈ {beta[0]:+.3f}·Δhb_density {beta[1]:+.3f}·Δn_contact   (within-protein)")
    print(f"  pooled within-group r = {pearsonr(pred, Yd).statistic:+.3f}  (n_pairs={len(Yd)})")
    print("  (negative coeffs = more H-bonds / contacts -> stronger, as physics demands)")


if __name__ == "__main__":
    main()
