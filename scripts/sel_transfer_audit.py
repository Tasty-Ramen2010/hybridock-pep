#!/usr/bin/env python
"""Feature-level post-mortem: what transfers from our docked blocks to the Coventry grid.

THE FACT TO EXPLAIN. A ranker trained on docked all-by-all blocks reaches leave-block-out
AUC 0.740 on its own data and 0.628 on Coventry, while plain ref2015 interface energy put through
the same cancellation reaches 0.735. The learned model is not short of capacity -- it is short of
features that mean the same thing in both places. This takes every feature apart and asks four
questions of each, all after the same within-block median polish the model itself sees, so
nothing here is measuring a main effect:

  DISCRIMINATION   single-feature AUC, cognate against non-binder, on each dataset separately.
  SIGN             which direction the feature points. A feature that says "more apolar contact
                   means cognate" on training and the reverse on Coventry is worse than useless:
                   the model has learned a weight whose sign is wrong on the test set.
  DRIFT            standardised mean difference and spread ratio between the two datasets. A
                   feature can keep its sign and still be useless if Coventry sits outside the
                   range it was fitted on.
  REDUNDANCY       how much of ref2015's polished interface energy our non-Rosetta features can
                   reconstruct. If that is high, "ours only" is just a noisy copy of the bar and
                   cannot beat it; if it is low, our features are measuring something else, and
                   the discrimination columns say whether that something is signal.

AND THE QUESTION UNDERNEATH ALL OF IT: which part of ref2015 is actually doing the selectivity
work? The per-term AUCs on the Coventry grid answer that directly, and they are the thing to
replace -- not "Rosetta" as a black box.

Usage: sel_transfer_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)

ENERGY = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
          "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb", "i_total"]
OURS = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
        "c_aromatic_frac", "c_mixed_frac", "hydro_pocket", "hydro_product", "hydro_absdiff",
        "charge_product", "pocket_charge_per_res", "n_contact_per_res",
        "rec_atoms_touched_per_res", "burial_mean", "burial_sd", "burial_cv", "burial_max",
        "frac_res_contacting"]
ALL = ENERGY + OURS


def polish(M: np.ndarray) -> np.ndarray:
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive scores below a random negative) -- lower is better, as with energy."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p < neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def load_train(agg: str = "best"):
    """Per-cell polished features from the docked blocks, plus the cognate label."""
    grid = defaultdict(dict)
    for line in (ROOT / "logs/sel_dock_features.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if agg in r:
                grid[r["block"]][(r["pep_idx"], r["rec_idx"])] = r[agg]
    X, y = [], []
    for cells in grid.values():
        n = max(max(k) for k in cells) + 1
        if len(cells) != n * n or n < 5:
            continue
        A = np.nan_to_num(np.array([[[float(cells[(i, j)].get(f, 0.0)) for f in ALL]
                                     for j in range(n)] for i in range(n)]), nan=0.0,
                          posinf=0.0, neginf=0.0)
        C = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
        for i in range(n):
            for j in range(n):
                X.append(C[i, j]); y.append(int(i == j))
    return np.array(X), np.array(y)


def load_coventry(agg: str = "best"):
    rows = {}
    for line in (ROOT / "logs/sel_coventry_features_hard.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if agg in r:
                rows[(r["peptide"], r["binder"])] = r[agg]
    A = np.nan_to_num(np.array([[[float(rows[(p, b)].get(f, 0.0)) for f in ALL]
                                 for b in ORDER] for p in ORDER]), nan=0.0,
                      posinf=0.0, neginf=0.0)
    C = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    X, y = [], []
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            m = truth.get((p, b), {}).get("measured")
            if i == j:
                X.append(C[i, j]); y.append(1)
            elif m == "0":
                X.append(C[i, j]); y.append(0)
    return np.array(X), np.array(y), C


def main() -> None:
    Xt, yt = load_train()
    Xc, yc, _ = load_coventry()
    print(f"training blocks: {len(yt)} cells, {yt.sum()} cognate")
    print(f"coventry grid:   {len(yc)} cells, {yc.sum()} cognate "
          f"(diagonal vs measured non-binders only)\n")

    print("PER-FEATURE TRANSFER.  AUC is sign-corrected to the TRAINING direction, so 0.500 is")
    print("chance and below 0.500 means the feature points the WRONG WAY on Coventry.")
    print("drift = (mean_cov - mean_train) / sd_train.  spread = sd_cov / sd_train.\n")
    print(f"  {'feature':<28}{'AUC train':>10}{'AUC cov':>9}{'gap':>8}{'drift':>8}{'spread':>8}")

    recs = []
    for k, f in enumerate(ALL):
        at = auc(Xt[yt == 1, k], Xt[yt == 0, k])
        sign = 1.0 if at >= 0.5 else -1.0          # orient so training says "lower = cognate"
        at_s = at if sign > 0 else 1 - at
        ac = auc(sign * Xc[yc == 1, k], sign * Xc[yc == 0, k])
        # a term that median-polishes to identically zero on the training blocks (a Rosetta
        # term with no interface contribution) has no scale to compare against -- report it
        # as degenerate rather than dividing by 1e-12 and printing a number in the billions
        sd = float(Xt[:, k].std())
        degen = sd < 1e-6
        recs.append(dict(f=f, at=at_s, ac=ac, gap=at_s - ac,
                         drift=float("nan") if degen else (Xc[:, k].mean() - Xt[:, k].mean()) / sd,
                         spread=float("nan") if degen else float(Xc[:, k].std()) / sd,
                         kind="ref2015" if f in ENERGY else "ours"))
    for r in sorted(recs, key=lambda r: -r["ac"]):
        flag = "  <-- flips" if r["ac"] < 0.47 else ("  <-- holds" if r["ac"] > 0.60 else "")
        print(f"  {r['f']:<28}{r['at']:>10.3f}{r['ac']:>9.3f}{r['gap']:>8.3f}"
              f"{r['drift']:>8.2f}{r['spread']:>8.2f}{flag}")

    holds = [r for r in recs if r["ac"] > 0.60]
    flips = [r for r in recs if r["ac"] < 0.47]
    print(f"\n  {len(holds)} of {len(ALL)} features still discriminate on Coventry (AUC > 0.60): "
          f"{', '.join(r['f'] for r in holds) or 'none'}")
    print(f"  {len(flips)} point the wrong way (AUC < 0.47): "
          f"{', '.join(r['f'] for r in flips) or 'none'}")
    o = [r for r in recs if r["kind"] == "ours"]
    e = [r for r in recs if r["kind"] == "ref2015"]
    print(f"\n  mean AUC on training  ours {np.mean([r['at'] for r in o]):.3f}   "
          f"ref2015 {np.mean([r['at'] for r in e]):.3f}")
    print(f"  mean AUC on Coventry  ours {np.mean([r['ac'] for r in o]):.3f}   "
          f"ref2015 {np.mean([r['ac'] for r in e]):.3f}")
    print(f"  mean |drift|          ours {np.nanmean([abs(r['drift']) for r in o]):.2f}   "
          f"ref2015 {np.nanmean([abs(r['drift']) for r in e]):.2f}")

    # ---- redundancy: can our features reconstruct the bar? -------------------------------
    from sklearn.linear_model import RidgeCV
    io = ALL.index("i_total")
    oi = [ALL.index(f) for f in OURS]
    for name, X, y in (("training blocks", Xt, yt), ("coventry grid", Xc, yc)):
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[:, oi], X[:, io])
        r2 = m.score(X[:, oi], X[:, io])
        pred = m.predict(X[:, oi])
        print(f"\n  our 18 features reconstruct polished i_total on {name}: R2 = {r2:.3f} "
              f"(r = {np.corrcoef(pred, X[:, io])[0, 1]:+.3f})")

    # ---- which part of ref2015 carries the selectivity ------------------------------------
    print("\nWHICH ref2015 TERM IS DOING THE WORK ON COVENTRY (single term, after cancellation):")
    print(f"  {'term':<24}{'AUC train':>10}{'AUC cov':>9}")
    for f in ENERGY:
        k = ALL.index(f)
        at = auc(Xt[yt == 1, k], Xt[yt == 0, k])
        sign = 1.0 if at >= 0.5 else -1.0
        ac = auc(sign * Xc[yc == 1, k], sign * Xc[yc == 0, k])
        print(f"  {f:<24}{(at if sign > 0 else 1 - at):>10.3f}{ac:>9.3f}")


if __name__ == "__main__":
    main()
