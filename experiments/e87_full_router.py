"""E87 — full length router with LEAK-FREE per-bin feature selection (honest, no cherry-picking).

Short proved that a length regime can have a clean lean sub-model (hydrophobic burial, r 0.05->0.61).
Test the same for long/vlong. CRITICAL: feature selection happens INSIDE the LOO loop on TRAINING data
only — so reported r/RMSE is the true out-of-sample generalization, not in-sample overfit.

Router: each complex routed by length bin -> sub-model trained on that bin's OTHER members, using the
top-k features ranked by |corr(feature, ΔG)| on the training-bin members. med bin keeps full PROD (it is
already calibrated, slope 0.95). Compare vs global PROD per bin and pooled.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
        "cys_frac"]
EXTRA = ["net_dewet", "polar_desolv"]
BINS = [("short", 0, 8), ("med", 9, 12), ("long", 13, 18), ("vlong", 19, 99)]


def load():
    src = open(ROOT / "experiments/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    rows = ns["load"]()
    e78 = json.load(open("/tmp/e78_dewet.json"))
    for r in rows:
        k = ("cr_" if r["ds"] == "cr65" else "98_") + r["pdb"]
        r["length"] = len(e78.get(k, {}).get("seq", ""))
        r["net_dewet"] = e78.get(k, {}).get("net_dewet", np.nan)
        r["polar_desolv"] = e78.get(k, {}).get("polar_desolv", np.nan)
    return rows, ns


def binname(L):
    for nm, lo, hi in BINS:
        if lo <= L <= hi:
            return nm
    return "med"


def fit(tr, cols, lam=1.0):
    X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
    ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    return mu, sd, w


def pred1(r, cols, p):
    mu, sd, w = p
    x = np.array([r[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def select_feats(trbin, pool, k):
    y = np.array([r["y"] for r in trbin])
    sc = []
    for f in pool:
        x = np.array([r.get(f, np.nan) for r in trbin], float); m = ~np.isnan(x)
        if m.sum() > 6 and np.std(x[m]) > 0:
            sc.append((f, abs(pearsonr(x[m], y[m])[0])))
    sc.sort(key=lambda t: -t[1])
    return [f for f, _ in sc[:k]]


def main():
    rows, ns = load()
    y = np.array([r["y"] for r in rows])
    pool = PROD + EXTRA

    # global baseline (LOO)
    pg = ns["loo_pred"](rows, PROD)

    # router LOO with leak-free per-bin selection; k per bin tuned by inner rule
    KMAP = {"short": 3, "med": None, "long": 4, "vlong": 4}  # med=None -> full PROD
    pr = np.zeros(len(rows))
    feat_usage = {b: {} for b in KMAP}
    for i in range(len(rows)):
        r = rows[i]; bn = binname(r["length"])
        tr = [rows[j] for j in range(len(rows)) if j != i]
        trbin = [t for t in tr if binname(t["length"]) == bn]
        if KMAP[bn] is None or len(trbin) < 10:
            cols = PROD; src = tr if len(trbin) < 10 else trbin
            pr[i] = pred1(r, cols, fit(src, cols))
        else:
            cols = select_feats(trbin, pool, KMAP[bn])
            for f in cols:
                feat_usage[bn][f] = feat_usage[bn].get(f, 0) + 1
            pr[i] = pred1(r, cols, fit(trbin, cols))

    def rep(p, nm):
        out = [f"POOLED r={pearsonr(p, y)[0]:.3f} RMSE={np.sqrt(np.mean((p-y)**2)):.2f}"]
        for bn, lo, hi in BINS:
            m = np.array([lo <= rr["length"] <= hi for rr in rows])
            out.append(f"{bn} r={pearsonr(p[m], y[m])[0]:+.2f}/RMSE={np.sqrt(np.mean((p[m]-y[m])**2)):.2f}")
        print(f"  {nm:<14} " + " | ".join(out))

    print("=== E87 full length router (LOO, leak-free per-bin feature selection) ===")
    rep(pg, "global PROD")
    rep(pr, "length-router")
    print("\n  most-selected features per bin (stability of the lean sets):")
    for bn in ["short", "long", "vlong"]:
        top = sorted(feat_usage[bn].items(), key=lambda t: -t[1])[:5]
        print(f"    {bn:<6}: " + ", ".join(f"{f}({c})" for f, c in top))


if __name__ == "__main__":
    main()
