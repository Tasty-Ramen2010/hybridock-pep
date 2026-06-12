"""E80 — the charged-binder gap autopsy: where does our predicted ΔG fail, and what predicts the error?

Assembles the fullest feature matrix we've ever built, makes HONEST out-of-sample (leave-one-complex-out)
predictions with the production model, then dissects the charged subset three ways:
  1. THE GAP: Pearson(pred,true) + RMSE + dynamic-range slope on charged (|Q|>=2) vs low-charge.
  2. FLIP TABLE: Pearson(feature,true) charged vs low-charge for every feature -> which carry / flip / wash.
  3. RESIDUAL AUTOPSY: Pearson(feature, true-pred) on charged -> what predicts our ERROR = missing physics.
Then: does the top residual-correlated feature actually improve charged LOO prediction, or is the error
unstructured (= irreducible single-pose floor)?
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
        "cys_frac"]


def load():
    g = json.load(open("/tmp/e69_geom_all.json"))
    e78 = json.load(open("/tmp/e78_dewet.json"))
    e74 = json.load(open("/tmp/e74_charged.json"))
    e75 = json.load(open("/tmp/e75_unsat.json"))
    electro = {r["seq"]: r for r in (json.loads(l) for l in open(ROOT / "data/electrostatic_decomp_dataset.jsonl"))}

    def key(r):
        return ("cr_" + r["pdb"]) if r["ds"] == "cr65" else ("98_" + r["pdb"])
    rows = []
    for r in g:
        k = key(r)
        e = e78.get(k)
        if e is None:
            continue
        r = dict(r)
        r["net_charge"] = e["net_charge"]; r["seq"] = e["seq"]
        for f in ["net_dewet", "polar_desolv", "hyd_dewet", "hyd_burial_flat"]:
            r[f] = e[f]
        for src, feats in [(e74, ["chg_burial", "chg_buried_frac", "hyd_shield", "chg_compl",
                                  "hyd_chg_balance", "n_buried_chg"]),
                           (e75, ["n_unsatisfied", "unsat_per_L", "satisfaction_frac", "net_satisfied"])]:
            sv = src.get(k, {})
            for f in feats:
                r[f] = sv.get(f, np.nan)
        ev = electro.get(r["seq"], {})
        for f in ["coul_per_L", "gbpol_per_L", "net_elec_per_L", "vdw", "charged_frac"]:
            r[f] = ev.get(f, np.nan)
        rows.append(r)
    return rows


def loo_pred(rows, cols, lam=1.0):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = np.arange(len(rows)) != i
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(tr.sum()), (Xt - mu) / sd])
        R = np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + lam * R, A.T @ yt)
        pred[i] = np.r_[1.0, (X[i] - mu) / sd] @ w
    return pred


def stat(pred, y):
    m = ~(np.isnan(pred) | np.isnan(y))
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    r = pearsonr(pred[m], y[m])[0]
    rmse = float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))
    slope = np.polyfit(pred[m], y[m], 1)[0]   # true ~ pred ; <1 = we compress range
    return r, rmse, slope


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    q = np.array([abs(r["net_charge"]) for r in rows])
    ch = q >= 2
    print(f"=== E80 charged-gap autopsy. n={len(rows)} (charged|Q|>=2: {ch.sum()}, low-charge: {(~ch).sum()}) ===")

    pred = loo_pred(rows, PROD)
    print("\n--- 1. THE GAP: production model, leave-one-complex-out ---")
    for lab, mask in [("ALL", np.ones(len(rows), bool)), ("charged |Q|>=2", ch), ("low-charge", ~ch)]:
        r, rmse, slope = stat(pred[mask], y[mask])
        print(f"  {lab:<16} Pearson={r:+.3f}  RMSE={rmse:.2f}  range-slope={slope:.2f}  (n={mask.sum()})")
    print("  (range-slope <1 = predictions span less than true ΔG = regression dilution.)")

    resid = y - pred
    feats = (PROD + ["net_dewet", "polar_desolv", "chg_burial", "hyd_shield", "chg_compl",
                     "hyd_chg_balance", "n_buried_chg", "n_unsatisfied", "unsat_per_L",
                     "satisfaction_frac", "net_satisfied", "coul_per_L", "gbpol_per_L",
                     "net_elec_per_L", "vdw", "charged_frac"])

    def col(f, mask):
        x = np.array([r.get(f, np.nan) for r in rows], float)
        return x[mask]

    print("\n--- 2. FLIP TABLE: Pearson(feature, true ΔG)  charged vs low-charge ---")
    print(f"{'feature':<18}{'charged':>9}{'low-chg':>9}   pattern")
    for f in feats:
        xc, yc = col(f, ch), y[ch]
        xl, yl = col(f, ~ch), y[~ch]
        mc = ~np.isnan(xc); ml = ~np.isnan(xl)
        rc = pearsonr(xc[mc], yc[mc])[0] if mc.sum() > 4 and np.std(xc[mc]) > 0 else np.nan
        rl = pearsonr(xl[ml], yl[ml])[0] if ml.sum() > 4 and np.std(xl[ml]) > 0 else np.nan
        if np.isnan(rc) or np.isnan(rl):
            pat = "n/a charged" if np.isnan(rc) else ""
        elif rc * rl < 0 and min(abs(rc), abs(rl)) > 0.12:
            pat = "FLIP"
        elif abs(rc) < 0.12 and abs(rl) > 0.2:
            pat = "WASHES on charged"
        elif rc * rl > 0 and min(abs(rc), abs(rl)) > 0.15:
            pat = "stable"
        else:
            pat = ""
        print(f"  {f:<16}{rc:>+9.3f}{rl:>+9.3f}   {pat}")

    print("\n--- 3. RESIDUAL AUTOPSY: Pearson(feature, true-pred) on CHARGED (what predicts our error) ---")
    rr = []
    for f in feats:
        x = col(f, ch); res = resid[ch]
        m = ~(np.isnan(x) | np.isnan(res))
        if m.sum() > 5 and np.std(x[m]) > 0:
            rr.append((f, pearsonr(x[m], res[m])[0], spearmanr(x[m], res[m]).statistic))
    rr.sort(key=lambda t: -abs(t[1]))
    print(f"{'feature':<18}{'Pearson':>9}{'Spearman':>10}")
    for f, rp, rs in rr[:10]:
        print(f"  {f:<16}{rp:>+9.3f}{rs:>+10.3f}")
    print("  (a feature strongly correlated with the residual = a missing-physics direction we could add.)")

    # 4. does the top residual feature actually improve charged LOO?
    top = rr[0][0]
    base_r = stat(loo_pred([r for r, c in zip(rows, ch) if c], PROD), y[ch])[0]
    aug_r = stat(loo_pred([r for r, c in zip(rows, ch) if c], PROD + [top]), y[ch])[0]
    print(f"\n--- 4. add top residual feature '{top}' to charged-only LOO ---")
    print(f"  charged LOO Pearson: PROD={base_r:+.3f}  PROD+{top}={aug_r:+.3f}  (Δ={aug_r-base_r:+.3f})")
    print("  >> if Δ≈0, the residual is unstructured = irreducible single-pose floor (needs MD/FEP, not a feature).")


if __name__ == "__main__":
    main()
